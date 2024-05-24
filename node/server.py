import copy
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from utils import init_model, init_optimizer, init_scheduler, init_generator, DiversityLoss


class Server:
    def __init__(self, args, model_type, generator_type, test_dataset=None, id=0):
        self.args = args
        self.id = id
        self.name = 'server'

        self.test_dataset = test_dataset
        self.device = args.device
        self.batch_size = args.batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = args.E2
        self.lr = args.lr
        self.lr_G = args.lr_G
        self.model_type = model_type
        self.optimizer_type = args.optimizer
        self.scheduler_type = args.scheduler
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.nz = args.nz
        self.nc = args.nc
        self.img_size = args.img_size
        self.num_classes = args.num_classes
        self.synthetic_size = args.synthetic_size
        self.temperature = args.temperature
        self.q = args.q
        self.eps = args.eps
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.beta3 = args.beta3

        self.model = init_model(model_type, self.num_classes, self.name)
        self.generator = init_generator(generator_type, self.nz, self.nc, self.img_size, self.num_classes)

    def select_active_clients(self, K, C):
        m = max(math.ceil(C * K), 1)
        selected_clients = sorted(np.random.choice(range(1, K + 1), m, replace=False))
        return selected_clients

    def merge(self, local_parameters, local_weights=None):  # for FedAvg
        ensemble_parameters = copy.deepcopy(local_parameters[0])
        for n in ensemble_parameters.keys():
            ensemble_parameters[n] = 0.
            for k in range(len(local_parameters)):
                if local_weights is not None:
                    ensemble_parameters[n] += local_weights[k] * local_parameters[k][n]
                else:
                    ensemble_parameters[n] += (1 / len(local_parameters)) * local_parameters[k][n]
        return ensemble_parameters

    def generate_data(self):
        self.generator.to(self.device).train()

        z = torch.randn(self.synthetic_size, self.nz).to(self.device)
        y = torch.randint(low=0, high=self.num_classes, size=(self.synthetic_size,))
        y = y.sort()[0]
        y = y.to(self.device)
        x = self.generator(z, y)

        self.generator.cpu()
        return z, y, x

    def construct_queries(self, x):
        x_queries = []
        u_norm_queries = []
        for _ in range(self.q):
            u = torch.randn(x.shape).to(self.device)
            u_flat = u.view([self.synthetic_size, -1])
            u_norm = u / torch.norm(u_flat, dim=1).view([-1, 1, 1, 1])
            x_mod = x + self.eps * u_norm  # evaluation point
            x_queries.append(x_mod)
            u_norm_queries.append(u_norm)
        return x_queries, u_norm_queries

    def ensemble_logits(self, local_logits, local_weights=None):
        ensemble_logits = 0
        for k in range(len(local_logits)):
            if local_weights is not None:
                ensemble_logits += local_weights[k] * local_logits[k]
            else:
                ensemble_logits += (1 / len(local_logits)) * local_logits[k]
        return ensemble_logits

    def global_model_update(self, x, ensemble_logits):
        self.model.to(self.device).train()

        test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size)
        optimizer = init_optimizer(self.optimizer_type, self.model, self.lr, self.weight_decay, self.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer, self.epochs)

        best_acc = -1
        for epoch in range(1, self.epochs + 1):
            # Training
            optimizer.zero_grad()
            logits = self.model(x.detach())
            loss = F.kl_div(F.log_softmax(logits / self.temperature, dim=-1), F.softmax(ensemble_logits / self.temperature, dim=-1), reduction='batchmean') * (self.temperature ** 2)
            loss.backward()
            optimizer.step()

            scheduler.step()

            train_loss = loss.item()
            preds = torch.argmax(logits, dim=-1)
            labels = torch.argmax(ensemble_logits, dim=-1)
            train_acc = accuracy_score(y_pred=preds.detach().cpu(), y_true=labels.detach().cpu())

            # Testing
            test_loss = 0
            test_acc = 0
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    logits = self.model(images)
                loss = self.criterion(logits, labels)

                test_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                test_acc += accuracy_score(y_pred=preds.detach().cpu(), y_true=labels.detach().cpu())
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), '{}/{}-{}.pkl'.format(self.args.model_dir, self.name, self.model_type))
            logging.info("Epoch: {}/{}\tTrain Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, *Best Acc: {:.4f}".format(epoch, self.epochs, train_loss, train_acc, test_loss, test_acc, best_acc))

        self.model.cpu()
        return best_acc

    def generator_update(self, x, ensemble_logits, x_queries, ensemble_logits_queries, u_norm_queries, y, z):
        self.generator.to(self.device).train()
        self.model.to(self.device).eval()

        optimizer = init_optimizer(self.optimizer_type, self.generator, self.lr_G, self.weight_decay, self.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer, self.epochs)

        optimizer.zero_grad()
        # Forward Finite Differences
        grad_est = torch.zeros_like(x)
        d = np.array(x.shape[1:]).prod()
        with torch.no_grad():
            logits = self.model(x)
            # 1. fidelity loss
            loss_fid = self.criterion(ensemble_logits, y)
            # 2. adversarial loss
            loss_adv = - F.kl_div(F.log_softmax(logits / self.temperature, dim=1), F.softmax(ensemble_logits / self.temperature, dim=1), reduction='batchmean')
            # 3. diversity loss
            loss_div = DiversityLoss()(z, x)
            # 4. information entropy loss
            ensemble_probs = torch.softmax(ensemble_logits, dim=1).mean(dim=0)
            loss_info = (ensemble_probs * torch.log(ensemble_probs)).sum()
            # total loss for the generator
            loss = loss_fid + self.beta1 * loss_adv + self.beta2 * loss_div + self.beta3 * loss_info
            for x_q, ensemble_logits_q, u_norm in zip(x_queries, ensemble_logits_queries, u_norm_queries):
                logits_q = self.model(x_q)
                # 1. fidelity loss
                loss_fid_q = self.criterion(ensemble_logits_q, y)
                # 2. adversarial loss
                loss_adv_q = - F.kl_div(F.log_softmax(logits_q / self.temperature, dim=1), F.softmax(ensemble_logits_q / self.temperature, dim=1), reduction='batchmean')
                # 3. diversity loss
                loss_div_q = DiversityLoss()(z, x_q)
                # 4. information entropy loss
                ensemble_probs_q = torch.softmax(ensemble_logits_q, dim=1).mean(dim=0)
                loss_info_q = (ensemble_probs_q * torch.log(ensemble_probs)).sum()
                loss_q = loss_fid_q + self.beta1 * loss_adv_q + self.beta2 * loss_div_q + self.beta3 * loss_info_q
                grad_est += ((d / self.q) * (loss_q - loss) / self.eps).view([-1, 1, 1, 1]) * u_norm
        grad_est /= self.synthetic_size
        x.backward(grad_est)
        optimizer.step()

        scheduler.step()

        self.generator.cpu()
        self.model.cpu()
