import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from utils import init_model, init_optimizer, init_scheduler


class Client:
    def __init__(self, args, id, model_type, train_dataset=None, test_dataset=None):
        self.args = args
        self.id = id
        self.name = 'client{}'.format(id)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = args.device
        self.batch_size = args.batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = args.E1
        self.dis_epochs = args.E3
        self.lr = args.lr_k
        self.model_type = model_type
        self.optimizer_type = args.optimizer
        self.scheduler_type = args.scheduler
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.temperature = args.temperature

        self.model = init_model(model_type, args.num_classes, self.name)

    def fork(self, ensemble_parameters):  # for FedAvg
        self.model.load_state_dict(ensemble_parameters)

    def local_update(self):
        self.model.to(self.device).train()

        train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size)
        test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size)
        optimizer = init_optimizer(self.optimizer_type, self.model, self.lr, self.weight_decay, self.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer, self.epochs)

        best_acc = -1
        for epoch in range(1, self.epochs + 1):
            # Training
            train_loss = 0
            train_acc = 0
            for step, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                train_acc += accuracy_score(y_pred=preds.detach().cpu(), y_true=labels.detach().cpu())
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)

            scheduler.step()

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
                torch.save(self.model.state_dict(), '{}/{}-{}.pkl'.format(self.args.model_dir, self.name, self.model_type))  # runs/cifar10/FedAvg/model/client0-resnet8.pkl
            logging.info("Epoch: {}/{}\tTrain Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, *Best Acc: {:.4f}".format(epoch, self.epochs, train_loss, train_acc, test_loss, test_acc, best_acc))

        self.model.cpu()

    def compute_logits(self, x):
        self.model.to(self.device)

        with torch.no_grad():
            logits = self.model(x)

        self.model.cpu()
        return logits

    def local_distillation(self, x, ensemble_logits):
        self.model.to(self.device).train()

        test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size)
        optimizer = init_optimizer(self.optimizer_type, self.model, self.lr, self.weight_decay, self.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer, self.epochs)

        best_acc = -1
        for epoch in range(1, self.dis_epochs + 1):
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
                torch.save(self.model.state_dict(), '{}/{}-{}.pkl'.format(self.args.model_dir, self.name, self.model_type))  # runs/cifar10/FedAvg/model/client0-resnet8.pkl
            logging.info("Epoch: {}/{}\tTrain Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, *Best Acc: {:.4f}".format(epoch, self.dis_epochs, train_loss, train_acc, test_loss, test_acc, best_acc))

        self.model.cpu()
        return best_acc


