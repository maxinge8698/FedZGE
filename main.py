import argparse
import logging
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

from dataset import load_dataset, partition_dataset

from node.server import Server
from node.client import Client
from utils import set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Federated
    parser.add_argument('--algorithm', default='FedZGE', type=str, help='Type of algorithms:{FedZGE}')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for initialization')
    parser.add_argument('--K', default=10, type=int, help='Number of clients: K')
    parser.add_argument('--C', default=1, type=float, help='Fraction of clients: C')
    parser.add_argument('--R', default=100, type=int, help='Number of communication rounds: R')
    parser.add_argument('--E1', default=10, type=int, help='Number of local update epochs: E')
    parser.add_argument('--E2', default=10, type=int, help='Number of server update epochs: E')
    parser.add_argument('--E3', default=10, type=int, help='Number of local distillation epochs: E')
    # Data
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, help='Type of datasets: {cifar10, cifar100}')
    parser.add_argument('--batch_size', default=256, type=int, help='input batch size')
    parser.add_argument('--partition', default='dirichlet', type=str, help='iid data or non-iid data with Dirichlet distribution')
    parser.add_argument('--alpha', default=1, type=float, help='ratio of Dirichlet distribution')
    # Model
    parser.add_argument('--global_model', default='resnet18', type=str, help='Type of global model: {resnet18, resnet34, resnet50}')
    parser.add_argument('--local_models', default='resnet18', type=str, help='Type of local model: {resnet18, resnet34, resnet50}')
    # Optimization
    parser.add_argument('--optimizer', default='adam', type=str, help='Type of optimizer: {sgd, adam}')
    parser.add_argument('--scheduler', default='cosine', type=str, help='Type of scheduler: {cosine}')
    parser.add_argument('--lr_k', default=0.01, type=float, help='Learning rate of the local model')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help="Weight decay if we apply")
    # Generator
    parser.add_argument('--generator', default='conditional_generator', type=str, help='Type of generator: {generator, conditional_generator}')
    parser.add_argument('--nz', default=100, type=int, help='dimensionality of random noise')
    parser.add_argument('--lr_G', default=0.001, type=float, help='learning rate of the generator')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate of the global model')
    parser.add_argument('--synthetic_size', default=500, type=int, help='Number of synthetic data per round')
    parser.add_argument('--temperature', default=5, type=float, help='Temperature')
    parser.add_argument('--beta1', default=1, type=float, help='Scaling factor for adversarial loss')
    parser.add_argument('--beta2', default=1, type=float, help='Scaling factor for diversity loss')
    parser.add_argument('--beta3', default=1, type=float, help='Scaling factor for information entropy loss')
    # Gradient Approximation
    parser.add_argument("--q", default=10, type=int, help="Number of random perturbation directions")
    parser.add_argument("--eps", default=1e-3, type=float, help="Smoothing parameter")
    # Output
    parser.add_argument("--output_dir", default="runs", type=str, help="The output directory where checkpoints/results/logs will be written.")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ser dir
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.algorithm)
    args.model_dir = os.path.join(args.output_dir, 'model')
    args.accuracy_dir = os.path.join(args.output_dir, 'accuracy')
    args.log_dir = os.path.join(args.output_dir, 'log')
    args.img_dir = os.path.join(args.output_dir, 'synthetic')
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.accuracy_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)

    # Set log
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[%(levelname)s](%(asctime)s) %(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.log_dir, 'log.txt')), logging.StreamHandler(sys.stdout)])

    # Set data
    train_dataset, test_dataset = load_dataset(args)
    local_datasets, user_cls_counts = partition_dataset(args, train_dataset)

    # Set model
    global_model = args.global_model
    local_models = args.local_models.split(',')
    if len(local_models) == 1:
        local_models = [local_models[0]] * args.K
    server = Server(args, id=0, model_type=global_model, generator_type=args.generator, test_dataset=test_dataset)
    clients = {k: Client(args, id=k, model_type=local_models[k - 1], train_dataset=local_datasets[k], test_dataset=test_dataset) for k in range(1, args.K + 1)}

    # Federated training
    logger.info("Algorithm: {}".format(args.algorithm))
    logger.info("Device: {}".format(args.device))
    logger.info("Dataset: {}".format(args.dataset))
    if args.partition == 'iid':
        logger.info("Partition: {}".format(args.partition))
    elif args.partition == 'dirichlet':
        logger.info("Partition: {}, Alpha: {}".format(args.partition, args.alpha))
    logger.info('Number of train datasets: {}'.format([len(local_datasets[k]) for k in range(1, args.K + 1)]))
    logger.info('Number of test dataset: {}'.format(len(test_dataset)))
    logger.info('Data statistics: %s' % str(user_cls_counts))
    logger.info("Number of clients: {}".format(args.K))
    logger.info("Number of communication rounds: {}".format(args.R))
    logger.info("Number of local training epochs: {}".format(args.E1))
    logger.info("Number of local distillation epochs: {}".format(args.E2))
    logger.info("Global model: {},\tLocal models: {}".format(global_model, local_models))

    current_accuracies = {k: None for k in range(args.K + 1)}
    test_accuracies = pd.DataFrame(columns=range(args.K + 1))
    communication_budgets = 0
    for t in range(1, args.R + 1):
        logger.info('===============The {:d}-th round==============='.format(t))

        ###################
        # ServerExecute() #
        ###################
        # the server randomly samples m = max(C*K, 1) partial clients.
        selected_clients = server.select_active_clients(args.K, args.C)

        ###################
        # ClientExecute() #
        ###################
        logger.info('#################### Local Update ####################')
        local_data_sizes = []
        for k in selected_clients:
            client = clients[k]
            logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

            """ Local Update: θ_k^t ← ClientUpdate(D_k; θ_k^t-1) """
            client.local_update()

            local_data_sizes.append(len(client.train_dataset))  # get the quantity of clients joined in the FL train for updating the clients weights
        local_weights = [local_data_size / sum(local_data_sizes) for local_data_size in local_data_sizes]

        ###################
        # ServerExecute() #
        ###################
        logger.info('#################### Server Update ####################')
        logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))

        # the server synthesize data samples
        z, y, x = server.generate_data()
        #################################
        # save synthetic images
        if isinstance(x, torch.Tensor):
            imgs = (x.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
        for idx, img in enumerate(imgs):
            if img.shape[0] == 1:
                img = Image.fromarray(img[0])
            else:
                img = Image.fromarray(img.transpose(1, 2, 0))  # (C,H,W) -> (H,W,C)
            img.save(os.path.join(args.img_dir, f'{idx}.png'))
        #################################

        # the server construct perturbed data samples along q random directions
        x_queries, u_norm_queries = server.construct_queries(x)

        # obtain the outputs of local models in a black-box manner
        local_logits = []
        for k in selected_clients:
            client = clients[k]
            logits = client.compute_logits(x)
            communication_budgets += x.numel() * x.element_size()
            local_logits.append(logits)
            communication_budgets += logits.numel() * logits.element_size()
        # ensemble
        ensemble_logits = server.ensemble_logits(local_logits, local_weights)

        ensemble_logits_queries = []
        for x_q in x_queries:
            local_logits_q = []
            for k in selected_clients:
                client = clients[k]
                logits_q = client.compute_logits(x_q)
                communication_budgets += x_q.numel() * x_q.element_size()
                local_logits_q.append(logits_q)
                communication_budgets += logits_q.numel() * logits_q.element_size()
            # ensemble
            ensemble_logit_q = server.ensemble_logits(local_logits_q, local_weights)
            ensemble_logits_queries.append(ensemble_logit_q)

        """ Generator Update: θ_G^t ← θ_G^t-1 - η_G * ▽θ L_G(x;θ_G^t-1) """
        server.generator_update(x, ensemble_logits, x_queries, ensemble_logits_queries, u_norm_queries, y, z)

        """ Global Model Update: θ^t ← θ^t-1 - η * ▽θ L_f(x;θ^t-1) """
        server_acc = server.global_model_update(x, ensemble_logits)
        current_accuracies[0] = '{:.2f}'.format(server_acc)

        ###################
        # ClientExecute() #
        ###################
        logger.info('#################### Local Distillation ####################')
        for k in selected_clients:
            client = clients[k]
            logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

            """ Local Distillation: θ_k^t ← θ_k^t - η_k * ▽θ_k L_k'(x;θ_k^t) """
            client_acc = client.local_distillation(x, ensemble_logits)
            current_accuracies[k] = '{:.2f}'.format(client_acc)

            communication_budgets += ensemble_logits.numel() * ensemble_logits.element_size()

        logger.info('Communication budgets: {:.2f}GB'.format(communication_budgets / (1024 * 1024 * 1024)))
        test_accuracies.loc[len(test_accuracies)] = current_accuracies
        test_accuracies.to_csv(os.path.join(args.accuracy_dir, 'test_accuracy.csv'))
