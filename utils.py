import logging
import os
import random

import numpy as np
import torch
from torch import nn

from model.resnet import resnet18, resnet34, resnet50
from model.generator import Generator


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_model(model_type, num_classes, name):
    if model_type == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif model_type == 'resnet34':
        model = resnet34(num_classes=num_classes)
    elif model_type == 'resnet50':
        model = resnet50(num_classes=num_classes)
    else:
        raise ValueError(model_type)
    logging.info('Model parameters of %s_%s: %2.2fM' % (name, model_type, (sum(p.numel() for p in model.parameters()) / (1000 * 1000))))
    logging.info('Model size of %s_%s: %2.2fM' % (name, model_type, (sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024))))
    return model


def init_optimizer(optimizer_type, model, lr, weight_decay=0., momentum=0.):
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    else:
        raise ValueError(optimizer_type)
    return optimizer


def init_scheduler(scheduler_type, optimizer, epochs):
    if scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    elif scheduler_type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.1, 0.3, 0.5] * epochs, gamma=0.3)
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        raise ValueError(scheduler_type)
    return scheduler


def init_generator(generator_type, nz, nc, img_size, num_classes):
    if generator_type == 'generator':
        generator = Generator(nz=nz, nc=nc, img_size=img_size, conditional=False)
    elif generator_type == 'conditional_generator':
        generator = Generator(nz=nz, nc=nc, img_size=img_size, num_classes=num_classes, conditional=True)
    else:
        raise ValueError(generator_type)
    return generator


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self):
        super().__init__()

    def pairwise_distance(self, tensor):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))

    def forward(self, noises, images):
        if len(images.shape) > 2:
            images = images.view((images.size(0), -1))
        images_dist = self.pairwise_distance(images)
        noises_dist = self.pairwise_distance(noises)
        return torch.exp(torch.mean(-images_dist * noises_dist))


