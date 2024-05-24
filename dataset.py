import os

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, TensorDataset

from utils import set_seed


def load_dataset(args):
    if args.dataset == 'mnist':
        train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        args.num_classes = len(train_dataset.classes)
        args.nc = 1
        args.img_size = 28
    elif args.dataset == 'fashionmnist':
        train_dataset = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_dataset = datasets.FashionMNIST(root=args.data_dir, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        args.num_classes = len(train_dataset.classes)
        args.nc = 1
        args.img_size = 28
    elif args.dataset == 'emnist':
        train_dataset = datasets.EMNIST(root=args.data_dir, split='letters', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_dataset = datasets.EMNIST(root=args.data_dir, split='letters', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        args.num_classes = len(train_dataset.classes)
        args.nc = 1
        args.img_size = 28
    elif args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        args.num_classes = len(train_dataset.classes)
        args.nc = 3
        args.img_size = 32
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_dataset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        args.num_classes = len(train_dataset.classes)
        args.nc = 3
        args.img_size = 32
    elif args.dataset == 'svhn':
        train_dataset = datasets.SVHN(root=os.path.join(args.data_dir, 'SVHN'), split="train", download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_dataset = datasets.SVHN(root=os.path.join(args.data_dir, 'SVHN'), split="test", download=True, transform=transforms.Compose([transforms.ToTensor()]))
        args.num_classes = len(np.unique(train_dataset.labels))
        args.nc = 3
        args.img_size = 32
    elif args.dataset == 'food101':
        train_dataset = datasets.Food101(root=args.data_dir, split='train', download=True, transform=transforms.Compose([transforms.Resize(size=(64, 64)), transforms.ToTensor()]))
        test_dataset = datasets.Food101(root=args.data_dir, split='test', download=True, transform=transforms.Compose([transforms.Resize(size=(64, 64)), transforms.ToTensor()]))
        args.num_classes = len(train_dataset.classes)
        args.nc = 3
        args.img_size = 64
    elif args.dataset == 'tinyimagenet':
        train_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'tiny-imagenet-200', 'train'), transform=transforms.Compose([transforms.ToTensor()]))
        test_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'tiny-imagenet-200', 'test'), transform=transforms.Compose([transforms.ToTensor()]))
        args.num_classes = len(train_dataset.classes)
        args.nc = 3
        args.img_size = 64
    else:
        raise ValueError(args.dataset)

    return train_dataset, test_dataset


def partition_dataset(args, train_dataset):
    y_train = np.array(train_dataset.targets)
    if args.partition == 'iid':  # iid distribution
        idxs = np.random.permutation(len(y_train))
        idx_batch = np.array_split(idxs, args.K)
        user_dataidx_map = {k: idx_batch[k] for k in range(1, args.K + 1)}
    elif args.partition == 'dirichlet':  # non-iid with Dirichlet distribution
        idx_batch = [[] for _ in range(args.K)]
        for c in range(args.num_classes):
            idx_c = np.where(y_train == c)[0]
            np.random.shuffle(idx_c)
            proportions = np.random.dirichlet(np.repeat(args.alpha, args.K))
            proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            idx_batch = [idx_c_k + idx.tolist() for idx_c_k, idx in zip(idx_batch, np.split(idx_c, proportions))]
        total = 0
        user_dataidx_map = {}
        for k in range(1, args.K + 1):
            np.random.shuffle(idx_batch[k - 1])
            user_dataidx_map[k] = idx_batch[k - 1]
            total += len(idx_batch[k - 1])
        assert total == len(y_train)
    else:
        raise ValueError(args.partition)
    local_datasets = {}
    for k, dataidx in user_dataidx_map.items():
        local_datasets[k] = Subset(train_dataset, indices=dataidx)

    user_cls_counts = {}
    for k, dataidx in user_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        user_cls_counts[k] = tmp
    del train_dataset, y_train
    assert len(local_datasets) == len(user_cls_counts) == args.K
    return local_datasets, user_cls_counts


if __name__ == '__main__':
    class args:
        data_dir = 'data'
        dataset = 'cifar10'
        batch_size = 256
        K = 10
        partition = 'dirichlet'
        alpha = 1
        seed = 0


    train_dataset, test_dataset = load_dataset(args)
    print(train_dataset.data.shape, len(train_dataset.targets))
    print(test_dataset.data.shape, len(test_dataset.targets))

    set_seed(args.seed)
    local_datasets, user_cls_counts = partition_dataset(args, train_dataset)
    for k in range(1, args.K + 1):
        print(f'{k}: {len(local_datasets[k])}, {user_cls_counts[k]}')

    train_loader = DataLoader(local_datasets[0], shuffle=True, batch_size=args.batch_size)
    print(len(train_loader))
    for batch in train_loader:
        print(batch[0].shape, batch[1].shape)
        break
