# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 20 March 2025, 1:33 AM
#   * File    : utils.py
#   * Project : untitled
# --========================================-- #

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np


def init_dataloader(dataset_name,
                    transform,
                    batch_size=64,
                    dataset_load_path='data/',
                    train_mode=True,
                    size=None):
    if dataset_load_path[-1] != '/':
        dataset_load_path += '/'
    dataset_name = dataset_name
    dataset_load_path = dataset_load_path + dataset_name + '_dataset'

    name2dataset = {
        'MNIST': datasets.MNIST,
        'FashionMNIST': datasets.FashionMNIST,
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100,
    }

    dataset = name2dataset[dataset_name](dataset_load_path,
                                         download=True,
                                         train=train_mode,
                                         transform=transform)
    if size is not None and size != -1:
        dataset = torch.utils.data.Subset(dataset, np.arange(size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train_mode, num_workers=2)
    return loader
