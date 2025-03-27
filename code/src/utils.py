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
from tqdm.auto import tqdm


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


def train(model,
          optimizer,
          criterion,
          dataloader,
          num_epochs,
          device, log=False):
    model.train()

    losses = []
    for epoch in tqdm(np.arange(1, num_epochs + 1)):
        running_loss = 0.0
        batches_count = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().item())

            batches_count += 1
            running_loss += losses[-1]

        avg_loss = running_loss / batches_count
        if log:
            print(f"Epoch [{epoch + 1}/{num_epochs}]: loss = {avg_loss:.4f}")

    return losses


def valid(model,
          criterion,
          dataloader,
          device):
    model.eval()

    acc = []
    losses = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)

        losses.append(loss.detach().cpu().item())
        acc.append(torch.mean((torch.argmax(outputs, dim=-1) == y).to(torch.float)).cpu().item())

    return losses, acc


def create_losses_func(dataloader, criterion):
    def calc_losses(model):
        losses, _ = valid(
            model,
            criterion,
            dataloader,
            model.device)
        return losses

    return calc_losses


def valid_loss(model, criterion, dataloader, device):
    """
    Вычисляет среднее значение loss по даталоадеру без отсоединения от графа вычислений.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        total_loss += loss
        count += 1
    return total_loss / count


def create_loss_func(dataloader, criterion):
    """
    Возвращает функцию, вычисляющую loss по даталоадеру для модели.
    Предполагается, что у модели есть атрибут device.
    """

    def calc_loss(model):
        return valid_loss(model, criterion, dataloader, model.device)

    return calc_loss
