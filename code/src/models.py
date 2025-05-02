# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 20 March 2025, 1:40 AM
#   * File    : models.py
#   * Project : untitled
# --========================================-- #

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from src.eigenvalues import HessianEigenvector
import numpy as np


class MLP(nn.Module):
    def __init__(self, layers_num, hidden, input_channels, input_sizes, classes):
        """
        layers_num: количество скрытых слоёв
        hidden: число нейронов в каждом скрытом слое
        input_channels: число каналов во входном изображении
        input_sizes: (height, width)
        classes: число выходных классов
        """
        super(MLP, self).__init__()

        self.layers_num = layers_num
        self.hidden = hidden
        self.input_channels = input_channels
        self.input_sizes = input_sizes
        self.classes = classes

        input_dim = input_channels * input_sizes[0] * input_sizes[1]

        layers = [nn.Linear(input_dim, hidden)]
        for _ in range(layers_num - 1):
            layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.Linear(hidden, classes))

        self.linears = nn.ModuleList(layers)

    def forward(self, x):
        x = x.view(-1, self.input_channels * self.input_sizes[0] * self.input_sizes[1])

        for i, layer in enumerate(self.linears):
            x = layer(x)

            if i < self.layers_num:
                x = F.relu(x)
        return x

    @property
    def device(self):
        return next(iter(self.parameters())).device


class ConvBlock(nn.Module):
    r"""
        Convolutional block

        Args:
            c_in, c_out - input/output number of channels
            kernel_size - size of kernel
    """

    def __init__(self, c_in, c_out, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out,
                      kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.body(x)


def _calc_flattened_size(channels_list, ker_size_list, input_sizes, padding):
    ker_delta = np.sum(np.array(ker_size_list) - 1)
    pad_delta = padding * 2 * (len(ker_size_list) - 1)
    final_sizes = (input_sizes[0] - ker_delta + pad_delta, input_sizes[1] - ker_delta + pad_delta)
    return final_sizes[0] * final_sizes[1] * channels_list[-1]


def _stack_conv_blocks(channels_list, ker_size_list, padding):
    convs_list = []
    for c_in, c_out, kernel_size in zip(channels_list[:-2], channels_list[1:-1], ker_size_list[:-1]):
        convs_list.append(ConvBlock(c_in, c_out, kernel_size, padding))
    convs_list.append(nn.Conv2d(channels_list[-2], channels_list[-1], ker_size_list[-1], padding))
    return nn.Sequential(*convs_list)


class ConvNet(nn.Module):
    r"""
    Convolutional network

    Args:
        channels_list list(int): list of layers number of channels
        input_sizes (int, int): width, height of input images
        classes (int): number of classes for classification

    """

    def __init__(self, layers_num,
                 hidden_channels,
                 kernel_size,
                 padding,
                 input_channels,
                 output_channels,
                 input_sizes,
                 classes):
        super(ConvNet, self).__init__()

        channels_list = [input_channels] + [hidden_channels for _ in range(layers_num)] + [output_channels]
        ker_size_list = [kernel_size for _ in channels_list[:-1]]
        self.model = _stack_conv_blocks(channels_list, ker_size_list, padding)
        self.flatten = nn.Flatten()

        flat_size = _calc_flattened_size(channels_list, ker_size_list, input_sizes, padding)
        self.head = nn.Sequential(
            nn.Linear(flat_size, classes)
        )

    def forward(self, images):
        out = self.model(images)
        flattened = self.flatten(out)
        logits = self.head(flattened)
        return logits

    @property
    def device(self):
        return next(iter(self.parameters())).device


def train(model, criterion, dataloader, optimizer, num_epochs=10, ret=False, log=False):
    model.train()

    losses = []
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        batches_count = 0

        for x, y in dataloader:
            x, y = x.to(model.device), y.to(model.device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().item())

            batches_count += 1
            running_loss += losses[-1]

        if log:
            avg_loss = running_loss / batches_count
            print(f"Epoch [{epoch}/{num_epochs}]: loss = {avg_loss:.4f}")

    return losses if ret else None


def get_loss(model, criterion, dataloader, device, border=None):
    model.eval()
    losses = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        losses.append(loss)
        if border is not None and len(losses) >= border:
            break
    return torch.stack(losses, dim=0)


def hessian_theoretical(eigen_matrix):
    diff = eigen_matrix[1:] - eigen_matrix[:-1]
    return (2 * np.square(diff).sum(axis=-1) + np.square(diff.sum(axis=-1))) / 4


def estimate_train(model, criterion, dataloader, optimizer, delta, directions=2,
                   num_epochs=10, ret=False, log=False):
    model.train()
    losses = []

    for x, y in dataloader:
        running_loss = 0.0

        for _ in range(1, num_epochs + 1):
            x, y = x.to(model.device), y.to(model.device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().item())

            running_loss += losses[-1]

        delta_e = None
        if len(losses) // num_epochs > 1:
            loss_iter = get_loss(model, criterion, dataloader, model.device, len(losses) // num_epochs)
            func_iter = torch.cumsum(loss_iter, dim=0) / torch.arange(1, loss_iter.size(0) + 1, device=model.device)
            last = HessianEigenvector(model.parameters(), func_iter[len(losses) // num_epochs - 2]).get(directions)[0]
            curr = HessianEigenvector(model.parameters(), func_iter[len(losses) // num_epochs - 1]).get(directions)[0]
            delta_e = hessian_theoretical(np.vstack((last, curr)))[0]

        if log:
            avg_loss = running_loss / num_epochs
            print(f"Batch [{len(losses) // num_epochs}/{len(dataloader)}]: loss = {avg_loss:.4f}, "
                  f"delta = {1e9 if delta_e is None else delta_e:.4f}")

        if delta_e is not None and delta_e < delta:
            break

    return losses if ret else None


def inplace_sum_models(model1, model2, coef1, coef2):
    final = model1
    for (name1, param1), (name2, param2) in zip(final.state_dict().items(), model2.state_dict().items()):
        transformed_param = param1 * coef1 + param2 * coef2
        param1.copy_(transformed_param)
    return final


def calc_sum_models(model1, model2, coef1, coef2):
    final = copy.deepcopy(model1)
    final.load_state_dict(copy.deepcopy(model1.state_dict()))
    return inplace_sum_models(final, model2, coef1, coef2)


def init_from_params(model, direction):
    for p_orig, p_other in zip(model.parameters(), direction):
        with torch.no_grad():
            p_orig.copy_(p_other)
