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


def train(model, criterion, dataloader, optimizer, num_epochs=10, ret=False, log=False):
    model.train()

    losses = []
    for epoch in tqdm(range(1, num_epochs + 1), desc="Epoch"):
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

        avg_loss = running_loss / batches_count
        if log:
            print(f"Epoch [{epoch}/{num_epochs}]: loss = {avg_loss:.4f}")

    return losses if ret else None


def get_loss(model, criterion, dataloader, device):
    model.eval()
    losses = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        losses.append(loss)
    return torch.stack(losses, dim=0)


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
