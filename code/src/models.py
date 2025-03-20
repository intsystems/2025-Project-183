# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 20 March 2025, 1:40 AM
#   * File    : models.py
#   * Project : untitled
# --========================================-- #

import torch.nn as nn
import torch.nn.functional as F


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
        # Разворачиваем входные изображения в вектор
        x = x.view(-1, self.input_channels * self.input_sizes[0] * self.input_sizes[1])

        # Проходим по всем слоям, добавляя ReLU после каждого скрытого слоя
        for i, layer in enumerate(self.linears):
            x = layer(x)
            # ReLU для всех слоёв, кроме выходного
            if i < self.layers_num:
                x = F.relu(x)
        return x

    @property
    def device(self):
        return next(iter(self.parameters())).device
