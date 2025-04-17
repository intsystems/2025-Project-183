# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 03 April 2025, 7:09 AM
#   * File    : directions_eigen.py
#   * Project : code
# --========================================-- #

import torch
from src.models import get_loss
from src.eigenvalues import HessianEigenvector


class RandomDirection:
    def __init__(self, model, criterion=None, dataloader=None):
        self.model = model
        self.type = "random"

    def get(self, k):
        return [self._create_random_direction(self.model) for _ in range(k)]

    @staticmethod
    def _normalize_direction(direction, weights, norm='filter'):
        """
            Rescale the direction so that it has similar norm as their corresponding
            model in different levels.

            Args:
              direction: a variables of the random direction for one layer
              weights: a variable of the original model for one layer
              norm: normalization method, 'filter' | 'layer' | 'weight'
        """
        if norm == 'filter':
            # Rescale the filters (weights in group) in 'direction' so that each
            # filter has the same norm as its corresponding filter in 'weights'.
            for d, w in zip(direction, weights):
                d.mul_(w.norm() / (d.norm() + 1e-10))
        elif norm == 'layer':
            # Rescale the layer variables in the direction so that each layer has
            # the same norm as the layer variables in weights.
            direction.mul_(weights.norm() / direction.norm())
        elif norm == 'weight':
            # Rescale the entries in the direction so that each entry has the same
            # scale as the corresponding weight.
            direction.mul_(weights)
        elif norm == 'dfilter':
            # Rescale the entries in the direction so that each filter direction
            # has the unit norm.
            for d in direction:
                d.div_(d.norm() + 1e-10)
        elif norm == 'dlayer':
            # Rescale the entries in the direction so that each layer direction has
            # the unit norm.
            direction.div_(direction.norm())

    @staticmethod
    def _normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
        """
            The normalization scales the direction entries according to the entries of weights.
        """
        assert (len(direction) == len(weights))
        for d, w in zip(direction, weights):
            if d.dim() <= 1:
                if ignore == 'biasbn':
                    d.fill_(0)  # ignore directions for weights with 1 dimension
                else:
                    d.copy_(w)  # keep directions for weights/bias that are only 1 per node
            else:
                RandomDirection._normalize_direction(d, w, norm)

    def _create_random_direction(self,
                                 ignore='biasbn',
                                 norm='filter',
                                 external_norm='unit',
                                 external_factor=1.0):
        """
            Setup a random (normalized) direction with the same dimension as
            the weights or states.

            Args:
              net: the given trained model
              norm: direction normalization method, including
                    'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'
              external_norm: external normalization method, including
                    'unit'
              external_factor: linalg norm of result direction

            Returns:
              direction: a random direction with the same dimension as weights or states.
        """

        weights = [p.data for p in self.model.parameters()]  # a list of parameters.
        direction = [torch.randn(w.size(), device=w.device) for w in weights]
        self._normalize_directions_for_weights(direction, weights, norm, ignore)

        squared_norms = torch.stack([d.norm() ** 2 for d in direction])
        full_direction_norm = torch.sqrt(squared_norms.sum())
        if external_norm == 'unit':
            for d in direction:
                d.div_(full_direction_norm)

        for d in direction:
            d.mul_(external_factor)

        return direction


class EigenDirection:
    def __init__(self, model, criterion, dataloader):
        self.model = model
        self.loss = get_loss(model, criterion, dataloader, model.device).mean()
        self.type = "eigen"

    def get(self, k, **kwargs):
        return HessianEigenvector(self.model.parameters(), self.loss).get(k, **kwargs)[1]
