# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Created : 03 April 2025, 7:09 AM
#   * File    : directions_eigen.py
#   * Project : code
# --========================================-- #

import torch
from tqdm.auto import tqdm


def get_loss_mean(model, criterion, dataloader, device):
    model.eval()

    count = 0
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)

        total_loss += loss
        count += 1

    return total_loss / count


def create_loss_mean_func(criterion, dataloader):
    def calc_loss(model):
        return get_loss_mean(model, criterion, dataloader, model.device)

    return calc_loss


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


class MaxEigenvector:
    def __init__(self, model, criterion, dataloader):
        self.model = model
        self.loss_func = create_loss_mean_func(criterion, dataloader)
        self.type = "eigen"

        self.eigenvectors = []
        self.eigenvalues = []

    def get(self, k, num_iters=100, tol=1e-6):
        loss = self.loss_func(self.model)

        if k <= len(self.eigenvectors):
            return self.eigenvalues[:k], self.eigenvectors[:k]

        for i in range(0, k - len(self.eigenvectors)):
            eigenvalue, eigenvector = self._power_iteration(loss, num_iters=num_iters, tol=tol)
            self.eigenvalues.append(eigenvalue)
            self.eigenvectors.append(eigenvector)
        return self.eigenvectors

    def _power_iteration(self, loss, num_iters=100, tol=1e-6):
        params = list(self.model.parameters())
        flat_params = torch.cat([p.view(-1) for p in params])

        v = torch.randn(flat_params.shape[0], device=self.model.device)
        v = v / v.norm()
        eigenvalue = None
        for _ in tqdm(range(1, num_iters + 1)):
            hv = self._hvp(params, loss, v)

            for d in self.eigenvectors:
                hv = hv - torch.dot(hv, d) * d
            new_v = hv / hv.norm()
            eigenvalue = torch.dot(new_v, hv)
            if torch.norm(new_v - v) < tol:
                break
            v = new_v
        return eigenvalue, v

    @staticmethod
    def _hvp(model_params, loss, v):
        grads = torch.autograd.grad(loss, model_params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grads])
        grad_v = torch.dot(flat_grad, v)
        hv = torch.autograd.grad(grad_v, model_params, retain_graph=True)
        hv_flat = torch.cat([h.contiguous().view(-1) for h in hv])
        return hv_flat
