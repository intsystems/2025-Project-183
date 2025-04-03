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


class MaxEigenvector:
    def __init__(self, model, criterion, dataloader):
        self.model = model
        self.loss_func = create_loss_mean_func(criterion, dataloader)

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
