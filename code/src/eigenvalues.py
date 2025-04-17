# --========================================-- #
#   * Author  : NTheme - All rights reserved
#   * Updated : 16 April 2025
#   * File    : eigenvalues.py
#   * Project : code
# --========================================-- #

from tqdm.auto import tqdm
import torch


class HessianEigenvector:
    def __init__(self, variables, func):
        self.variables = list(variables)
        self.func = func

        self.eigenvectors = []
        self.eigenvalues = []

    def get(self, k, num_iters=1000, tol=1e-6):
        if k <= len(self.eigenvectors):
            return self.eigenvalues[:k], self.eigenvectors[:k]

        for i in range(0, k - len(self.eigenvectors)):
            eigenvalue, eigenvector = self._power_iteration(num_iters, tol)
            self.eigenvalues.append(eigenvalue.detach().cpu().item())
            self.eigenvectors.append(eigenvector)
        return self.eigenvalues[:k], self.eigenvectors[:k]

    def _power_iteration(self, num_iters, tol):
        v = [torch.randn_like(p) for p in self.variables]
        v = self._normalize(v)

        eigenvalue = None
        for _ in tqdm(range(num_iters), desc="Power iter"):
            hv = self._hvp(self.variables, self.func, v)

            for d in self.eigenvectors:
                hv = self._projection_diff(hv, d)

            new_v = self._normalize(hv)
            eigenvalue = self._dot(new_v, hv)

            if self._norm_diff(new_v, v) < tol:
                v = new_v
                break
            v = new_v

        return eigenvalue, v

    @staticmethod
    def _hvp(variables, func, v):
        grads = torch.autograd.grad(func, variables, create_graph=True)
        dots = [torch.dot(g.contiguous().view(-1), vi.contiguous().view(-1)) for g, vi in zip(grads, v)]
        grad_v = torch.sum(torch.stack(dots))
        hv = torch.autograd.grad(grad_v, variables, retain_graph=True)
        return [h.contiguous() for h in hv]

    @staticmethod
    def _dot(a, b):
        dots = [torch.dot(ai.view(-1), bi.view(-1)) for ai, bi in zip(a, b)]
        return torch.stack(dots).sum()

    def _normalize(self, vec):
        norm = torch.sqrt(self._dot(vec, vec))
        return [vi / norm for vi in vec]

    def _norm_diff(self, a, b):
        diff = [ai - bi for ai, bi in zip(a, b)]
        return torch.sqrt(self._dot(diff, diff))

    def _projection_diff(self, hv, d):
        proj = self._dot(hv, d)
        return [hvi - proj * di for hvi, di in zip(hv, d)]
