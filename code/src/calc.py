import copy
import scipy.stats as sps
import numpy as np

from src.directions import create_random_direction
from src.directions import init_from_params
from src.directions import inplace_sum_models
from src.utils import create_losses_func

from tqdm.auto import tqdm
from src.directions import calc_sum_models


class LossCalculator:
    def __init__(self, model, loader, criterion):
        self.model = model
        self.calc_losses_func = create_losses_func(loader, criterion)

    def calc_losses(self, coef_grid, direction_norm):
        """
            grid: 2d from [-1, 1]*[-1, 1]
        """

        result = {}
        direction1 = create_random_direction(self.model, external_factor=direction_norm)
        direction2 = create_random_direction(self.model, external_factor=direction_norm)

        for coef1, coef2 in tqdm(coef_grid):
            target_add = [p1 * coef1 + p2 * coef2 for p1, p2 in zip(direction1, direction2)]
            target_add_model = copy.deepcopy(self.model)
            init_from_params(target_add_model, target_add)

            target_model = calc_sum_models(self.model, target_add_model, 1, 1)
            losses = self.calc_losses_func(target_model)
            result[(coef1, coef2)] = losses
        return result


class DeltaCalculator:
    r"""
        Base class for calc delta
        delta_k = (L_{k+1} - L_k)
    """

    def __init__(self, model, loader, criterion):
        self.model = model
        self.calc_losses_func = create_losses_func(loader, criterion)
        self.directions = None

    def calc_shifted_losses(self, mode, mode_params):
        target_model = None
        if mode == 'random-subspace-proj':
            if self.directions is None:
                self.directions = [create_random_direction(self.model, external_factor=1.0) for _ in
                                   range(mode_params['dim'])]
            coefs = list(sps.norm(np.zeros(mode_params['dim']), mode_params['sigma']).rvs())
            target_add_params = [coef * d[i] for coef, d in zip(coefs, self.directions) for i in
                                 range(len(self.directions[0]))]

            target_model = copy.deepcopy(self.model)
            init_from_params(target_model, target_add_params)

            target_model = inplace_sum_models(target_model, self.model, 1.0, 1.0)

        return self.calc_losses_func(target_model)

    def calc_diff_lists(self, mode, mode_params, num_samples=10):
        """
            mode: string
            mode_params: dict, params to esim delta in appropriate mode

            returns: np.ndarray of shape (num_samples, L)
            L - number of batches
        """

        diff_lists = []
        for _ in tqdm(np.arange(1, num_samples + 1)):
            diff_lists.append(
                self.calc_differences(
                    self.calc_shifted_losses(mode, mode_params)
                )
            )
        return np.array(diff_lists)

    def calc_deltas(self, mode, mode_params, num_samples=10):
        diff_lists = self.calc_diff_lists(mode, mode_params, num_samples)
        if mode_params['estim_func'] == 'square':
            diff_lists = diff_lists ** 2
        elif mode_params['estim_func'] == 'abs':
            diff_lists = np.abs(diff_lists)
        elif mode_params['estim_func'] is None:
            pass
        deltas = np.mean(diff_lists, axis=0)
        return deltas

    @staticmethod
    def calc_differences(array):
        cum_mean = np.cumsum(array) / np.arange(1, len(array) + 1)
        diffs = cum_mean[1:] - cum_mean[:-1]
        return diffs
