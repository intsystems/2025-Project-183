import copy
import torch
import scipy.stats as sps
import numpy as np
from tqdm.auto import tqdm


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


def get_loss_list(model, criterion, dataloader, device):
    model.eval()
    losses = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)

        losses.append(loss.detach().cpu().item())

    return losses


def create_loss_list_func(criterion, dataloader):
    def calc_losses(model):
        return get_loss_list(model, criterion, dataloader, model.device)

    return calc_losses


class LossCalculator:
    def __init__(self, model, criterion, dataloader, core):
        self.model = model
        self.core = core
        self.loss_func = create_loss_list_func(criterion, dataloader)
        self.directions = []

    def calc_losses(self, grid):
        if len(self.directions) < 2:
            self.directions = self.directions + self.core.get(2 - len(self.directions))

        result = {}

        for coef1, coef2 in tqdm(grid):
            target_add = [p1 * coef1 + p2 * coef2 for p1, p2 in zip(self.directions[0], self.directions[1])]
            target_add_model = copy.deepcopy(self.model)
            init_from_params(target_add_model, target_add)

            target_model = calc_sum_models(self.model, target_add_model, 1, 1)
            losses = self.loss_func(target_model)
            result[(coef1, coef2)] = losses

        return result


class DeltaCalculator:
    def __init__(self, model, criterion, dataloader, core):
        self.model = model
        self.core = core
        self.calc_losses_func = create_loss_list_func(criterion, dataloader)
        self.directions = []

    def calc_shifted_losses(self, mode_params):
        if len(self.directions) < mode_params['dim']:
            self.directions = self.directions + self.core.get(mode_params['dim'] - len(self.directions))

        coefs = list(sps.norm(np.zeros(mode_params['dim']), mode_params['sigma']).rvs())
        target_add_params = [coef * d[i] for coef, d in zip(coefs, self.directions) for i in
                             range(len(self.directions[0]))]

        target_model = copy.deepcopy(self.model)
        init_from_params(target_model, target_add_params)
        target_model = inplace_sum_models(target_model, self.model, 1.0, 1.0)

        return self.calc_losses_func(target_model)

    def calc_deltas(self, mode_params, num_samples=10):
        diff_lists = self.calc_diff_lists(mode_params, num_samples)
        diff_lists = diff_lists ** 2
        deltas = np.mean(diff_lists, axis=0)
        return deltas

    def calc_diff_lists(self, mode_params, num_samples=10):
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
                    self.calc_shifted_losses(mode_params)
                )
            )
        return np.array(diff_lists)

    @staticmethod
    def calc_differences(array):
        cum_mean = np.cumsum(array) / np.arange(1, len(array) + 1)
        diffs = cum_mean[1:] - cum_mean[:-1]
        return diffs
