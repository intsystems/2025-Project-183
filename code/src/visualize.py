import copy
import itertools
import numpy as np
from tqdm.auto import tqdm

from matplotlib import cm
import matplotlib.pyplot as plt


class LossVisualizer:
    def __init__(self, calculator, grid_step=0.1):
        """
            grid_step in [0, 1]
        """
        self.core_type = calculator.core.type
        self.grid_step = grid_step
        self.grid_loss = calculator.calc_losses(
            list(itertools.product(np.arange(-1, 1 + self.grid_step, step=self.grid_step),
                                   np.arange(-1, 1 + self.grid_step, step=self.grid_step))))

    def _set_xy_grid(self, x_grid_bounds, y_grid_bounds):
        xs = np.arange(-1, 1 + self.grid_step, step=self.grid_step)
        ys = xs
        xs = np.array([x for x in xs if x_grid_bounds[1] >= x >= x_grid_bounds[0]])
        ys = np.array([y for y in ys if y_grid_bounds[1] >= y >= y_grid_bounds[0]])
        return xs, ys

    def visualize_all(self, size1, size2,
                      x_grid_bounds=(-1, 1), y_grid_bounds=(-1, 1), z_grid_bounds=(-float('inf'), float('inf'))):
        xs, ys = self._set_xy_grid(x_grid_bounds, y_grid_bounds)
        xgrid, ygrid = np.meshgrid(xs, ys)

        zgrid1 = np.array([[np.mean(self.grid_loss[(x, y)][:size1]) for x in xs] for y in ys])
        zgrid2 = np.array([[np.mean(self.grid_loss[(x, y)][:size2]) for x in xs] for y in ys])
        zgrid_dif = np.square(zgrid2 - zgrid1)

        def bounds_func(x):
            if z_grid_bounds is None:
                return x
            x = min(x, z_grid_bounds[1])
            x = max(x, z_grid_bounds[0])
            return x

        zgrid = np.array([[bounds_func(np.mean(self.grid_loss[(x, y)][:size1])) for x in xs] for y in ys])
        zgrid_dif = np.array([[bounds_func(x) for x in r] for r in zgrid_dif])

        fig = plt.figure(figsize=(17, 6))

        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        surf1 = ax1.plot_surface(xgrid, ygrid, zgrid, linewidth=0, antialiased=False, cmap=cm.get_cmap("coolwarm"),
                                 alpha=1.0)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
        ax1.view_init(40, 20)
        if self.core_type == "random" or (self.core_type == "eigen" and size2 < 0):
            ax1.tick_params(axis='z', pad=15)

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        surf2 = ax2.plot_surface(xgrid, ygrid, zgrid_dif, linewidth=0, antialiased=False, cmap=cm.get_cmap("coolwarm"),
                                 alpha=1.0)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
        ax2.view_init(40, 20)
        if (self.core_type == "random" and size2 > 0) or (self.core_type == "eigen" and size2 < 0):
            ax2.tick_params(axis='z', pad=15)

        plt.savefig("../paper/img/loss_" + self.core_type + f"_{size1}_{size2}" + ".pdf")
        plt.show()


class DeltaVisualizer:
    def __init__(self, calculator):
        self.calculator = calculator
        self.core_type = calculator.core.type

    def visualize_all(self, params, num_samples=64, begin=10):
        params = copy.deepcopy(params)

        deltas = self.calculator.calc_deltas(params, num_samples=num_samples)
        mult_coef = np.square(np.arange(1, len(deltas) + 1))

        fig, axs = plt.subplots(figsize=(16, 6), nrows=1, ncols=2)

        axs[0].plot(deltas)
        axs[1].plot(deltas * mult_coef)

        ylabels = [r'$\Delta_k$',
                   r'$\Delta_k \cdot k^2$']

        axs[0].set(
            xlabel='k',
            ylabel=ylabels[0],
            xlim=[begin, len(deltas)],
            ylim=[min(deltas[begin:]), max(deltas[begin:]) * 1.2]
        )
        axs[0].set_ylabel(ylabels[0], rotation=0,
                          ha='center',
                          va='bottom',
                          labelpad=10)
        axs[0].yaxis.set_label_coords(-0.04, 1.03)

        axs[1].set(
            xlabel='k',
            ylabel=ylabels[1],
            xlim=[begin, len(deltas)],
        )
        axs[1].set_ylabel(ylabels[1], rotation=0,
                          ha='center',
                          va='bottom',
                          labelpad=10)
        axs[1].yaxis.set_label_coords(-0.04, 1.03)

        plt.savefig(
            "../paper/img/delta_" + self.core_type + f"_{params['sigma']}_{params['dim']}_{num_samples}" + ".pdf")
        plt.show()

    def visualize_border(self, border, params, num_samples=64, begin=10, striding_func=None, return_deltas=False):
        params = copy.deepcopy(params)

        deltas = self.calculator.calc_deltas(params, num_samples=num_samples)

        fig, ax = plt.subplots(figsize=(15, 8))

        ax.plot(np.arange(1, len(deltas) + 1), striding_func(deltas), label='Empirical')
        ax.plot(np.arange(1, len(deltas) + 1), border, label='Theoretical', linestyle='--')

        ax.set(
            xlabel='k',
            xlim=[begin, len(deltas) + 1],
            ylim=[min(min(deltas[begin:]), min(border)), max(max(deltas[begin:]), max(border)) * 1.2],
            yscale='log'
        )
        ax.set_ylabel(r'$\Delta_k$', rotation=0,
                      ha='center',
                      va='bottom',
                      labelpad=10)
        ax.yaxis.set_label_coords(-0.04, 1.03)
        ax.legend()

        plt.savefig(
            "../paper/img/delta_border" + f"_{params['sigma']}_{params['dim']}_{num_samples}" + ".pdf")
        plt.show()

        if return_deltas:
            return deltas
        return None

    def compare_params(self,
                       params,
                       target_param_key,
                       target_param_grid,
                       num_samples=64,
                       begin=10):
        params = copy.deepcopy(params)
        target_param_to_deltas = {}
        deltas = []
        for target_param in tqdm(target_param_grid):
            params[target_param_key] = target_param
            deltas = self.calculator.calc_deltas(params, num_samples=num_samples)
            target_param_to_deltas[target_param] = deltas

        fig, axs = plt.subplots(figsize=(18, 7), nrows=1, ncols=2)

        for target_param, deltas in target_param_to_deltas.items():
            axs[0].plot(deltas, label=target_param)
            mult_coef = np.square(np.arange(1, len(deltas) + 1))
            axs[1].plot(deltas * mult_coef, label=target_param)

        ylabels = [r'$\Delta_k$',
                   r'$\Delta_k \cdot k^2$']

        axs[0].set(
            xlabel='k',
            xlim=[begin, len(deltas)],
            ylim=[min(deltas[begin:]), max(deltas[begin:]) * 1.2]
        )
        axs[0].set_ylabel(ylabels[0], rotation=0,
                          ha='center',
                          va='bottom',
                          labelpad=10)
        axs[0].yaxis.set_label_coords(-0.04, 1.03)
        axs[0].set_yscale('log')

        axs[1].set(
            xlabel='k',
            ylabel=ylabels[1],
            xlim=[begin, len(deltas)],
        )
        axs[1].set_ylabel(ylabels[1], rotation=0,
                          ha='center',
                          va='bottom',
                          labelpad=10)
        axs[1].yaxis.set_label_coords(-0.04, 1.03)

        unfixed_param = 'Dimension' if target_param_key == 'dim' else 'Variance'
        axs[0].legend(title=unfixed_param)
        axs[1].legend(title=unfixed_param)

        plt.savefig(
            "../paper/img/delta_" + self.core_type + f"_{target_param_key}" + f"_{params[list(params.keys())[0]]}_{num_samples}" + ".pdf")
        plt.show()

    def compare_samples_num(self, params, num_samples_grid, begin=10):
        max_samples_num = max(num_samples_grid)
        diff_lists = self.calculator.calc_diff_lists(params, num_samples=max_samples_num)
        diff_lists = diff_lists ** 2

        cummean_diff_lists = np.cumsum(diff_lists, axis=0) / np.arange(1, len(diff_lists) + 1).reshape(-1, 1)
        num_samples_to_deltas = {b: cummean_diff_lists[b - 1] for b in num_samples_grid}
        fig, axs = plt.subplots(figsize=(14, 6), nrows=1, ncols=2)

        deltas = []
        for num_samples, deltas in num_samples_to_deltas.items():
            axs[0].plot(deltas, label=num_samples)
            mult_coef = np.square(np.arange(1, len(deltas) + 1))
            axs[1].plot(deltas * mult_coef, label=num_samples)

        ylabels = [r'$\Delta_k$',
                   r'$\Delta_k \cdot k^2$']

        axs[0].set(
            xlabel='k',
            ylabel=ylabels[0],
            xlim=[begin, len(deltas)],
            ylim=[min(deltas[begin:]), max(deltas[begin:]) * 1.2]
        )

        axs[1].set(
            xlabel='k',
            ylabel=ylabels[1],
            xlim=[begin, len(deltas)],
        )

        axs[0].legend(title='num_samples')
        axs[1].legend(title='num_samples')

        plt.savefig(
            "../paper/img/delta_" + self.core_type + "_num_samples" + f"_{params['sigma']}_{params['dim']}" + ".pdf")
        plt.show()


def vis_compare(theory, empiric, loss_func, params, num_samples=64, begin=10):
    loss = loss_func.cpu().detach().numpy()[:-1]
    _, axs = plt.subplots(figsize=(18, 14), nrows=2, ncols=1)

    axs[0].scatter(loss[begin:], empiric[begin:])

    k1, b1 = np.polyfit(loss[begin:], empiric[begin:], 1)
    y1_trend = k1 * loss[begin:] + b1
    axs[0].plot(loss[begin:], y1_trend, color='red', linewidth=2, label=f'Trend line')
    axs[0].set(xlabel=r'$\mathcal {L}_k$', ylabel=r'$\Delta_k$ Empirical')
    axs[0].legend()

    axs[1].scatter(loss[begin:], theory[begin:])

    k2, b2 = np.polyfit(loss[begin:], theory[begin:], 1)
    y2_trend = k2 * loss[begin:] + b2
    axs[1].plot(loss[begin:], y2_trend, color='green', linewidth=2, label=f'Trend line')
    axs[1].set(xlabel=r'$\mathcal {L}_k$', ylabel=r'$\Delta_k$ Theoretical')
    axs[1].legend()

    plt.savefig("../paper/img/delta_loss" + f"_{params['sigma']}_{params['dim']}_{num_samples}" + ".pdf")
    plt.show()
