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

        title = rf'$\mathcal{{L}}_{{{size1}}}$'
        zgrid = np.array([[bounds_func(np.mean(self.grid_loss[(x, y)][:size1])) for x in xs] for y in ys])

        title_dif = rf'$(\mathcal{{L}}_{{{size2}}} - \mathcal{{L}}_{{{size1}}})^2$'
        zgrid_dif = np.array([[bounds_func(x) for x in r] for r in zgrid_dif])

        fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.set_title(title)
        surf1 = ax1.plot_surface(xgrid, ygrid, zgrid, linewidth=0, antialiased=False, cmap=cm.get_cmap("coolwarm"),
                                 alpha=1.0)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
        ax1.view_init(40, 20)

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.set_title(title_dif)
        surf2 = ax2.plot_surface(xgrid, ygrid, zgrid_dif, linewidth=0, antialiased=False, cmap=cm.get_cmap("coolwarm"),
                                 alpha=1.0)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
        ax2.view_init(40, 20)
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

        fig, axs = plt.subplots(figsize=(14, 6), nrows=1, ncols=2)
        fig.suptitle(rf'$\Delta_k$: $\sigma$ = {params['sigma']}, dimensions = {params['dim']}, K = {num_samples}')

        axs[0].plot(deltas)
        axs[1].plot(deltas * mult_coef)

        ylabels = [r'$\mathbb{E}(\mathcal{L}_{k+1} - \mathcal{L}_{k})^2$',
                   r'$\mathbb{E}(\mathcal{L}_{k+1} - \mathcal{L}_{k})^2 \cdot k^2$']

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
        plt.savefig(
            "../paper/img/delta_" + self.core_type + f"_{params['sigma']}_{params['dim']}_{num_samples}" + ".pdf")
        plt.show()

    def visualize_border(self, border, params, num_samples=64, begin=10):
        params = copy.deepcopy(params)

        deltas = self.calculator.calc_deltas(params, num_samples=num_samples)

        fig, ax = plt.subplots(figsize=(8, 6))

        offset = len(deltas) - len(border)
        ax.plot(np.arange(offset, len(deltas)), deltas[offset:], label='Δ_k')
        ax.plot(np.arange(offset, len(deltas)), border, label='border', linestyle='--')

        ax.set(
            title=rf'$\Delta_k$: $\sigma$ = {params["sigma"]}, dimensions = {params["dim"]}, K = {num_samples}',
            xlabel='k',
            ylabel=r'$\mathbb{E}(\mathcal{L}_{k+1} - \mathcal{L}_{k})^2$',
            xlim=[max(offset, begin), len(deltas)],
            ylim=[
                min(min(deltas[max(offset, begin):]), min(border)),
                max(max(deltas[max(offset, begin):]), max(border)) * 1.2]
        )
        ax.legend()

        plt.savefig(
            "../paper/img/delta_border" + f"_{params['sigma']}_{params['dim']}_{num_samples}" + ".pdf"
        )
        plt.show()

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

        fixed_param = 'Sigma' if target_param_key == 'dim' else 'Dimensions'
        fig, axs = plt.subplots(figsize=(14, 6), nrows=1, ncols=2)
        fig.suptitle(rf'$\Delta_k$, Compare {target_param_key}: {fixed_param} = {list(params.values())[0]}')

        for target_param, deltas in target_param_to_deltas.items():
            axs[0].plot(deltas, label=target_param)
            mult_coef = np.square(np.arange(1, len(deltas) + 1))
            axs[1].plot(deltas * mult_coef, label=target_param)

        ylabels = [r'$\mathbb{E}(\mathcal{L}_{k+1} - \mathcal{L}_{k})^2$',
                   r'$\mathbb{E}(\mathcal{L}_{k+1} - \mathcal{L}_{k})^2 \cdot k^2$']

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

        axs[0].legend(title=target_param_key)
        axs[1].legend(title=target_param_key)

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

        fig.suptitle(rf'$\Delta_k$, Compare num_samples: Sigma = {params["sigma"]}, Dimensions = {params["dim"]}')

        deltas = []
        for num_samples, deltas in num_samples_to_deltas.items():
            axs[0].plot(deltas, label=num_samples)
            mult_coef = np.square(np.arange(1, len(deltas) + 1))
            axs[1].plot(deltas * mult_coef, label=num_samples)

        ylabels = [r'$\mathbb{E}(\mathcal{L}_{k+1} - \mathcal{L}_{k})^2$',
                   r'$\mathbb{E}(\mathcal{L}_{k+1} - \mathcal{L}_{k})^2 \cdot k^2$']

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
