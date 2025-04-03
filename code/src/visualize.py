import itertools
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import scipy.stats as sps
from tqdm.auto import tqdm

from src.calc import DeltaCalculator


class LossVisualizer:
    def __init__(self, calculator, grid_step=0.1):
        """
            grid_step in [0, 1]
        """
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

    def visualize_diff(self, size1, size2,
                       x_grid_bounds=(-1, 1), y_grid_bounds=(-1, 1),
                       diff_type=None, bounds=None, distrib_params=None):
        """
           type: None | abs | square | realative | relative_abs | relative_squar | square_dot_normal | abs_dot_normal
           bounds: None | (-min, max)
        """
        assert bounds is None or bounds[0] < bounds[1]

        xs, ys = self._set_xy_grid(x_grid_bounds, y_grid_bounds)
        xgrid, ygrid = np.meshgrid(xs, ys)
        zgrid1 = np.array([[np.mean(self.grid_loss[(x, y)][:size1]) for x in xs] for y in ys])
        zgrid2 = np.array([[np.mean(self.grid_loss[(x, y)][:size2]) for x in xs] for y in ys])
        zgrid = zgrid2 - zgrid1

        best_loss = np.round(min(np.min(zgrid1), np.min(zgrid2)), 4)

        relative = False
        if diff_type is not None and diff_type[0:2] == 're':
            zgrid /= zgrid2
            relative = True

        if diff_type is not None and diff_type.find('normal') != -1:
            pdf = lambda x, y: np.prod(sps.norm(**distrib_params).pdf(np.array([x, y])))
            zgrid *= np.array([[pdf(x, y) for x in xs] for y in ys])

        title = r""
        if diff_type is None:
            title = r'$\mathcal{L}_{s_2} - \mathcal{L}_{s_1}$'
            if relative:
                title += r'$/ \mathcal{L}_{s2}$'
        elif diff_type == 'abs' or diff_type == 'abs_dot_normal' or diff_type == 'relative_abs':
            title = r'$|\mathcal{L}_{s_2} - \mathcal{L}_{s_1}|$'
            zgrid = np.abs(zgrid)
            if relative:
                title += r'$/ \mathcal{L}_{s2}$'
        elif diff_type == 'square' or diff_type == 'square_dot_normal' or diff_type == 'relative_square':
            title = r'$(\mathcal{L}_{s_2} - \mathcal{L}_{s_1})^2$'
            zgrid = np.square(zgrid)
            if relative:
                title += r'$/ \mathcal{L}_{s2}^2$'

        dot_distrib_flag = False
        if diff_type is not None and diff_type.find('normal') != -1:
            title += r' $p(\mathbf{w})$'
            dot_distrib_flag = True

        title += rf'; $s_1 = {size1}, s_2 = {size2}$'
        title += r', optimal loss :' + rf' {best_loss}'
        if dot_distrib_flag:
            title += rf', distribution params: {distrib_params}'

        def bounds_func(x):
            if bounds is None:
                return x
            x = min(x, bounds[1])
            x = max(x, bounds[0])
            return x

        zgrid = np.array([[bounds_func(x) for x in r] for r in zgrid])

        fig = plt.figure(figsize=(16, 6))
        ax_3d = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax_3d)
        ax_3d.set(
            title=title
        )
        surf = ax_3d.plot_surface(xgrid, ygrid, zgrid, linewidth=0, antialiased=False, cmap=cm.get_cmap("coolwarm"),
                                  alpha=1.0)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax_3d.view_init(40, 20)

        plt.savefig(f"Ld_{size1}.pdf")
        plt.show()

    def visualize(self, size=None,
                  x_grid_bounds=(-1, 1), y_grid_bounds=(-1, 1), z_grid_bounds=(-float('inf'), float('inf'))):
        grid_loss = self.grid_loss

        xs, ys = self._set_xy_grid(x_grid_bounds, y_grid_bounds)
        xgrid, ygrid = np.meshgrid(xs, ys)

        def bounds_func(x):
            bounds = z_grid_bounds
            if bounds is None:
                return x
            x = min(x, bounds[1])
            x = max(x, bounds[0])
            return x

        if size is None:
            zgrid = np.array([[bounds_func(np.mean(grid_loss[(x, y)])) for x in xs] for y in ys])
            title = r'$\mathcal{L}_{s}; s = -1$'
        else:
            zgrid = np.array([[bounds_func(np.mean(grid_loss[(x, y)][:size])) for x in xs] for y in ys])
            title = r'$\mathcal{L}_{s}; $' + f's = {size}'
        best_loss = np.round(np.min(zgrid), 2)
        title += rf', optimal loss: {best_loss}'

        fig = plt.figure(figsize=(16, 6))
        ax_3d = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax_3d)
        ax_3d.set(title=title)
        surf = ax_3d.plot_surface(xgrid, ygrid, zgrid, linewidth=0, antialiased=False, cmap=cm.get_cmap("coolwarm"),
                                  alpha=1.0)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax_3d.view_init(40, 20)
        plt.savefig(f"Ls_{size}.pdf")
        plt.show()


class DeltaVisualizer:
    def __init__(self, calculator):
        self.calculator = calculator

    def visualize_diff(self,
                       params,
                       num_samples,
                       begin):
        params = copy.deepcopy(params)

        deltas = self.calculator.calc_deltas(params, num_samples=num_samples)

        fig, axs = plt.subplots(figsize=(9, 6), nrows=1, ncols=1)
        fig.suptitle(fr'$\Delta_k$; {params}')
        axs.plot(deltas, label="Test")
        mult_coef = np.arange(1, len(deltas) + 1)
        if params['estim_func'] == 'square':
            mult_coef = mult_coef ** 2
        # axs[1].plot(deltas * mult_coef)

        if params['estim_func'] == 'abs':
            ylabels = [r'$\mathbb{E}_{p(\mathbf{w})}|L_k - L_{k-1}|$',
                       r'$\mathbb{E}_{p(\mathbf{w})}|L_{k} - L_{k-1}| \cdot k$']
        elif params['estim_func'] == 'square':
            ylabels = [r'$\mathbb{E}_{p(\mathbf{w})}|L_k - L_{k-1}|^2$',
                       r'$\mathbb{E}_{p(\mathbf{w})}|L_k - L_{k-1}|^2 \cdot k^2$']
        else:
            ylabels = [r'$\mathbb{E}_{p(\mathbf{w})}(L_k - L_{k-1})$',
                       r'$\mathbb{E}_{p(\mathbf{w})}(L_k - L_{k-1}) \cdot k$']

        axs.set(
            xlabel='k',
            ylabel=ylabels[0],
            xlim=[begin, len(deltas)],
            ylim=[min(deltas[begin:]), max(deltas[begin:]) * 1.2]
        )

        # axs[1].set(
        #     xlabel='k',
        #     ylabel=ylabels[1],
        # )
        plt.savefig("2.eps")
        plt.show()

    def compare_params(self,
                       mode,
                       params,
                       target_param_key,
                       target_param_grid,
                       num_samples,
                       begin):
        params = copy.deepcopy(params)
        target_param_to_deltas = {}
        deltas = []
        for target_param in tqdm(target_param_grid):
            params[target_param_key] = target_param
            deltas = self.calculator.calc_deltas(mode, params, num_samples=num_samples)
            target_param_to_deltas[target_param] = deltas

        fig, axs = plt.subplots(figsize=(16, 6), nrows=1, ncols=2)
        fig.suptitle(fr'$\Delta_k$; {params}')
        for target_param, deltas in target_param_to_deltas.items():
            axs[0].plot(deltas, label=target_param)
            mult_coef = np.arange(1, len(deltas) + 1)
            if params['estim_func'] == 'square':
                mult_coef = mult_coef ** 2
            axs[1].plot(deltas * mult_coef, label=target_param)

        if params['estim_func'] == 'abs':
            ylabels = [r'$\mathbb{E}_{p(\mathbf{w})}|L_k - L_{k-1}|$',
                       r'$\mathbb{E}_{p(\mathbf{w})}|L_{k} - L_{k-1}| \cdot k$']
        elif params['estim_func'] == 'square':
            ylabels = [r'$\mathbb{E}_{p(\mathbf{w})}|L_k - L_{k-1}|^2$',
                       r'$\mathbb{E}_{p(\mathbf{w})}|L_k - L_{k-1}|^2 \cdot k^2$']
        else:
            ylabels = [r'$\mathbb{E}_{p(\mathbf{w})}(L_k - L_{k-1})$',
                       r'$\mathbb{E}_{p(\mathbf{w})}(L_k - L_{k-1}) \cdot k$']

        axs[0].set(
            title=f'Compare {target_param_key}',
            xlabel='k',
            ylabel=ylabels[0],
            xlim=[begin, len(deltas)],
            ylim=[min(deltas[begin:]), max(deltas[begin:]) * 1.2]
        )

        axs[1].set(
            title=f'Compare {target_param_key}',
            xlabel='k',
            ylabel=ylabels[1],
        )

        axs[0].legend(title=target_param_key)
        axs[1].legend(title=target_param_key)

        plt.show()

    def compare_samples_num(self,
                            mode,
                            params,
                            num_samples_grid,
                            begin=0
                            ):

        max_samples_num = max(num_samples_grid)
        diff_lists = self.calculator.calc_diff_lists(mode, params, num_samples=max_samples_num)
        if params['estim_func'] == 'square':
            diff_lists = diff_lists ** 2
        elif params['estim_func'] == 'abs':
            diff_lists = np.abs(diff_lists)
        cummean_diff_lists = np.cumsum(diff_lists, axis=0) / np.arange(1, len(diff_lists) + 1).reshape(-1, 1)
        num_samples_to_deltas = {b: cummean_diff_lists[b - 1] for b in num_samples_grid}
        fig, axs = plt.subplots(figsize=(16, 6), nrows=1, ncols=2)
        fig.suptitle(f'params: {params}')

        deltas = []
        for num_samples, deltas in num_samples_to_deltas.items():
            axs[0].plot(deltas, label=num_samples)
            mult_coef = np.arange(1, len(deltas) + 1)
            if params['estim_func'] == 'square':
                mult_coef = mult_coef ** 2
            axs[1].plot(deltas * mult_coef, label=num_samples)

        if params['estim_func'] == 'abs':
            ylabels = [r'$\mathbb{E}_{p(\mathbf{w})}|L_k - L_{k-1}|$',
                       r'$\mathbb{E}_{p(\mathbf{w})}|L_{k} - L_{k-1}| \cdot k$']
        elif params['estim_func'] == 'square':
            ylabels = [r'$\mathbb{E}_{p(\mathbf{w})}|L_k - L_{k-1}|^2$',
                       r'$\mathbb{E}_{p(\mathbf{w})}|L_k - L_{k-1}|^2 \cdot k^2$']
        else:
            ylabels = [r'$\mathbb{E}_{p(\mathbf{w})}(L_k - L_{k-1})$',
                       r'$\mathbb{E}_{p(\mathbf{w})}(L_k - L_{k-1}) \cdot k$']

        axs[0].set(
            title=f'Compare num_samples',
            xlabel='k',
            ylabel=ylabels[0],
            xlim=[begin, len(deltas)],
            ylim=[min(deltas[begin:]), max(deltas[begin:]) * 1.2]
        )

        axs[1].set(
            title=f'Compare num_samples',
            xlabel='k',
            ylabel=ylabels[1],
        )

        axs[0].legend(title='num_samples')
        axs[1].legend(title='num_samples')

        plt.show()
