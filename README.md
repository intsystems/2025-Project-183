# Neural Networks Loss Landscape Convergence in Hessian Low-Dimensional Space

[![License](https://badgen.net/github/license/intsystems/2025-Project-183?color=green)](https://github.com/intsystems/2025-Project-183/blob/main/LICENSE)
[![GitHub Contributors](https://img.shields.io/github/contributors/intsystems/2025-Project-183)](https://github.com/intsystems/2025-Project-183/graphs/contributors)
[![GitHub Issues](https://img.shields.io/github/issues-closed/intsystems/2025-Project-183.svg?color=0088ff)](https://github.com/intsystems/2025-Project-183/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-closed/intsystems/2025-Project-183.svg?color=7f29d6)](https://github.com/intsystems/2025-Project-183/pulls)

<table>
    <tr>
        <td align="left"> <b> Author </b> </td>
        <td> Artem Nikitin </td>
    </tr>
    <tr>
        <td align="left"> <b> Consultant </b> </td>
        <td> Nikita Kiselev </td>
    </tr>
    <tr>
        <td align="left"> <b> Advisor </b> </td>
        <td> Andrey Grabovoy, PhD </td>
    </tr>
</table>

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main.pdf)
- [Slides](slides/full/main.pdf)

## Abstract
  Understanding how a neural network’s loss landscape changes as we add more training data is important for efficient training.
  Although larger datasets are known to reshape this high-dimensional surface, the point when extra data stop making a big difference
  is still unclear.

  In this paper, we study this issue and show that the loss landscape near a local minimum stabilizes once the dataset exceeds a
  certain size. To analyze this, we project the full parameter space onto a smaller subspace formed by the Hessian’s top eigenvectors.
  This low-dimensional view highlights how the loss surface changes in its most important directions. We then apply Monte Carlo sampling
  within this subspace to estimate these changes more precisely.

  We test our approach on standard image‑classification tasks and find that our low-dimensional analysis pinpoints when the landscape
  stops evolving. These findings clarify how dataset size affects optimization and offer practical guidance for balancing training cost
  with performance gains.

## Citation

If you find our work helpful, please cite us.

```BibTeX
@article{intsystems183,
    title={Neural Networks Loss Landscape Convergence in Hessian Low-Dimensional Space},
    author={Tem Nikitin, Nikita Kiselev, Vladislav Meshkov, Andrey Grabovoy},
    year={2025}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
