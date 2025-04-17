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
- [Paper](paper)
- [Slides](slides)

## Abstract

Understanding how the loss landscape of neural networks evolves as the training set size increases is crucial for
optimizing performance. It is well known that larger datasets can alter the shape of this high-dimensional landscape.
But the exact point at which additional data no longer brings substantial changes remains underexplored.

In this paper, we examine neural network models and show that their loss landscapes begin to stabilize once the
training set grows beyond a certain threshold, revealing a connection between dataset size and the geometry of the
loss surface. To understand this phenomenon, we propose a method that projects the full parameter space onto a
low-dimensional subspace derived from top eigenvectors of the Hessian. That provides a more interpretable view of how
the loss surface in the vicinity of local minima behaves as more data become available. We use different sampling
strategies, applying Monte-Carlo estimation to capture the structure of this reduced loss landscape more precisely.

We validate our insights through experiments on image classification tasks, demonstrating that this low-dimensional
analysis can reveal when the landscape effectively settles, and thus helps determine a minimum viable dataset size.
Our findings shed light on the relationship between dataset scale and optimization geometry, and suggest
practical strategies for balancing computational costs with the benefits of additional training data.

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
