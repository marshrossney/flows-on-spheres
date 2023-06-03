from typing import Any, Callable, TypeAlias

import torch
import torch.nn.functional as F

from flows_on_spheres.nn import TransformModule

Tensor: TypeAlias = torch.Tensor
Transform: TypeAlias = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]


def affine_transform(x: Tensor, s: Tensor, t: Tensor):
    return (x - t) * s, s


def make_mixture(transform: Transform, n_mixture: int):
    transform_vmap = torch.vmap(transform, in_dims=(None, 0), out_dims=(0, 0))

    def _mixture_transform(x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        y, dydx = transform_vmap(x, params)
        return y.mean(dim=0), dydx.mean(dim=0)

    return _mixture_transform


def make_weighted_mixture(transform: Transform, *, min_weight: float = 1e-2):
    transform_vmap = torch.vmap(transform, in_dims=(None, 0), out_dims=(0, 0))

    def _mixture_transform(x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        params, weights = params.tensor_split([-1], dim=1)

        n, ε = weights.shape[0], min_weight
        weights = F.softmax(weights, dim=0) * (1 - n * ε) + ε

        y, dydx = transform_vmap(x, params)
        assert y.shape == weights.shape
        return (weights * y).mean(dim=0), (weights * dydx).mean(dim=0)

    return _mixture_transform


def mixture_transform(transform: Transform, n_mixture: int) -> Transform:
    _transform = torch.vmap(transform, in_dims=(None, 0, 0), out_dims=(0, 0))

    def _mixture_transform(x: Tensor, *args, **kwargs):
        y, dydx = _transform(x, *args, **kwargs)
        return y.mean(dim=0), dydx.mean(dim=0)

    return _mixture_transform


def weighted_mixture(transform: Transform, weights: Tensor) -> Transform:
    weights = torch.softmax(weights, dim=0)

    _transform = torch.vmap(transform, in_dims=(None, 0, 0), out_dims=(0, 0))

    def _mixture_transform(x: Tensor, *args, **kwargs):
        y, dydx = _transform(x, *args, **kwargs)
        assert y.shape == weights.shape
        return (y * weights).mean(dim=0), (dydx * weights).mean(dim=0)

    return _mixture_transform


if __name__ == "__main__":
    x = torch.randn(1)
    s = torch.rand(10, 1)
    t = torch.rand(10, 1)
    n = 10

    f = mixture_transform(affine_transform, n)
    y, dydx = f(x, s, t)

    print(y.shape, dydx.shape)

    g = weighted_mixture(affine_transform, torch.rand(10, 1))
    y, dydx = g(x, s, t)

    print(y.shape, dydx.shape)

    h = torch.vmap(f)
    x = torch.randn(100, 1)
    s = torch.randn(100, 10, 1)
    t = torch.randn(100, 10, 1)
    y, dydx = h(x, s, t)

    print(y.shape, dydx.shape)
