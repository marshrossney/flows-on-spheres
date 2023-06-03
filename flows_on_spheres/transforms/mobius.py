from functools import partial
import math
from typing import Optional

import torch
import torch.nn.functional as F

from flows_on_spheres.nn import CircularTransformModule
from flows_on_spheres.geometry import as_angle, as_vector
from flows_on_spheres.linalg import dot, norm
from flows_on_spheres.utils import mod_2pi

Tensor = torch.Tensor


def mobius(x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
    c = params

    x_x0 = torch.stack([as_vector(x), as_vector(torch.zeros_like(x))])

    dydx = ((1 - torch.dot(c, c)) / torch.dot(x_x0 - c, x_x0 - c)).unsqueeze(
        -1
    )
    y_y0 = dydx * (x_x0 - c) - c

    y, y0 = as_angle(y_y0)
    y = mod_2pi(y - y0)

    dydx, _ = dydx

    return y, dydx


# _mobius_vmap = torch.vmap(mobius, in_dims=(None, 0), out_dims=(0, 0))

"""
def mobius_mixture(x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
    y, dydx = _mobius_vmap(x, params)
    return y.mean(dim=0), dydx.mean(dim=0)


def mobius_weighted_mixture(x: Tensor, params: Tensor:) -> tuple[Tensor, Tensor]:
    c, w = params
    y, dydx = _mobius_vmap(x, c)
    return (w * y).mean(dim=0), (w * dydx).mean(dim=0)
"""


class MobiusModule(CircularTransformModule):
    def __init__(
        self,
        *,
        n_mixture: int,
        weighted: bool,
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: Optional[str] = None,
        epsilon: float = 1e-2,
    ):
        assert not (n_mixture == 1 and weighted)
        super().__init__(
            n_params=(2 + int(weighted)) * n_mixture,
            net_hidden_shape=net_hidden_shape,
            net_activation=net_activation,
        )
        self.n_mixture = n_mixture
        self.weighted = weighted
        self.epsilon = epsilon

        if n_mixture == 1:
            self.mobius_func = mobius
        elif weighted is True:
            self.mobius_func = make_weighted_mixture(mobius)
        else:
            self.mobius_func = make_mixture(mobius)

    def _constrain_to_disk(self, c: Tensor) -> Tensor:
        c = torch.tanh(c) * (1 - self.epsilon)
        c1, c2 = c.split(1, dim=-1)
        c = torch.cat([c1, c2 * (1 - c1.pow(2)).sqrt()], dim=-1)
        return c

    def forward(self, k: Tensor | None = None):
        params = self.params(k)

        if self.mixture > 1:
            params = params.unflatten(-1, (self.n_mixture, -1))

        if self.weighted:
            params, weights = params.split([2, 1], dim=-1)
            params = self._constrain_to_disk(params)
            params = torch.cat([params, weights], dim=-1)
        else:
            params = self._constrain_to_disk(params)

        if k is None:
            func = torch.vmap(self.mobius_func, (0, 0), (0, 0))
        else:
            func = torch.vmap(self.mobius_func, (0, None), (0, 0))

        return partial(func, params=params)


"""

        if self.n_mixture == 1:
            c = self._constrain_to_disk(params)
            func = partial(mobius, c=c)
            return torch.vmap(func)

        if not self.weighted:
            params = params.unflatten(1, (self.n_mixture, 2))
            c = self._constrain_c(params)
            return _MobiusMixtureTransform(c)

        else:
            params = params.unflatten(1, (self.n_mixture, 3))
            c, weights = params.split([2, 1], dim=2)
            c = self._constrain_c(c)
            weights = torch.softmax(weights, dim=1)
            return _MobiusMixtureTransform(c, weights)

"""


class _MobiusTransform:
    def __init__(self, points: Tensor):
        assert points.shape[-1] == 2
        self.points = points

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        assert x.shape[-1] == 2
        ω = self.points
        dydx = ((1 - dot(ω, ω)) / dot(x - ω, x - ω)).unsqueeze(-1)
        y = dydx * (x - ω) - ω
        ldj = dydx.log().sum(dim=-1)
        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        assert y.shape[-1] == 2
        ω = self.points
        dxdy = ((1 - dot(ω, ω)) / dot(y + ω, y + ω)).unsqueeze(-1)
        x = dxdy * (y + ω) + ω
        ldj = dxdy.log().sum(dim=-1)
        return x, ldj


"""
class _MobiusMixtureTransform:
    def __init__(
            self,
            points: Tensor,
            weights: Tensor | None = None,
        ):
        ω, ρ = points, weights
        assert (norm(ω) < 1).all()

        n_batch, n_mixture, n_coords = ω.shape
        assert n_coords == 2  # only valid for circle

        if ρ is not None:
            assert ρ.shape == (n_batch, n_mixture, 1), f"{ρ.shape}"
            log_ρ = F.log_softmax(ρ, dim=1)
        else:
            log_ρ = -math.log(n_mixture)

        self.points = ω
        self.log_weights = log_ρ

        self.func = torch.vmap(mobius_transform, in_dims=(None, 1), out_dims=(1, 1))

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        assert x.shape[-1] == 2
        log_ρ = self.points, self.log_weights

        y, ldj = self.func(x, self.points)
        print(y.shape, ldj.shape)

        ldj = torch.logsumexp(ldj + self.log_weights, dim=1)

        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError
class _MobiusMixtureTransform:
    def __init__(
        self,
        c: Tensor,
        weights: Tensor | None = None,
    ):
        ω, ρ = c, weights
        assert ω.dim() == 3
        assert (norm(ω) < 1).all()

        n_batch, n_mixture, n_coords = ω.shape
        assert n_coords == 2  # only valid for circle

        if ρ is not None:
            assert ρ.shape == (n_batch, n_mixture, 1), f"{ρ.shape}"
            assert torch.allclose(ρ.sum(dim=1), ρ.new_ones(1))
        else:
            ρ = 1 / n_mixture

        self.c = ω
        self.weights = ρ

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        assert x.dim() == 2
        assert x.shape[1] == 2
        ω, ρ = self.c, self.weights

        x0 = torch.tensor([1, 0]).type_as(x).expand_as(x)
        x_x0 = torch.stack([x, x0], dim=0).unsqueeze(-2)  # add mixture dim

        dydx = ((1 - dot(ω, ω)) / dot(x_x0 - ω, x_x0 - ω)).unsqueeze(-1)
        y_y0 = dydx * (x_x0 - ω) - ω

        # Now rotate s.t (1, 0) -> (1, 0)
        ϕ, ϕ0 = torch.atan2(*reversed(y_y0.split(1, dim=-1)))
        ϕ = mod_2pi(ϕ - ϕ0)

        Σρϕ = (ρ * ϕ).sum(dim=1)
        y = torch.cat([torch.cos(Σρϕ), torch.sin(Σρϕ)], dim=-1)

        dydx, _ = dydx
        dydx = (ρ * dydx).sum(dim=1)
        ldj = dydx.log().sum(dim=1)

        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

"""
