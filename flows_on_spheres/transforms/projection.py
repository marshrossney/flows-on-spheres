from math import pi as π
from typing import Optional

import torch

from flows_on_spheres.geometry import (
    circle_vectors_to_angles,
    circle_angles_to_vectors,
)
from flows_on_spheres.nn import TransformModule
from flows_on_spheres.utils import mod_2pi

Tensor = torch.Tensor


class _ProjectedAffineTransform:
    def __init__(
        self,
        log_scale: Tensor,
        shift: Tensor | None = None,
        linear_thresh: float | None = None,
    ):
        assert log_scale.dim() == 2
        assert log_scale.shape[1] == 1
        self.log_scale = log_scale
        self.shift = shift if shift is not None else 0
        self._linear_thresh = linear_thresh

    def _call(self, x: Tensor) -> tuple[Tensor, Tensor]:
        log_α, β = self.log_scale, self.shift
        α = torch.exp(log_α)

        y = mod_2pi(2 * torch.atan(α * torch.tan((x - π) / 2) + β) + π)

        dxdy = (
            (1 + β**2) / α * torch.sin(x / 2) ** 2
            + α * torch.cos(x / 2) ** 2
            - β * torch.sin(x)
        )

        if self._linear_thresh is None:
            ldj = dxdy.log().negative().squeeze(1)
            return y, ldj

        m1 = x < self._linear_thresh
        m2 = (2 * π - x) < self._linear_thresh

        y[m1] = (x / α)[m1]
        y[m2] = (2 * π - (2 * π - x) / α)[m2]
        dxdy[m1 | m2] = α[m1 | m2]

        ldj = dxdy.log().negative().squeeze(1)

        return y, ldj

    def _inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        log_α, β = self.log_scale, self.shift
        α = torch.exp(log_α)

        x = mod_2pi(
            2 * torch.atan((1 / α) * torch.tan((y - π) / 2) - (β / α)) + π
        )

        dxdy = (
            (1 + β**2) / α * torch.sin(x / 2) ** 2
            + α * torch.cos(x / 2) ** 2
            - β * torch.sin(x)
        )

        if self._linear_thresh is None:
            ldj = dxdy.log().squeeze(1)
            return x, ldj

        m1 = y < self._linear_thresh
        m2 = (2 * π - y) < self._linear_thresh

        y[m1] = (y * α)[m1]
        y[m2] = (2 * π - (2 * π - y) * α)[m2]
        dxdy[m1 | m2] = α[m1 | m2]

        ldj = dxdy.log().squeeze(1)

        return x, ldj

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        assert x.dim() == 2
        assert x.shape[1] == 2
        x = circle_vectors_to_angles(x)
        y, ldj = self._call(x)
        y = circle_angles_to_vectors(y)
        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        assert y.dim() == 2
        assert y.shape[1] == 2
        y = circle_vectors_to_angles(y)
        x, ldj = self._inverse(y)
        x = circle_angles_to_vectors(x)
        return x, ldj


class _ProjectedAffineMixtureTransform:
    def __init__(
        self,
        log_scale: Tensor,
        shift: Tensor | None,
        weights: Tensor | None,
        linear_thresh: float | None = None,
    ):
        assert log_scale.dim() == 3
        assert log_scale.shape[2] == 1

        n_mixture = log_scale.shape[1]

        if weights is not None:
            assert weights.dim() == 3
            assert weights.shape[2] == 1
            assert weights.shape[1] == n_mixture
            assert torch.allclose(weights.sum(dim=1), weights.new_ones(1))
        else:
            weights = 1 / n_mixture

        self._n_mixture = n_mixture
        self.log_scale = log_scale
        self.shift = shift if shift is not None else 0
        self.weights = weights

        self._linear_thresh = linear_thresh

    def _call(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.unsqueeze(1)  # mixture dimension

        log_α, β, ρ = self.log_scale, self.shift, self.weights
        α = torch.exp(log_α)

        y = mod_2pi(2 * torch.atan(α * torch.tan((x - π) / 2) + β) + π)

        dxdy = (
            (1 + β**2) / α * torch.sin(x / 2) ** 2
            + α * torch.cos(x / 2) ** 2
            - β * torch.sin(x)
        )

        if self._linear_thresh is not None:
            m1 = x < self._linear_thresh
            m2 = (2 * π - x) < self._linear_thresh

            y[m1] = (x / α)[m1]
            y[m2] = (2 * π - (2 * π - x) / α)[m2]
            dxdy[m1 | m2] = α[m1 | m2]

        y = (ρ * y).sum(dim=1)
        dydx = (ρ * (1 / dxdy)).sum(dim=1)

        ldj = dydx.log().squeeze(1)

        return y, ldj

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        assert x.dim() == 2
        assert x.shape[1] == 2
        x = circle_vectors_to_angles(x)
        y, ldj = self._call(x)
        y = circle_angles_to_vectors(y)
        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class ProjectedAffineModule(TransformModule):
    def __init__(
        self,
        *,
        n_mixture: int,
        weighted: bool,
        no_shift: bool,
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: Optional[str] = None,
        linear_thresh: Optional[float] = None,
    ):
        super().__init__(
            n_params=(1 + int(not no_shift) + int(weighted)) * n_mixture,
            net_hidden_shape=net_hidden_shape,
            net_activation=net_activation,
        )
        self.n_mixture = n_mixture
        self.weighted = weighted
        self.no_shift = no_shift
        self.linear_thresh = linear_thresh

    def __call__(
        self, k: Tensor | None
    ) -> _ProjectedAffineTransform | _ProjectedAffineMixtureTransform:
        params = self.params(k)

        if self.n_mixture == 1:
            if self.no_shift:
                log_scale, shift = params, None
            else:
                log_scale, shift = params.split(1, dim=1)
            return _ProjectedAffineTransform(
                log_scale, shift, self.linear_thresh
            )

        else:
            params = params.unflatten(1, (self.n_mixture, -1))
            if self.no_shift and not self.weighted:
                log_scale, shift, weights = params, None, None
            elif self.no_shift and self.weighted:
                log_scale, weights = params.split(1, dim=2)
                shift = None
            elif (not self.no_shift) and not self.weighted:
                log_scale, shift = params.split(1, dim=2)
                weights = None
            elif (not self.no_shift) and self.weighted:
                log_scale, shift, weights = params.split(1, dim=2)

            if weights is not None:
                weights = torch.softmax(weights, dim=1)

            return _ProjectedAffineMixtureTransform(
                log_scale, shift, weights, self.linear_thresh
            )
