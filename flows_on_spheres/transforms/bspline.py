from typing import Optional, TypeAlias

import torch
import torch.nn.functional as F

from flows_on_spheres.nn import TransformModule

Tensor: TypeAlias = torch.Tensor


class _BSplineTransform:
    def __init__(
        self,
        intervals: Tensor,
        weights: Tensor,
        omega: Tensor,
        knots_x: Tensor,
        knots_y: Tensor,
    ):
        self.intervals = intervals
        self.weights = weights
        self.omega = omega
        self.knots_x = knots_x
        self.knots_y = knots_y

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = (x + 1) / 2  # rescale [-1, 1] -> [0, 1]

        n_batch, n_coords = x.shape
        assert n_coords == 1

        i = torch.searchsorted(
            self.knots_x.expand(n_batch, -1), x, side="right"
        ).clamp(1, len(self.intervals))

        # Get parameters of the segments that x falls in
        Δ = torch.gather(self.intervals.expand(n_batch, -1), -1, i)
        ρ = torch.gather(self.weights.expand(n_batch, -1), -1, i)
        ωi = torch.gather(self.omega.expand(n_batch, -1), -1, i)
        ωim1 = torch.gather(self.omega.expand(n_batch, -1), -1, i - 1)
        x0 = torch.gather(self.knots_x.expand(n_batch, -1), -1, i - 1)
        y0 = torch.gather(self.knots_y.expand(n_batch, -1), -1, i - 1)

        θ = (x - x0) / Δ

        y = (
            y0
            + ρ * Δ * θ
            - ωim1 * Δ**2 * θ * (1 - θ)
            + (1 / 3) * (ωi - ωim1) * Δ**2 * θ**3
        )

        dydx = ρ + ωi * Δ * θ**2 - ωim1 * Δ * (1 - θ) ** 2

        ldj = dydx.log().sum(dim=1)

        y = y * 2 - 1  # rescale [0, 1] -> [-1, 1]

        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class BSplineModule(TransformModule):
    def __init__(
        self,
        *,
        n_intervals: int,
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: Optional[str] = None,
        min_interval: float = 1e-1,
        min_weight: float = 1e-3,
    ):
        super().__init__(
            n_params=2 * n_intervals + 2,
            net_hidden_shape=net_hidden_shape,
            net_activation=net_activation,
        )

        self.n_intervals = n_intervals
        self.min_interval = min_interval
        self.min_weight = min_weight

    def forward(self, k: Tensor | None = None) -> _BSplineTransform:
        params = self.params(k)

        intervals, weights = params.split(
            (self.n_intervals, self.n_intervals + 2),
            dim=1,
        )

        intervals = torch.sigmoid(intervals) + self.min_interval
        weights = F.softplus(weights) + self.min_weight

        Δ, ρ = intervals, weights

        Δ = Δ / Δ.sum(dim=-1, keepdim=True)
        ρ = ρ / (((ρ[..., :-2] + ρ[..., 1:-1] + ρ[..., 2:]) / 3) * Δ).sum(
            dim=-1, keepdim=True
        )

        Δpad = F.pad(Δ, (1, 1), "constant", 0)

        ω = (ρ[..., 1:] - ρ[..., :-1]) / (Δpad[..., :-1] + Δpad[..., 1:])
        h = ρ[..., 1:-1] * Δ + (1 / 3) * (ω[..., 1:] - ω[..., :-1]) * Δ**2

        zeros = Δ.new_zeros(
            (*Δ.shape[:-1], 1),
        )
        knots_x = torch.cat(
            (
                zeros,
                torch.cumsum(Δ, dim=-1),
            ),
            dim=-1,
        )
        knots_y = torch.cat(
            (
                zeros,
                torch.cumsum(h, dim=-1),
            ),
            dim=-1,
        )

        return _BSplineTransform(
            intervals=Δpad,
            weights=ρ,
            omega=ω,
            knots_x=knots_x,
            knots_y=knots_y,
        )
