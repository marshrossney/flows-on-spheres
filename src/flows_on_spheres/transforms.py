from abc import ABC, abstractmethod
from math import pi as π
from typing import TypeAlias

import torch
import torch.nn.functional as F

from flows_on_spheres.geometry import (
    circle_vectors_to_angles,
    circle_angles_to_vectors,
)

Tensor: TypeAlias = torch.Tensor


class Transformer(ABC):
    @property
    @abstractmethod
    def identity_params(self) -> Tensor:
        ...

    @property
    @abstractmethod
    def n_params(self) -> int:
        ...

    @abstractmethod
    def forward(self, inputs: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        ...

    def __call__(
        self, inputs: Tensor, params: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self.forward(inputs, params)


class MobiusMixtureTransform(Transformer):
    def __init__(
        self, n_mixture: int, weighted: bool = True, epsilon: float = 1e-3
    ) -> None:
        self.n_mixture = n_mixture
        self.weighted = weighted and n_mixture > 1
        self.epsilon = epsilon

    @property
    def identity_params(self) -> Tensor:
        return torch.zeros(self.n_params)

    @property
    def n_params(self) -> int:
        return self.n_mixture * (2 + int(self.weighted))

    def _constrain_parameters(self, params: Tensor) -> tuple[Tensor, Tensor]:
        params = params.view(*params.shape[:-1], self.n_mixture, -1)
        if params.shape[-1] == 3:
            omega, rho = params.split([2, 1], dim=-1)
            rho = torch.softmax(rho, dim=-2)
        elif params.shape[-1] == 2:
            omega = params
            rho = 1 / self.n_mixture
        else:
            raise ValueError("Wrong number of parameters provided")

        omega = torch.tanh(omega) * (1 - self.epsilon)
        if omega.isnan().any():
            print("oh no")
        omega1, omega2 = omega.split(1, dim=-1)
        omega = torch.cat(
            [omega1, omega2 * (1 - omega1.pow(2)).sqrt()], dim=-1
        )
        if omega.isnan().any():
            print("oh no v2")

        return omega, rho

    def forward(self, x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        *data_shape, n_coords = x.shape
        assert n_coords == 2
        assert params.shape == torch.Size([*data_shape, self.n_params])

        omega, rho = self._constrain_parameters(params)

        if not torch.all(omega.pow(2).sum(dim=-1) < 1):
            print(f"Error in omega: {omega.pow(2).sum(dim=-1).abs().max()}")
            assert False

        x = x.unsqueeze(dim=-2)
        x_10 = torch.tensor([1.0, 0.0], device=x.device).view(
            *[1 for _ in data_shape], 1, 2
        )

        assert x.shape == torch.Size([*data_shape, 1, 2])
        if isinstance(rho, Tensor):
            assert rho.shape == torch.Size([*data_shape, self.n_mixture, 1])
        assert omega.shape == torch.Size([*data_shape, self.n_mixture, 2])

        one_minus_modsq_omega = 1 - omega.pow(2).sum(dim=-1, keepdim=True)
        modsq_x_minus_omega = (x - omega).pow(2).sum(dim=-1, keepdim=True)
        modsq_x_10_minus_omega = (
            (x_10 - omega).pow(2).sum(dim=-1, keepdim=True)
        )
        vol_factor = one_minus_modsq_omega / modsq_x_minus_omega
        vol_factor_10 = one_minus_modsq_omega / modsq_x_10_minus_omega

        g = vol_factor * (x - omega) - omega
        g_10 = vol_factor_10 * (x_10 - omega) - omega

        g1, g2 = g.split(1, dim=-1)
        g1_10, g2_10 = g_10.split(1, dim=-1)

        # Rotate such that g((1, 0)) = (1, 0)
        g1_rot = g1_10 * g1 + g2_10 * g2
        g2_rot = -g2_10 * g1 + g1_10 * g2

        if self.n_mixture == 1:
            y = torch.cat([g1_rot, g2_rot], dim=-1).squeeze(dim=-2)
            log_delta_vol = vol_factor.log().flatten(start_dim=1).sum(dim=1)
        else:
            theta = torch.fmod(torch.atan2(g2_rot, g1_rot) + 2 * π, 2 * π)
            theta_mean = (rho * theta).sum(dim=-2)
            y = torch.cat([theta_mean.cos(), theta_mean.sin()], dim=-1)
            log_delta_vol = (
                (rho * vol_factor)
                .sum(dim=-2)
                .log()
                .flatten(start_dim=1)
                .sum(dim=1)
            )

        assert y.shape == torch.Size([*data_shape, 2])

        return y, log_delta_vol


class RQSplineTransform(Transformer):
    """
    Pointwise rational quadratic spline transformation.
    """

    def __init__(self, n_segments: int) -> None:
        self.n_segments = n_segments

    @property
    def lower_boundary(self) -> float:
        return -1

    @property
    def upper_boundary(self) -> float:
        return 1

    @property
    def n_knots(self) -> int:
        return self.n_segments + 1

    @property
    def n_params(self) -> int:
        return 2 * self.n_segments + self.n_knots

    @property
    def identity_params(self) -> Tensor:
        return torch.cat(
            (
                torch.full(
                    size=(2 * self.n_segments,),
                    fill_value=1 / self.n_segments,
                ),
                (torch.ones(self.n_knots).exp() - 1).log(),
            ),
            dim=0,
        )

    @staticmethod
    def pad_derivs(derivs: Tensor) -> Tensor:
        return derivs

    def build_spline(
        self,
        widths: Tensor,
        heights: Tensor,
        derivs: Tensor,
    ) -> tuple[Tensor]:
        """Builds a rational quadratic spline function."""
        # Normalise the widths and heights to the interval
        interval_size = self.upper_boundary - self.lower_boundary
        widths = torch.nn.functional.softmax(widths, dim=-1).mul(interval_size)
        heights = torch.nn.functional.softmax(heights, dim=-1).mul(
            interval_size
        )

        # Let the derivatives be positive definite
        derivs = torch.nn.functional.softplus(derivs)
        derivs = self.pad_derivs(derivs)

        # Just a convenient way to ensure it's on the correct device
        zeros = torch.zeros_like(widths).sum(dim=-1, keepdim=True)

        knots_xcoords = torch.cat(
            (
                zeros,
                torch.cumsum(widths, dim=-1),
            ),
            dim=-1,
        ).add(self.lower_boundary)

        knots_ycoords = torch.cat(
            (
                zeros,
                torch.cumsum(heights, dim=-1),
            ),
            dim=-1,
        ).add(self.lower_boundary)

        return widths, heights, derivs, knots_xcoords, knots_ycoords

    def forward(self, x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        *data_shape, n_coords = x.shape
        assert params.shape[:-1] == torch.Size(data_shape)
        assert n_coords == 1
        assert params.shape[-1] == self.n_params

        x = x.contiguous()

        widths, heights, derivs = params.split(
            (self.n_segments, self.n_segments, self.n_knots),
            dim=-1,
        )
        (
            widths,
            heights,
            derivs,
            knots_xcoords,
            knots_ycoords,
        ) = self.build_spline(widths, heights, derivs)

        ix = torch.searchsorted(knots_xcoords, x) - 1
        ix.clamp_(0, self.n_segments - 1)

        # Get parameters of the segments that x falls in
        w = torch.gather(widths, -1, ix)
        h = torch.gather(heights, -1, ix)
        d0 = torch.gather(derivs, -1, ix)
        d1 = torch.gather(derivs, -1, ix + 1)
        X0 = torch.gather(knots_xcoords, -1, ix)
        Y0 = torch.gather(knots_ycoords, -1, ix)

        s = h / w
        # s.clamp(1e-4, 1 - 1e-4)  # don't want tiny slopes

        θ = (x - X0) / w
        # θ.clamp_(1e-4, 1 - 1e-4)

        denominator_recip = torch.reciprocal(
            s + (d1 + d0 - 2 * s) * θ * (1 - θ)
        )
        y = Y0 + h * (s * θ.pow(2) + d0 * θ * (1 - θ)) * denominator_recip

        gradient = (
            s.pow(2)
            * (d1 * θ.pow(2) + 2 * s * θ * (1 - θ) + d0 * (1 - θ).pow(2))
            * denominator_recip.pow(2)
        )

        if gradient.less(0.0).any():
            # TODO: log rather than print
            print("Gradient has negative values")
            gradient = gradient.clamp(min=1e-6)

        ldj = gradient.log().flatten(start_dim=1).sum(dim=1)

        return y, ldj


class CircularRQSplineTransform(RQSplineTransform):
    """
    Rational quadratic spline transformation on the circle.
    """

    def __init__(self, n_segments: int):
        super().__init__(n_segments)

    @property
    def lower_boundary(self) -> float:
        return 0

    @property
    def upper_boundary(self) -> float:
        return 2 * π

    @property
    def n_knots(self) -> int:
        return self.n_segments

    @staticmethod
    def pad_derivs(derivs: Tensor) -> Tensor:
        return F.pad(derivs.unsqueeze(1), (0, 1), "circular").squeeze(1)

    def forward(self, x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        ϕ = circle_vectors_to_angles(x)
        ϕ, ldj = super().forward(ϕ, params)
        x = circle_angles_to_vectors(ϕ)
        return x, ldj


class BSplineTransform(Transformer):
    def __init__(self, n_segments: int) -> None:
        self.n_segments = n_segments

    @property
    def lower_boundary(self) -> float:
        return -1

    @property
    def upper_boundary(self) -> float:
        return 1

    @property
    def identity_params(self) -> Tensor:
        return torch.ones(self.n_segments * 2 + 2)

    @property
    def n_params(self) -> int:
        return 2 * self.n_segments + 2

    def build_spline(
        self,
        intervals: Tensor,
        weights: Tensor,
    ) -> tuple[Tensor]:
        # Δ = F.softplus(intervals) + 1e-3
        Δ = torch.sigmoid(intervals) + 0.1
        Δ = Δ / Δ.sum(dim=-1, keepdim=True)
        Δpad = F.pad(Δ, (1, 1), "constant", 0)

        ρ = F.softplus(weights) + 1e-3  # min weight TODO make configurable
        ρ = ρ / (((ρ[..., :-2] + ρ[..., 1:-1] + ρ[..., 2:]) / 3) * Δ).sum(
            dim=-1, keepdim=True
        )

        ω = (ρ[..., 1:] - ρ[..., :-1]) / (Δpad[..., :-1] + Δpad[..., 1:])
        Int = ρ[..., 1:-1] * Δ + (1 / 3) * (ω[..., 1:] - ω[..., :-1]) * Δ**2

        zeros = torch.zeros_like(Int).sum(dim=-1, keepdim=True)

        X = torch.cat(
            (
                zeros,
                torch.cumsum(Δ, dim=-1),
            ),
            dim=-1,
        )
        Y = torch.cat(
            (
                zeros,
                torch.cumsum(Int, dim=-1),
            ),
            dim=-1,
        )

        return Δpad, ρ, ω, X, Y

    def forward(self, x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        x = x.contiguous()

        x = (x - self.lower_boundary) / (
            self.upper_boundary - self.lower_boundary
        )

        intervals, weights = params.split(
            (self.n_segments, self.n_segments + 2),
            dim=-1,
        )
        Δ, ρ, ω, X, Y = self.build_spline(intervals, weights)

        ix = torch.searchsorted(X, x, side="right")
        ix.clamp_(1, self.n_segments)

        # Get parameters of the segments that x falls in
        Δ = torch.gather(Δ, -1, ix)
        ρ = torch.gather(ρ, -1, ix)
        ωi = torch.gather(ω, -1, ix)
        ωim1 = torch.gather(ω, -1, ix - 1)
        x0 = torch.gather(X, -1, ix - 1)
        y0 = torch.gather(Y, -1, ix - 1)

        θ = (x - x0) / Δ

        y = (
            y0
            + ρ * Δ * θ
            - ωim1 * Δ**2 * θ * (1 - θ)
            + (1 / 3) * (ωi - ωim1) * Δ**2 * θ**3
        )

        gradient = ρ + ωi * Δ * θ**2 - ωim1 * Δ * (1 - θ) ** 2

        if gradient.less(0.0).any():
            values = gradient[gradient < 0]
            print(f"Gradient has negative values: {values.tolist()}")
            print(f"interval is: ix = {ix[gradient < 0]}")
            print(f"theta is: {θ[gradient < 0]}")
            print(f"Delta is: {Δ[gradient < 0]}")
            print(f"rho is: {ρ[gradient < 0]}")
            print(f"omega is: {ωi[gradient < 0]}")
            print(f"omega-1 is: {ωim1[gradient < 0]}")
            gradient = gradient.clamp(min=1e-6)

        ldj = gradient.log().flatten(start_dim=1).sum(dim=1)

        y = (
            y * (self.upper_boundary - self.lower_boundary)
            + self.lower_boundary
        )

        return y, ldj
