from itertools import chain
import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.linalg as LA

Tensor = torch.Tensor

PI = math.pi


class MobiusMixtureTransform:
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
        params = params.view(*params.shape[:-2], self.n_mixture, -1)
        if params.shape[-1] == 3:
            omega, rho = params.split([2, 1], dim=-1)
            rho = torch.softmax(rho, dim=-2)
        elif params.shape[-1] == 2:
            omega = params
            rho = 1 / self.n_mixture
        else:
            raise ValueError("Wrong number of parameters provided")

        omega = torch.tanh(omega) * (1 - self.epsilon)
        omega1, omega2 = omega.split(1, dim=-1)
        omega = torch.cat([omega1, omega2 * (1 - omega1.pow(2)).sqrt()], dim=-1)

        return omega, rho

    def forward(self, x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        *data_shape, n_coords = x.shape
        assert n_coords == 2
        assert params.shape == torch.Size([*data_shape, 2, self.n_params // 2])

        omega, rho = self._constrain_parameters(params)

        assert torch.all(
            omega.pow(2).sum(dim=-1) < 1
        ), f"{omega.pow(2).sum(dim=-1).abs().max()}"

        x.unsqueeze_(dim=-2)
        x_10 = torch.tensor([1.0, 0.0], device=x.device).view(
            *[1 for _ in data_shape], 1, 2
        )

        assert x.shape == torch.Size([*data_shape, 1, 2])
        if type(rho) is Tensor:
            assert rho.shape == torch.Size([*data_shape, self.n_mixture, 1])
        assert omega.shape == torch.Size([*data_shape, self.n_mixture, 2])

        one_minus_modsq_omega = 1 - omega.pow(2).sum(dim=-1, keepdim=True)
        modsq_x_minus_omega = (x - omega).pow(2).sum(dim=-1, keepdim=True)
        modsq_x_10_minus_omega = (x_10 - omega).pow(2).sum(dim=-1, keepdim=True)
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
            theta = torch.fmod(torch.atan2(g2_rot, g1_rot) + 2 * PI, 2 * PI)
            theta_mean = (rho * theta).sum(dim=-2)
            y = torch.cat([theta_mean.cos(), theta_mean.sin()], dim=-1)
            log_delta_vol = (
                (rho * vol_factor).sum(dim=-2).log().flatten(start_dim=1).sum(dim=1)
            )

        assert y.shape == torch.Size([*data_shape, 2])

        return y, log_delta_vol

    def __call__(self, x, params):
        return self.forward(x, params)


class RQSplineTransform:
    """
    Pointwise rational quadratic spline transformation.
    """

    def __init__(self, interval: tuple[float], n_segments: int) -> None:
        self.interval = interval
        self.n_segments = n_segments

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
        interval_size = self.interval[1] - self.interval[0]
        widths = torch.nn.functional.softmax(widths, dim=-1).mul(interval_size)
        heights = torch.nn.functional.softmax(heights, dim=-1).mul(interval_size)

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
        ).add(self.interval[0])

        knots_ycoords = torch.cat(
            (
                zeros,
                torch.cumsum(heights, dim=-1),
            ),
            dim=-1,
        ).add(self.interval[0])

        return widths, heights, derivs, knots_xcoords, knots_ycoords

    def forward(self, x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
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

        segment_idx = torch.searchsorted(knots_xcoords, x.unsqueeze(-1)) - 1
        segment_idx.clamp_(0, self.n_segments - 1)

        # Get parameters of the segments that x falls in
        w = torch.gather(widths, -1, segment_idx).squeeze(-1)
        h = torch.gather(heights, -1, segment_idx).squeeze(-1)
        d0 = torch.gather(derivs, -1, segment_idx).squeeze(-1)
        d1 = torch.gather(derivs, -1, segment_idx + 1).squeeze(-1)
        x0 = torch.gather(knots_xcoords, -1, segment_idx).squeeze(-1)
        y0 = torch.gather(knots_ycoords, -1, segment_idx).squeeze(-1)

        s = h / w
        s.clamp(1e-4, 1 - 1e-4)  # don't want tiny slopes

        alpha = (x - x0) / w
        alpha.clamp_(1e-4, 1 - 1e-4)

        denominator_recip = torch.reciprocal(
            s + (d1 + d0 - 2 * s) * alpha * (1 - alpha)
        )

        beta = (s * alpha.pow(2) + d0 * alpha * (1 - alpha)) * denominator_recip
        y = y0 + h * beta

        gradient = (
            s.pow(2)
            * (
                d1 * alpha.pow(2)
                + 2 * s * alpha * (1 - alpha)
                + d0 * (1 - alpha).pow(2)
            )
            * denominator_recip.pow(2)
        )

        if gradient.less(0.0).any():
            log.warning("Gradient has negative values")
            gradient = gradient.clamp(min=1e-6)

        ldj = gradient.log().flatten(start_dim=1).sum(dim=1)

        return y, ldj

    def __call__(self, x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward(x, params)

    def visualise(self, params: Tensor) -> plt.Figure:
        assert params.dim() == 2


class RQSplineTransformCircularDomain(RQSplineTransform):
    """
    Rational quadratic spline transformation on the circle.
    """

    def __init__(self, n_segments: int):
        super().__init__((0, 2 * PI), n_segments)

    @property
    def n_knots(self) -> int:
        return self.n_segments

    @staticmethod
    def pad_derivs(derivs: Tensor) -> Tensor:
        return F.pad(derivs.flatten(1, -2), (0, 1), "circular").view(
            *derivs.shape[:-1], -1
        )


class BSplineTransform:
    def __init__(self, interval: tuple[float], n_segments: int) -> None:
        self.interval = interval
        self.n_segments = n_segments

    @property
    def identity_params(self) -> Tensor:
        return torch.ones(self.n_segments * 2 + 2)

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
        I = ρ[..., 1:-1] * Δ + (1 / 3) * (ω[..., 1:] - ω[..., :-1]) * Δ**2

        zeros = torch.zeros_like(I).sum(dim=-1, keepdim=True)

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
                torch.cumsum(I, dim=-1),
            ),
            dim=-1,
        )

        return Δpad, ρ, ω, X, Y

    def forward(self, x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        x = x.contiguous()

        x = x - self.interval[0]
        x = x / (self.interval[1] - self.interval[0])

        intervals, weights = params.split(
            (self.n_segments, self.n_segments + 2),
            dim=-1,
        )
        Δ, ρ, ω, X, Y = self.build_spline(intervals, weights)

        ix = torch.searchsorted(X, x.unsqueeze(-1), side="right")
        ix.clamp_(1, self.n_segments)

        # Get parameters of the segments that x falls in
        Δ = torch.gather(Δ, -1, ix).squeeze(-1)
        ρ = torch.gather(ρ, -1, ix).squeeze(-1)
        ωi = torch.gather(ω, -1, ix).squeeze(-1)
        ωim1 = torch.gather(ω, -1, ix - 1).squeeze(-1)
        x0 = torch.gather(X, -1, ix - 1).squeeze(-1)
        y0 = torch.gather(Y, -1, ix - 1).squeeze(-1)

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

        y = y * (self.interval[1] - self.interval[0])
        y = y + self.interval[0]

        return y, ldj

    def __call__(self, x: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward(x, params)
