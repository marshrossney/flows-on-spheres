from math import pi as π
from typing import Optional

import torch
import torch.nn.functional as F

from flows_on_spheres.geometry import (
    circle_vectors_to_angles,
    circle_angles_to_vectors,
)
from flows_on_spheres.nn import TransformModule

Tensor = torch.Tensor


class _RQSplineTransform:
    """
    References:
        Gregory, J. A. & Delbourgo, R. C2 Rational \
        Quadratic Spline Interpolation to Monotonic Data, IMA Journal of \
        Numerical Analysis, 1983, 3, 141-152
        """

    tol: float = 1e-5

    def __init__(
        self,
        knots_x: Tensor,
        knots_y: Tensor,
        knots_dydx: Tensor,
    ):
        _, n_knots = knots_x.shape
        assert knots_y.shape == knots_x.shape
        assert knots_dydx.shape == knots_x.shape
        assert (knots_dydx > 0).all()

        self.knots_x = knots_x
        self.knots_y = knots_y
        self.knots_dydx = knots_dydx

        self.n_segments = n_knots - 1

    def _get_segment(
        self, inputs: Tensor, inverse: bool
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        assert inputs.dim() == 2
        n_batch, n_coords = inputs.shape
        assert n_coords == 1

        knots = self.knots_y if inverse else self.knots_x

        outside_bounds = (
            inputs < knots.min(dim=1, keepdim=True).values - self.tol
        ) | (inputs > knots.max(dim=1, keepdim=True).values + self.tol)
        if outside_bounds.any():
            print(inputs[outside_bounds])
            raise ValueError("inputs outside of the spline bounds")

        # NOTE: calling contiguous() just to silence the warning message
        # don't really care about optimal perf
        i0 = (
            torch.searchsorted(
                knots.expand(n_batch, -1).contiguous(), inputs.contiguous()
            )
            - 1
        ).clamp_(0, self.n_segments - 1)
        i0_i1 = torch.stack((i0, i0 + 1), dim=0)

        x0_x1 = self.knots_x.expand(2, n_batch, -1).gather(-1, i0_i1)
        y0_y1 = self.knots_y.expand(2, n_batch, -1).gather(-1, i0_i1)
        d0_d1 = self.knots_dydx.expand(2, n_batch, -1).gather(-1, i0_i1)

        x0, x1 = x0_x1
        y0, y1 = y0_y1
        d0, d1 = d0_d1

        assert x0.shape == inputs.shape

        s = (y1 - y0) / (x1 - x0)

        return x0, x1, y0, y1, d0, d1, s

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x0, x1, y0, y1, d0, d1, s = self._get_segment(x, inverse=False)

        θx = (x - x0) / (x1 - x0)

        denominator = s + (d1 + d0 - 2 * s) * θx * (1 - θx)

        θy = (s * θx**2 + d0 * θx * (1 - θx)) / denominator

        y = y0 + (y1 - y0) * θy

        dydx = (
            s**2
            * (d1 * θx**2 + 2 * s * θx * (1 - θx) + d0 * (1 - θx) ** 2)
            / denominator**2
        )
        assert torch.all(dydx > 0)

        ldj = dydx.log().squeeze(1)

        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        x0, x1, y0, y1, d0, d1, s = self._get_segment(y, inverse=True)

        θy = (y - y0) / (y1 - y0)

        b = d0 - (d1 + d0 - 2 * s) * θy
        a = s - b
        c = -s * θy

        θx = (-2 * c) / (b + (b**2 - 4 * a * c).sqrt())

        x = x0 + (x1 - x0) * θx

        denominator = s + (d1 + d0 - 2 * s) * θx * (1 - θx)

        dydx = (
            s**2
            * (d1 * θx**2 + 2 * s * θx * (1 - θx) + d0 * (1 - θx) ** 2)
            / denominator**2
        )

        ldj = dydx.log().negative().squeeze(1)

        return x, ldj


class _RQSplineTransformCircular(_RQSplineTransform):
    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        ϕ = circle_vectors_to_angles(x)
        ϕ, ldj = super().__call__(ϕ)
        y = circle_angles_to_vectors(ϕ)
        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        ϕ = circle_vectors_to_angles(y)
        ϕ, ldj = super().__call__(ϕ)
        x = circle_angles_to_vectors(ϕ)
        return x, ldj


class RQSplineModule(TransformModule):
    def __init__(
        self,
        *,
        n_segments: int,
        circular: bool,
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: Optional[str] = None,
        min_slope: float = 1e-3,
    ):
        n_derivs = n_segments if circular else n_segments + 1
        super().__init__(
            n_params=(2 * n_segments + n_derivs),
            net_hidden_shape=net_hidden_shape,
            net_activation=net_activation,
        )
        self.n_segments = n_segments
        self.n_derivs = n_derivs
        self.circular = circular
        self.min_slope = min_slope
        self.lower_bound = 0 if circular else -1
        self.upper_bound = 2 * π if circular else +1

    def forward(
        self, k: Tensor | None = None
    ) -> _RQSplineTransform | _RQSplineTransformCircular:
        params = self.params(k)

        widths, heights, derivs = params.split(
            [self.n_segments, self.n_segments, self.n_derivs],
            dim=1,
        )

        # Normalise the widths and heights to the interval
        widths = F.softmax(widths, dim=1) * (
            self.upper_bound - self.lower_bound
        )
        heights = F.softmax(heights, dim=1) * (
            self.upper_bound - self.lower_bound
        )

        # Ensure the derivatives are positive and > min_slope
        derivs = F.softplus(derivs) + self.min_slope

        if self.circular:
            # match derivs at 0 and 2pi
            derivs = F.pad(derivs.unsqueeze(1), (0, 1), "circular").squeeze(1)

        zeros = widths.new_zeros((*widths.shape[:-1], 1))

        # Build the spline
        knots_x = torch.cat(
            (
                zeros,
                torch.cumsum(widths, dim=-1),
            ),
            dim=-1,
        ).add(self.lower_bound)
        knots_y = torch.cat(
            (
                zeros,
                torch.cumsum(heights, dim=-1),
            ),
            dim=-1,
        ).add(self.lower_bound)
        knots_dydx = derivs
        if not (knots_dydx > 0).all():
            print(knots_dydx[~(knots_dydx > 0)])

        if self.circular:
            return _RQSplineTransformCircular(knots_x, knots_y, knots_dydx)
        else:
            return _RQSplineTransform(knots_x, knots_y, knots_dydx)
