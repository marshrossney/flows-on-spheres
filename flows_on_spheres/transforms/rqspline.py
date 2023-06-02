import math
from math import pi as π
from typing import Optional
import warnings

import torch
import torch.nn.functional as F

from flows_on_spheres.geometry import (
    circle_vectors_to_angles,
    circle_angles_to_vectors,
)
from flows_on_spheres.nn import TransformModule, CircularTransformModule

Tensor = torch.Tensor

warnings.filterwarnings("ignore", message="torch.searchsorted")


class RQSplineTransform:
    """
    References:
        Gregory, J. A. & Delbourgo, R. C2 Rational \
        Quadratic Spline Interpolation to Monotonic Data, IMA Journal of \
        Numerical Analysis, 1983, 3, 141-152
        """

    _tol: float = 1e-4

    def __init__(
        self,
        widths: Tensor,
        heights: Tensor,
        derivs: Tensor,
        *,
        lower_bound: float,
        upper_bound: float,
        periodic: bool = False,
        linear_tails: bool = False,
        min_width: float = 1e-3,
        min_height: float = 1e-3,
        min_deriv: float = 1e-3,
    ):
        assert widths.isfinite().all()
        assert heights.isfinite().all()
        assert derivs.isfinite().all()

        assert widths.shape == heights.shape
        assert widths.shape[:-1] == derivs.shape[:-1]

        n_bins = widths.shape[-1]
        assert min_width * n_bins < 1
        assert min_height * n_bins < 1

        assert lower_bound < upper_bound

        assert not (periodic and linear_tails)

        # Normalise the widths and heights to the interval
        widths = (
            F.softmax(widths, dim=1) * (1 - min_width * n_bins) + min_width
        ) * (upper_bound - lower_bound)
        heights = (
            F.softmax(heights, dim=1) * (1 - min_height * n_bins) + min_height
        ) * (upper_bound - lower_bound)

        # Ensure the derivatives are positive and > min_slope
        # Specifying β = log(2) / (1 - ε) means softplus(0, β) = 1
        derivs = (
            F.softplus(derivs, beta=math.log(2) / (1 - min_deriv)) + min_deriv
        )

        n_derivs = derivs.shape[-1]
        if periodic:
            assert n_derivs == n_bins
            derivs = F.pad(derivs.unsqueeze(1), (0, 1), "circular").squeeze(1)
        elif linear_tails:
            assert n_derivs == n_bins - 1
            derivs = F.pad(derivs, (1, 1), "constant", 1.0)
        else:
            assert n_derivs == n_bins + 1

        # Build the spline
        zeros = widths.new_zeros((*widths.shape[:-1], 1))
        knots_x = torch.cat(
            (
                zeros,
                torch.cumsum(widths, dim=-1),
            ),
            dim=-1,
        ).add(lower_bound)
        knots_y = torch.cat(
            (
                zeros,
                torch.cumsum(heights, dim=-1),
            ),
            dim=-1,
        ).add(lower_bound)

        self.knots_x = knots_x
        self.knots_y = knots_y
        self.knots_dydx = derivs

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_bins = n_bins

    def _check_bounds(self, inputs: Tensor) -> Tensor:
        outside_bounds = (inputs < self.lower_bound) | (
            inputs > self.upper_bound
        )
        if outside_bounds.any():
            print("input outside bounds!", inputs[outside_bounds])
            raise ValueError("inputs outside of the spline bounds")

        # NOTE: For linear tails I need to add a function that
        # creates a masked transformation

        inputs = inputs.clamp(
            self.lower_bound + self._tol, self.upper_bound - self._tol
        )
        return inputs

    def _get_segment(
        self, inputs: Tensor, inverse: bool
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        assert inputs.dim() == 2
        n_batch, n_coords = inputs.shape
        assert n_coords == 1

        knots = self.knots_y if inverse else self.knots_x

        i0 = torch.searchsorted(knots.expand(n_batch, -1), inputs) - 1
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
        x = self._check_bounds(x)
        x0, x1, y0, y1, d0, d1, s = self._get_segment(x, inverse=False)

        θx = (x - x0) / (x1 - x0)
        assert (θx >= 0).all()
        assert (θx <= 1).all()

        denom = s + (d1 + d0 - 2 * s) * θx * (1 - θx)

        θy = (s * θx**2 + d0 * θx * (1 - θx)) / denom

        y = y0 + (y1 - y0) * θy

        dydx = (
            s**2
            * (d1 * θx**2 + 2 * s * θx * (1 - θx) + d0 * (1 - θx) ** 2)
            / denom**2
        )
        assert torch.all(dydx > 0)

        ldj = dydx.log().squeeze(1)

        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        y = self._check_bounds(y)
        x0, x1, y0, y1, d0, d1, s = self._get_segment(y, inverse=True)

        θy = (y - y0) / (y1 - y0)

        b = d0 - (d1 + d0 - 2 * s) * θy
        a = s - b
        c = -s * θy

        θx = (-2 * c) / (b + (b**2 - 4 * a * c).sqrt())

        x = x0 + (x1 - x0) * θx

        denom = s + (d1 + d0 - 2 * s) * θx * (1 - θx)

        dydx = (
            s**2
            * (d1 * θx**2 + 2 * s * θx * (1 - θx) + d0 * (1 - θx) ** 2)
            / denom**2
        )

        ldj = dydx.log().negative().squeeze(1)

        return x, ldj


class RQSplineModule(TransformModule):
    def __init__(
        self,
        *,
        n_segments: int,
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: Optional[str] = None,
        min_width: float = 1e-3,
        min_height: float = 1e-3,
        min_deriv: float = 1e-3,
    ):
        super().__init__(
            n_params=3 * n_segments + 1,
            net_hidden_shape=net_hidden_shape,
            net_activation=net_activation,
        )
        self.n_segments = n_segments
        self.min_width = min_width
        self.min_height = min_height
        self.min_deriv = min_deriv

    def forward(self, k: Tensor | None = None) -> RQSplineTransform:
        params = self.params(k)
        widths, heights, derivs = params.split(
            [self.n_segments, self.n_segments, self.n_segments + 1],
            dim=1,
        )
        return RQSplineTransform(
            widths,
            heights,
            derivs,
            lower_bound=-1.0,
            upper_bound=+1.0,
            min_width=self.min_width,
            min_height=self.min_height,
            min_deriv=self.min_deriv,
        )


class CircularSplineModule(CircularTransformModule):
    def __init__(
        self,
        *,
        n_segments: int,
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: Optional[str] = None,
        min_width: float = 1e-3,
        min_height: float = 1e-3,
        min_deriv: float = 1e-3,
    ):
        super().__init__(
            n_params=3 * n_segments,
            net_hidden_shape=net_hidden_shape,
            net_activation=net_activation,
        )
        self.n_segments = n_segments
        self.min_width = min_width
        self.min_height = min_height
        self.min_deriv = min_deriv

    def forward(self, k: Tensor | None = None) -> RQSplineTransform:
        params = self.params(k)
        widths, heights, derivs = params.split(
            [self.n_segments, self.n_segments, self.n_segments],
            dim=1,
        )
        return RQSplineTransform(
            widths,
            heights,
            derivs,
            lower_bound=0,
            upper_bound=2 * π,
            periodic=True,
            min_width=self.min_width,
            min_height=self.min_height,
            min_deriv=self.min_deriv,
        )
