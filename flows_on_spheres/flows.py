from math import pi as π
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

from flows_on_spheres.abc import Flow
from flows_on_spheres.geometry import as_angle, as_vector, mod_2pi
from flows_on_spheres.nn import TransformModule, CircularTransformModule
from flows_on_spheres.linalg import norm_keepdim

Tensor: TypeAlias = torch.Tensor
BoolTensor: TypeAlias = torch.BoolTensor


class CircularFlow(Flow):
    dim = 1

    def __init__(self, transform: CircularTransformModule):
        super().__init__()
        self.transform = transform
        self.rotation = nn.Parameter(torch.empty(1).uniform_(0, 2 * π))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        f, θ = self.transform, self.rotation

        ϕ, ldj = f(None)(as_angle(x))

        y = as_vector(mod_2pi(ϕ + θ))

        return y, ldj


class RecursiveFlowS2(Flow):
    dim = 2

    def __init__(
        self,
        interval_transform: TransformModule,
        circular_transform: CircularTransformModule,
        ordering: list[int],
        epsilon: float = 1e-3,
    ):
        super().__init__()
        self.interval_transform = interval_transform
        self.circular_transform = circular_transform
        self.rotation = nn.Parameter(torch.empty(1).uniform_(0, 2 * π))

        assert len(ordering) == self.dim + 1
        assert len(set(ordering)) == self.dim + 1
        assert min(ordering) == 1
        assert max(ordering) == self.dim + 1

        self.ordering = [i - 1 for i in ordering]  # zero indexing
        self.inverse_ordering = sorted(
            range(self.dim + 1), key=self.ordering.__getitem__
        )
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x[:, self.ordering]
        x3, x1_x2 = x.tensor_split([1], dim=1)
        ϕ12 = as_angle(x1_x2)

        y3, ldj_3 = self.interval_transform(None)(x3)
        ψ12, ldj_12 = self.circular_transform(y3)(ϕ12)
        z1_z2 = as_vector(ψ12)  # + self.rotation)  # unit circle

        # False elements are very close to poles +/- 1
        safe_mask = (1 - y3**2) > self.epsilon**2

        # Double where to avoid NaNs in gradients
        r3_masked = torch.where(
            safe_mask, 1 - y3**2, self.epsilon**2
        ).sqrt()
        assert r3_masked.isfinite().all()
        y1_y2 = torch.where(
            safe_mask,
            z1_z2 * r3_masked,
            z1_z2 * self.epsilon,
        )

        y = torch.cat([y3, y1_y2], dim=1)
        y = y[:, self.inverse_ordering]

        ldj_total = ldj_3 + ldj_12

        return y, ldj_total


class RecursiveFlowSD(Flow):
    def __init__(
        self,
        interval_transforms: list[TransformModule],
        circular_transform: CircularTransformModule,
        ordering: list[int],
        epsilon: float = 1e-3,
    ):
        super().__init__()

        D = len(interval_transforms) + 1
        assert D > 1
        assert len(ordering) == D + 1
        assert max(ordering) == D + 1
        assert min(ordering) == 1

        self.interval_transforms = nn.ModuleList(interval_transforms)
        self.circular_transform = circular_transform
        self.rotation = nn.Parameter(torch.empty(1).uniform_(0, 2 * π))

        self.ordering = [i - 1 for i in ordering]  # zero indexing
        self.inverse_ordering = sorted(
            range(D + 1), key=self.ordering.__getitem__
        )

        self._dim = D
        self.epsilon = epsilon

    @property
    def dim(self) -> int:
        return self._dim

    def _safe_mask(self, x: Tensor) -> BoolTensor:
        return (1 - x**2) > self.epsilon**2

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        assert x.dim() == 2
        assert x.shape[1] == self.dim + 1

        D = self.dim
        ε = self.epsilon

        ldj_total = x.new_zeros(x.shape[0])

        x = x[:, self.ordering]

        # Unwrap onto cylinder, counting down from i=D to i=2
        x_sphere = x
        x_intervals = []
        scale_factors = []
        for i in range(D, 1, -1):
            x_i, x_sphere = x_sphere.tensor_split([1], dim=1)
            x_intervals.append(x_i)

            safe_mask = self._safe_mask(x_i)
            ρ_i = torch.where(safe_mask, 1 - x_i**2, ε**2).sqrt()
            assert ρ_i.isfinite().all()

            x_sphere = torch.where(safe_mask, x_sphere / ρ_i, x_sphere / ε)

            scale_factors.append(ρ_i)

        # Should just have circular components remaining
        assert x_sphere.shape[1] == 2
        x_circle = x_sphere

        zip_x_and_transform = zip(x_intervals, self.interval_transforms)

        # Unconditionally transform the first component
        x1, f1 = next(zip_x_and_transform)
        y1, ldj_1 = f1(None)(x1)
        ldj_total += ldj_1

        # Transform remaining, conditioned on those already transformed
        y_intervals = y1
        for x_i, f_i in zip_x_and_transform:
            y_i, ldj_i = f_i(y_intervals)(x_i)
            ldj_total += ldj_i

            y_intervals = torch.cat([y_intervals, y_i], dim=1)

        # Transform circular part, conditioned on all interval parts
        x_D = as_angle(x_circle)
        y_D, ldj_D = self.circular_transform(y_intervals)(x_D)
        ldj_total += ldj_D
        y_circle = as_vector(y_D + self.rotation)

        # Wrap back onto the sphere, counting up from i=2 to i=D
        y_sphere = y_circle
        scale_factors_inverse = []
        for i in range(2, D + 1, +1):
            y_intervals, y_i = y_intervals.tensor_split([-1], dim=1)

            safe_mask = self._safe_mask(y_i)
            r_i = torch.where(safe_mask, 1 - y_i**2, ε**2).sqrt()
            assert r_i.isfinite().all()

            y_sphere = torch.where(safe_mask, y_sphere * r_i, y_sphere * ε)

            y_sphere = torch.cat([y_i, y_sphere], dim=1)  # prepended!
            scale_factors_inverse.insert(0, r_i)  # prepended!

        assert y_intervals.numel() == 0
        assert y_sphere.shape[1] == self.dim + 1

        # reorder
        y = y_sphere[:, self.inverse_ordering]

        # Compute ldj for the cylinder->sphere transformation and inverse
        # Take advantage of cancellation of large ρ and r near the poles
        for D_i, ρ_i, r_i in zip(
            range(D, 1, -1),
            scale_factors,
            scale_factors_inverse,
            strict=True,
        ):
            ldj_total -= (D_i - 2) * torch.log(ρ_i / r_i).squeeze(1)

        return y, ldj_total


class Composition(Flow):
    def __init__(self, flows: list[Flow]):
        super().__init__()
        assert len(set([flow.dim for flow in flows])) == 1
        self.flows = nn.ModuleList(flows)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        ldj_total = x.new_zeros(len(x))
        for flow in self.flows:
            y, ldj = flow(x)
            ldj_total += ldj
            x = y
        return y, ldj_total
