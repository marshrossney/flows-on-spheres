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

    def _safe_mask(self, x: Tensor) -> BoolTensor:
        return (1 - x**2) > self.epsilon**2

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x[:, self.ordering]
        x3, x1_x2 = x.tensor_split([1], dim=1)
        ϕ12 = as_angle(x1_x2)

        y3, ldj_3 = self.interval_transform(None)(x3)
        ψ12, ldj_12 = self.circular_transform(y3)(ϕ12)
        z1_z2 = as_vector(ψ12)# + self.rotation)  # unit circle

        # False elements are very close to poles +/- 1
        safe_mask = self._safe_mask(y3)

        # Double where to avoid NaNs in gradients
        r3_masked = torch.where(safe_mask, 1 - y3**2, self.epsilon**2).sqrt()
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
        transforms: list[TransformModule],
        ordering: list[int],
        softplus_beta: float = 1e8,
    ):
        super().__init__()

        D = len(transforms)
        assert D > 1
        assert len(ordering) == D + 1
        assert max(ordering) == D + 1
        assert min(ordering) == 1

        # TODO: check domains of transforms
        # should be all [-1, 1] except final one

        self.transforms = nn.ModuleList(transforms)
        self.rotation = nn.Parameter(torch.empty(1).uniform_(0, 2 * π))

        self.ordering = [i - 1 for i in ordering]  # zero indexing
        self.inverse_ordering = sorted(
            range(D + 1), key=self.ordering.__getitem__
        )

        self.softplus_beta = softplus_beta
        self._dim = D

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        assert x.dim() == 2
        assert x.shape[1] == self.dim + 1

        ldj_total = x.new_zeros(x.shape[0])

        x = x[:, self.ordering]

        # Unwrap onto cylinder
        x_sphere = x
        x_intervals = []
        scale_factors = []
        for D in range(self.dim, 1, -1):
            x_i, x_sphere = x_sphere.tensor_split([1], dim=1)
            x_intervals.append(x_i)

            ρ_i = F.softplus(1 - x_i**2, beta=self.softplus_beta).sqrt()
            x_sphere = x_sphere / ρ_i

            scale_factors.append(ρ_i)

            # ldj_total -= (D - 2) * ρ_i.log().squeeze(1)

        assert x_sphere.shape[1] == 2
        x_circle = x_sphere

        zip_x_and_transform = zip(x_intervals + [x_circle], self.transforms)

        x1, f1 = next(zip_x_and_transform)

        # Unconditionally transform the first component
        y1, ldj_1 = f1(None)(x1)
        ldj_total += ldj_1

        y_cylinder = y1
        for x_i, f_i in zip_x_and_transform:
            y_i, ldj_i = f_i(y_cylinder)(x_i)
            ldj_total += ldj_i

            y_cylinder = torch.cat([y_cylinder, y_i], dim=1)

        y_cylinder = rotate_2d(y_cylinder, self.rotation)

        y_intervals, y_sphere = y_cylinder.tensor_split([-2], dim=1)
        scale_factors_inverse = []
        for D in range(2, self.dim + 1, +1):
            y_intervals, y_i = y_intervals.tensor_split([-1], dim=1)

            r_i = F.softplus(1 - y_i**2, beta=self.softplus_beta).sqrt()
            y_sphere = y_sphere * r_i

            y_sphere = torch.cat([y_i, y_sphere], dim=1)  # prepend!!

            scale_factors_inverse.insert(0, r_i)  # prepend!!

            # ldj_total += (D - 2) * r_i.log().squeeze(1)

        assert y_intervals.numel() == 0
        assert y_sphere.shape[1] == self.dim + 1

        # reorder
        y = y_sphere[:, self.inverse_ordering]

        # NOTE: the idea is to compute the log-det-jacobian from the
        # sphere<->cylinder mappings here, taking advantage of cancellation
        # Should also linearise where ρ is very small, but I need to figure out how
        # this works with the softplus I'm currently using
        for D, ρ, r in zip(
            range(self.dim, 1, -1),
            scale_factors,
            scale_factors_inverse,
            strict=True,
        ):
            ldj_total -= (D - 2) * torch.log(ρ / r).squeeze(1)

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
