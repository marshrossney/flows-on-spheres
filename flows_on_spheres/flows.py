from math import pi as π
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

from flows_on_spheres.abc import Flow
from flows_on_spheres.geometry import rotate_2d
from flows_on_spheres.nn import TransformModule

Tensor: TypeAlias = torch.Tensor

Transformer = None


class CircularFlow(Flow):
    dim = 1

    def __init__(self, transforms: list[TransformModule]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.rotations = nn.ParameterList(
            [
                nn.Parameter(torch.empty(1).uniform_(0, 2 * π))
                for _ in transforms
            ]
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        xy = inputs
        ldj_full = torch.zeros(inputs.shape[0], device=inputs.device)

        for f, θ in zip(self.transforms, self.rotations):
            xy, ldj = f(None)(xy)
            xy = rotate_2d(xy, θ)
            ldj_full += ldj

        return xy, ldj_full


class RecursiveFlowS2(Flow):
    dim = 2

    def __init__(
        self,
        z_transforms: list[TransformModule],
        xy_transforms: list[TransformModule],
        *,
        softmax_beta: float = 1e8,
    ):
        super().__init__()
        assert len(z_transforms) == len(xy_transforms)
        self.z_transforms = nn.ModuleList(z_transforms)
        self.xy_transforms = nn.ModuleList(xy_transforms)
        self.rotations = nn.ParameterList(
            [
                nn.Parameter(torch.empty(1).uniform_(0, 2 * π))
                for _ in z_transforms
            ]
        )

        self.softmax_beta = softmax_beta

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        ldj_total = inputs.new_zeros(inputs.shape[0])

        xy, z = inputs.split([2, 1], dim=-1)

        # Map sphere to cylinder
        ρ = F.softplus(1 - z**2, beta=self.softmax_beta).sqrt()
        xy = xy / ρ

        for z_transform, xy_transform, θ in zip(
            self.z_transforms, self.xy_transforms, self.rotations, strict=True
        ):
            z, ldj_z = z_transform(None)(z)

            xy, ldj_xy = xy_transform(z)(xy)

            xy = rotate_2d(xy, θ)

            ldj_total += ldj_z + ldj_xy

        # Back to sphere
        ρ = F.softplus(1 - z**2, beta=self.softmax_beta).sqrt()
        xy = ρ * xy

        outputs = torch.cat([xy, z], dim=-1)

        return outputs, ldj_total

    def pushforward(self, vectors: Tensor) -> Tensor:
        ...

        # mm Jacobian of sphere to cylinder
        # mm Jacobian inverse of inner flow
        # mm Jacobian of cylinder to sphere
