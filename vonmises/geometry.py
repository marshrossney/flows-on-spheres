from math import pi as π
from typing import TypeAlias

import torch

from vonmises.utils import mod_2pi

Tensor: TypeAlias = torch.Tensor


def circle_vectors_to_angles(x: Tensor) -> Tensor:
    return mod_2pi(torch.atan2(*list(reversed(x.split(1, dim=-1)))))


def circle_angles_to_vectors(ϕ: Tensor) -> Tensor:
    return torch.cat([ϕ.cos(), ϕ.sin()], dim=-1)


def apply_global_rotation(xy: Tensor, θ: Tensor) -> Tensor:
    *data_shape, coord_dims = xy.shape
    assert coord_dims == 2
    cosθ, sinθ = θ.cos(), θ.sin()
    # counter-clockwise rotation
    R = torch.tensor(
        [[cosθ, -sinθ], [sinθ, cosθ]],
        device=xy.device,
    ).view(*[1 for _ in data_shape], 2, 2)
    xy.unsqueeze_(dim=-2)
    return (R * xy).sum(dim=-1)


def spherical_mesh(n: int, theta_uniform: bool = True) -> Tensor:
    theta = (
        torch.linspace(0, π, n)
        if theta_uniform
        else torch.acos(1 - 2 * torch.linspace(0, 1, n))
    )
    phi = torch.linspace(0, 2 * π, n)

    theta, phi = torch.meshgrid(theta, phi)

    x = theta.sin() * phi.cos()
    y = theta.sin() * phi.sin()
    z = theta.cos()

    return x, y, z
