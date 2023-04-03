from math import pi as π
from typing import TypeAlias

import torch

from flows_on_spheres.utils import mod_2pi

Tensor: TypeAlias = torch.Tensor


def circle_vectors_to_angles(xy: Tensor) -> Tensor:
    return mod_2pi(torch.atan2(*list(reversed(xy.split(1, dim=-1)))))


def circle_angles_to_vectors(ϕ: Tensor) -> Tensor:
    return torch.cat([ϕ.cos(), ϕ.sin()], dim=-1)


def sphere_vectors_to_angles(xyz: Tensor) -> Tensor:
    x, y, z = xyz.split(1, dim=-1)
    return torch.cat([torch.asin(z), mod_2pi(torch.atan2(y, x))], dim=-1)


def sphere_angles_to_vectors(θϕ: Tensor) -> Tensor:
    θ, ϕ = θϕ.split(1, dim=-1)
    return torch.cat(
        [
            θ.sin() * ϕ.cos(),
            θ.sin() * ϕ.sin(),
            θ.cos(),
        ],
        dim=-1,
    )


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


def spherical_mesh(n: int) -> Tensor:
    theta = torch.acos(1 - 2 * torch.linspace(0 + 1e-3, 1 - 1e-3, n))
    phi = torch.linspace(0 + 1e-3, 2 * π - 1e-3, n)

    theta, phi = torch.meshgrid(theta, phi)

    xyz = torch.stack(
        [
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            theta.cos(),
        ],
        dim=-1,
    )

    return xyz
