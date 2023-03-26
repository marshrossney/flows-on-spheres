from typing import TypeAlias

import torch

from vonmises.utils.torchops import mod_2pi

Tensor: TypeAlias = torch.Tensor

__all__ = (
    "circle_vectors_to_angles",
    "circle_angles_to_vectors",
    "apply_global_rotation",
    "spherical_mesh",
)


def circle_vectors_to_angles(x: Tensor) -> Tensor:
    return mod_2pi(torch.atan2(*list(reversed(x.split(1, dim=-1)))))


def circle_angles_to_vectors(ϕ: Tensor) -> Tensor:
    return torch.cat([ϕ.cos(), ϕ.sin()], dim=-1)


def apply_global_rotation(xy_or_ϕ: Tensor, θ: Tensor) -> Tensor:
    *data_shape, coord_dims = xy_or_ϕ.shape

    if coord_dims == 1:  # working with angles
        assert coord_dims == 1
        ϕ = xy_or_ϕ
        return mod_2pi(ϕ + θ)
    elif coord_dims == 2:  # working with euclidean coords
        assert coord_dims == 2
        xy = xy_or_ϕ
        cosθ, sinθ = θ.cos(), θ.sin()
        # counter-clockwise rotation
        R = torch.tensor(
            [[cosθ, -sinθ], [sinθ, cosθ]],
            device=xy.device,
        ).view(*[1 for _ in data_shape], 2, 2)
        xy.unsqueeze_(dim=-2)
        return (R * xy).sum(dim=-1)
    else:
        raise ValueError("input has too many dimensions")


def spherical_mesh(n: int, theta_uniform: bool = True) -> Tensor:
    theta = (
        torch.linspace(0, PI, n)
        if theta_uniform
        else torch.acos(1 - 2 * torch.linspace(0, 1, n))
    )
    phi = torch.linspace(0, 2 * PI, n)

    theta, phi = torch.meshgrid(theta, phi)

    x = theta.sin() * phi.cos()
    y = theta.sin() * phi.sin()
    z = theta.cos()

    return x, y, z


## Angles to vectors for higher dimensional spheres
