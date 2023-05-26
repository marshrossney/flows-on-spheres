from math import pi as π
from typing import TypeAlias

import torch
import torch.linalg as LA

from flows_on_spheres.utils import (
    mod_2pi,
    batched_dot,
    batched_cross,
    batched_mv,
)

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


def get_rotation_matrix(x: Tensor, y: Tensor) -> Tensor:
    # norm = lambda v: torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    # print("|x| ", norm(x))
    # print("|y| ", norm(y))

    xdoty = batched_dot(x, y, keepdim=True)
    xcrossy = batched_cross(x, y)

    cosθ = xdoty
    sinθ = LA.vector_norm(xcrossy, dim=-1, keepdim=True)
    # print("sin2 + cos2 ", sinθ ** 2 + cosθ ** 2)
    zero = torch.zeros_like(xdoty)
    one = torch.ones_like(xdoty)
    G = torch.cat(
        [
            cosθ,
            -sinθ,
            zero,
            sinθ,
            cosθ,
            zero,
            zero,
            zero,
            one,
        ],
        dim=-1,
    ).view(*x.shape, 3)

    # print("G[0] ", G[0])
    # print("det G ", torch.det(G))

    rej = y - xdoty * x

    u = x
    v = rej / LA.vector_norm(rej, dim=-1, keepdim=True)
    w = xcrossy / sinθ
    F = torch.stack([u, v, w], dim=-2)
    Finv = F.transpose(-2, -1)

    # print("F[0] ", F[0])
    # print("det F ", torch.det(F))
    # print("Finv[0] ", Finv[0])
    # print("det Finv ", torch.det(Finv))

    R = torch.matmul(Finv, torch.matmul(G, F))

    # print("R[0] ", R[0])
    # print("det R ", torch.det(R))

    # print("x ", x)
    # print("y ", y)
    # print("Rx ", batched_mv(R, x))

    return R


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
