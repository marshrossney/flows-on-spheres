from math import pi as π
from typing import TypeAlias

import torch
import torch.linalg as LA

from flows_on_spheres.utils import mod_2pi
from flows_on_spheres.linalg import dot, cross, mv

Tensor: TypeAlias = torch.Tensor


def as_angle(x: Tensor) -> Tensor:
    return mod_2pi(torch.atan2(*list(reversed(x.split(1, dim=-1)))))


def as_vector(ϕ: Tensor) -> Tensor:
    return torch.cat([ϕ.cos(), ϕ.sin()], dim=-1)


def circle_vectors_to_angles(x: Tensor) -> Tensor:
    return as_angle(x)


def circle_angles_to_vectors(ϕ: Tensor) -> Tensor:
    return as_vector(ϕ)


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
    xdoty = dot(x, y, keepdim=True)
    xcrossy = cross(x, y)

    cosθ = xdoty
    sinθ = LA.vector_norm(xcrossy, dim=-1, keepdim=True)
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

    rej = y - xdoty * x

    u = x
    v = rej / LA.vector_norm(rej, dim=-1, keepdim=True)
    w = xcrossy / sinθ
    F = torch.stack([u, v, w], dim=-2)
    Finv = F.transpose(-2, -1)

    R = torch.matmul(Finv, torch.matmul(G, F))

    return R


def rotate_2d(x: Tensor, θ: Tensor) -> Tensor:
    # NOTE: for now it is assumed that the last 2 dimensions define the
    # 2d plane for rotation
    _, coord_dims = x.shape
    R = torch.block_diag(
        torch.eye(coord_dims - 2, device=θ.device),
        torch.eye(2, device=θ.device) * θ.cos()
        + torch.tensor([[0, -1], [1, 0]], device=θ.device) * θ.sin(),
    )
    return mv(R, x)


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
