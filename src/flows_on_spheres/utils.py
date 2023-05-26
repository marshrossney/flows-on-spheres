from itertools import chain
from math import pi as π
from typing import TypeAlias

import torch
import torch.linalg as LA

Tensor: TypeAlias = torch.Tensor


def make_fnn(
    in_features: int,
    out_features: int,
    hidden_shape: list[int],
    activation: str = "Tanh",
):
    layers = [
        torch.nn.Linear(f_in, f_out)
        for f_in, f_out in zip(
            [in_features, *hidden_shape], [*hidden_shape, out_features]
        )
    ]
    activation = getattr(torch.nn, activation)
    activations = [activation() for _ in hidden_shape] + [torch.nn.Identity()]

    return torch.nn.Sequential(*list(chain(*zip(layers, activations))))


def mod_2pi(angles: Tensor) -> Tensor:
    return torch.remainder(angles, 2 * π)


def batched_dot(x: Tensor, y: Tensor, keepdim: bool = False) -> Tensor:
    return (x * y).sum(dim=-1, keepdim=keepdim)

def batched_cross(x: Tensor, y: Tensor) -> Tensor:
    return LA.cross(x, y, dim=-1)


def batched_outer(x: Tensor, y: Tensor) -> Tensor:
    return x.unsqueeze(dim=-2) * y.unsqueeze(dim=-1)


def batched_mv(M: Tensor, v: Tensor) -> Tensor:
    return (M * v.unsqueeze(dim=-2)).sum(dim=-1)


def orthogonal_projection(x: Tensor) -> Tensor:
    # assert norm of x is 1
    return torch.eye(x.shape[-1], dtype=x.dtype, device=x.device) - batched_outer(x, x)


def project_onto_tangent(v: Tensor, x: Tensor) -> Tensor:
    return batched_mv(orthogonal_projection(x), v)
