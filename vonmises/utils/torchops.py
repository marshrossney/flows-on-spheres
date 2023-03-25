from math import pi as π
from typing import TypeAlias

import torch

Tensor: TypeAlias = torch.Tensor

__all__ = (
    "mod_2pi",
    "batched_dot",
    "batched_outer",
    "batched_mv",
)

def mod_2pi(angles: Tensor) -> Tensor:
    return torch.remainder(angles, 2 * π)


def batched_dot(x: Tensor, y: Tensor) -> Tensor:
    return (x * y).sum(dim=-1)


def batched_outer(x: Tensor, y: Tensor) -> Tensor:
    return x.unsqueeze(dim=-2) * y.unsqueeze(dim=-1)


def batched_mv(M: Tensor, v: Tensor) -> Tensor:
    return (M * v.unsqueeze(dim=-2)).sum(dim=-1)


# M = torch.rand(3, 3)
# v = torch.rand(3)
# res = batched_mv(M.view(1, 3, 3), v.view(1, 3))
# assert torch.allclose(M @ v, res)
