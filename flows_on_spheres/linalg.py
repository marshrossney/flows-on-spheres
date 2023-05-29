import torch
from torch import einsum
import torch.linalg as LA

Tensor = torch.Tensor


def dot(x: Tensor, y: Tensor) -> Tensor:
    return einsum("...i,...i->...", x, y)


def dot_keepdim(x: Tensor, y: Tensor) -> Tensor:
    return dot(x, y).unsqueeze(-1)


def outer(x: Tensor, y: Tensor) -> Tensor:
    return einsum("...i,...j->...ij", x, y)


def cross(x: Tensor, y: Tensor) -> Tensor:
    return LA.cross(x, y, dim=-1)


def mv(M: Tensor, v: Tensor) -> Tensor:
    return einsum("...ij,...j->...i", M, v)


def vm(v: Tensor, M: Tensor) -> Tensor:
    return einsum("...i,...ij->...j", v, M)


def norm(x: Tensor) -> Tensor:
    return LA.vector_norm(x, dim=-1)


def norm_keepdim(x: Tensor) -> Tensor:
    return LA.vector_norm(x, dim=-1, keepdim=True)


def projector(x: Tensor) -> Tensor:
    return torch.eye(x.shape[-1], dtype=x.dtype, device=x.device) - outer(
        x, x
    ) / dot(x, x).unflatten(-1, (-1, 1, 1))


def orthogonal_projection(v: Tensor, x: Tensor) -> Tensor:
    return mv(projector(x), v)
