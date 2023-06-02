import logging
from math import isclose
from typing import TypeAlias, Optional, Callable

from tqdm import trange

import torch
import torch.nn as nn

from flows_on_spheres.abc import Density, Flow, Hamiltonian
from flows_on_spheres.prior import uniform_prior
from flows_on_spheres.linalg import (
    orthogonal_projection,
    dot,
    dot_keepdim,
    norm_keepdim,
)

Tensor: TypeAlias = torch.Tensor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _get_n_steps(traj_length: float, step_size: float) -> float:
    n_steps = max(1, round(traj_length / abs(step_size)))
    if not isclose(n_steps * abs(step_size), traj_length):
        msg = f"Warning: trajectory length will be {n_steps * step_size} which is different from the requested length {traj_length}"
        log.warn(msg)
    return n_steps


def euler_integrator(
    x0: Tensor,
    flow_action:,
    step_size: float,
    traj_length: float,
    on_step_func: Optional[Callable[[Tensor, Tensor, float], None]] = None,
) -> tuple[Tensor, float]:
    if not x0.dtype == torch.float64:
        log.warn("Not using 64 bit precision. This may be a problem")
    
    n_steps = _get_n_steps(traj_length, step_size)

    x = x0.clone()
    t = 0
    ε = step_size
    
    for _ in range(n_steps):
        if on_step_func is not None:
            on_step_func(x, t)

        mod_v = norm_keepdim(v)
        cos_εv = (ε * mod_v).cos()
        sin_εv = (ε * mod_v).sin()

        p = cos_εv * p - (sin_εv * mod_p) * x
        x = cos_εv * x + (sin_εv / mod_v) * v

        x = x / norm_keepdim(x)

        F = hamiltonian.grad_wrt_x(x).negative()

        # x . F can be larger than 1e-5 for large forces!
        # assert torch.allclose(dot(F, x), torch.zeros(1), atol=1e-5)

        p += (ε / 2) * F

        t += ε

    return x, p, t


def leapfrog_integrator(
    x0: Tensor,
    p0: Tensor,
    hamiltonian: Hamiltonian,
    step_size: float,
    traj_length: float,
    on_step_func: Optional[Callable[[Tensor, Tensor, float], None]] = None,
) -> tuple[Tensor, Tensor, float]:
    assert p0.dtype == x0.dtype
    if not x0.dtype == torch.float64:
        log.warn("Not using 64 bit precision. This may be a problem")
    
    n_steps = _get_n_steps(traj_length, step_size)

    x = x0.clone()
    p = p0.clone()
    t = 0
    ε = step_size

    F = hamiltonian.grad_wrt_x(x).negative()

    # assert torch.allclose(dot(F, x), torch.zeros(1), atol=1e-5)

    for _ in range(n_steps):
        if on_step_func is not None:
            on_step_func(x, p, t)

        # NOTE: avoid in-place here in case p stored in on_step_func
        p = p + (ε / 2) * F

        v = hamiltonian.grad_wrt_p(p)

        mod_p = norm_keepdim(p)
        mod_v = norm_keepdim(v)
        cos_εv = (ε * mod_v).cos()
        sin_εv = (ε * mod_v).sin()

        p = cos_εv * p - (sin_εv * mod_p) * x
        x = cos_εv * x + (sin_εv / mod_v) * v

        x = x / norm_keepdim(x)

        F = hamiltonian.grad_wrt_x(x).negative()

        # x . F can be larger than 1e-5 for large forces!
        # assert torch.allclose(dot(F, x), torch.zeros(1), atol=1e-5)

        p += (ε / 2) * F

        t += ε

    return x, p, t


