from math import sin, cos, exp
from random import random
from typing import TypeAlias, Optional

import torch
import torch.linalg as LA

from vonmises.distributions import Density, uniform_prior
from vonmises.models import BaseFlow

Tensor: TypeAlias = torch.Tensor

MAX_STEPS = 1e6


@torch.no_grad()
def hmc(
    target: Density,
    n_traj: int,
    step_size: float,
    traj_length: float = 1.0,
) -> Tensor:
    ε, T = step_size, traj_length
    D = target.dim
    assert D >= 2

    n_accepted = 0
    sample = torch.empty(n_traj, D + 1)

    x0, _ = next(uniform_prior(D, 1))
    x0.squeeze_()

    for i in range(n_traj):
        x = x0.clone()

        F = target.grad_log_density(x.unsqueeze(0))
        P = torch.eye(D + 1) - torch.outer(x, x)

        k = torch.empty(D + 1).normal_()
        k = P @ k

        # TODO: could replace log_density with action - normalisation not required
        H0 = (1 / 2) * k.dot(k) - target.log_density(x.unsqueeze(0))

        # Begin leapfrog

        k += (1 / 2) * ε * (P @ F)

        t = 0
        while t < T:
            t += ε

            mod_k = LA.vector_norm(k)
            cos_εk = cos(ε * mod_k)
            sin_εk = sin(ε * mod_k)
            x_tmp = cos_εk * x + (1 / mod_k) * sin_εk * k
            k = -mod_k * sin_εk * x + cos_εk * k
            x = x_tmp

            x /= LA.vector_norm(x)

            F = target.grad_log_density(x.unsqueeze(0))
            P = torch.eye(D + 1) - torch.outer(x, x)
            k += ε * (P @ F)

        k -= (1 / 2) * ε * (P @ F)

        # End leapfrog

        HT = (1 / 2) * k.dot(k) - target.log_density(x.unsqueeze(0))

        if HT < H0 or exp(H0 - HT) > random():
            n_accepted += 1
            sample[i] = x
            x0 = x
        else:
            sample[i] = x0

    acceptance = n_accepted / n_traj
    print("acceptance: ", acceptance)  # TODO logging

    return sample


@torch.no_grad()
def flow_hmc(
    model: BaseFlow,
    n_traj: int,
    step_size: float,
    traj_length: float = 1.0,
    target: Optional[Density] = None,
) -> Tensor:
       
    ε, T = step_size, traj_length

    if target is not None:
        model.target = target
    else:
        target = model.target
    model.hmc_mode()
    
    D = model.target.dim
    assert D >= 2

    n_accepted = 0
    sample = torch.empty(n_traj, D + 1)

    z0, _ = next(uniform_prior(D, batch_size=1))
    with torch.enable_grad():
        x0, delta_log_vol = model(z0)
    F = z0.grad.squeeze()

    assert z0.shape == torch.Size([1, D + 1])
    assert x0.shape == torch.Size([1, D + 1])

    for i in range(n_traj):
        z = z0.clone()
        x = x0.clone()

        P = torch.eye(D + 1) - (z * z.T)
        k = torch.empty(D + 1).normal_()
        k = P @ k

        H0 = (1 / 2) * k.dot(k) - target.log_density(x) - delta_log_vol

        # Begin leapfrog

        k += (1 / 2) * ε * (P @ F)

        t = 0
        while t < T:
            t += ε

            mod_k = LA.vector_norm(k)
            cos_εk = cos(ε * mod_k)
            sin_εk = sin(ε * mod_k)
            z_tmp = cos_εk * z + (1 / mod_k) * sin_εk * k
            k = -mod_k * sin_εk * z.squeeze() + cos_εk * k
            z = z_tmp

            z = z / LA.vector_norm(z)

            with torch.enable_grad():
                x, delta_log_vol = model(z)
            F = z.grad.squeeze()
            P = torch.eye(D + 1) - (z * z.T)
            k += ε * (P @ F)

        k -= (1 / 2) * ε * (P @ F)

        # End leapfrog

        HT = (1 / 2) * k.dot(k) - target.log_density(x) - delta_log_vol

        if HT < H0 or exp(H0 - HT) > random():
            n_accepted += 1
            sample[i] = x
            x0 = x
        else:
            sample[i] = x0

    model.hmc_mode(False)

    acceptance = n_accepted / n_traj
    print("acceptance: ", acceptance)  # TODO logging

    return sample
