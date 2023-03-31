from math import sin, cos, exp
from random import random
from typing import TypeAlias

from tqdm import trange

import torch
import torch.linalg as LA

from vonmises.distributions import Density, uniform_prior
from vonmises.flows import Flow
from vonmises.model import FlowBasedModel

Tensor: TypeAlias = torch.Tensor


def hmc(
    target: Density,
    sample_size: int,
    step_size: float,
    traj_length: float = 1.0,
) -> tuple[Tensor, float]:
    ε, T = step_size, traj_length
    D = target.dim
    assert D >= 2

    n_accepted = 0
    sample = torch.empty(sample_size, D + 1)

    x0, _ = next(uniform_prior(D, 1))

    for i in trange(sample_size):
        x = x0.clone()

        F = target.grad_log_density(x)
        P = torch.eye(D + 1) - (x * x.T)

        k = torch.empty(D + 1).normal_()
        k = P @ k

        # TODO: could replace log_density with action - normalisation not required
        H0 = (1 / 2) * k.dot(k) - target.log_density(x)

        # Begin leapfrog

        k += (1 / 2) * ε * (P @ F)

        t = 0
        while t < T:
            t += ε

            mod_k = LA.vector_norm(k)
            cos_εk = cos(ε * mod_k)
            sin_εk = sin(ε * mod_k)
            x_tmp = cos_εk * x + (1 / mod_k) * sin_εk * k
            k = -mod_k * sin_εk * x.squeeze() + cos_εk * k
            x = x_tmp

            x = x / LA.vector_norm(x)

            F = target.grad_log_density(x)
            P = torch.eye(D + 1) - (x * x.T)
            k += ε * (P @ F)

        k -= (1 / 2) * ε * (P @ F)

        # End leapfrog

        HT = (1 / 2) * k.dot(k) - target.log_density(x)

        if HT < H0 or exp(H0 - HT) > random():
            n_accepted += 1
            sample[i] = x
            x0 = x
        else:
            sample[i] = x0

    acceptance = n_accepted / sample_size

    return sample, acceptance


def add_fhmc_hooks(module: Flow | FlowBasedModel, target: Density) -> tuple:
    def forward_pre_hook(module, inputs: tuple[Tensor]) -> None:
        (x,) = inputs
        x.requires_grad_(True)
        x.grad = None

    def forward_post_hook(
        module, inputs: Tensor, outputs: tuple[Tensor, Tensor]
    ) -> None:
        (x_in,) = inputs
        x_out, delta_log_vol = outputs
        negative_effective_action = target.log_density(x_out) + delta_log_vol
        negative_effective_action.backward(
            gradient=torch.ones_like(negative_effective_action)
        )

    pre_hook_handle = module.register_forward_pre_hook(forward_pre_hook)
    post_hook_handle = module.register_forward_hook(forward_post_hook)

    return pre_hook_handle, post_hook_handle


@torch.no_grad()
def fhmc(
    flow: Flow,
    target: Density,
    sample_size: int,
    step_size: float,
    traj_length: float = 1.0,
) -> tuple[Tensor, float]:
    ε, T = step_size, traj_length

    D = target.dim
    assert D >= 2

    n_accepted = 0
    sample = torch.empty(sample_size, D + 1)

    hooks = add_fhmc_hooks(flow, target)

    z0, _ = next(uniform_prior(D, batch_size=1))

    with torch.enable_grad():
        x0, delta_log_vol = flow(z0)
    F = z0.grad.squeeze()

    assert z0.shape == torch.Size([1, D + 1])
    assert x0.shape == torch.Size([1, D + 1])

    # TODO annotate progress bar
    for i in trange(sample_size):
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
                x, delta_log_vol = flow(z)
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

    acceptance = n_accepted / sample_size

    [hook.remove() for hook in hooks]

    return sample, acceptance
