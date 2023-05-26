from math import isclose
from typing import TypeAlias, Optional, Callable

from tqdm import trange

import torch
import torch.linalg as LA

from flows_on_spheres.abc import Density, Flow, Hamiltonian
from flows_on_spheres.prior import uniform_prior
from flows_on_spheres.utils import (
    orthogonal_projection,
    batched_dot,
    batched_mv,
)
from flows_on_spheres.geometry import get_rotation_matrix

Tensor: TypeAlias = torch.Tensor


class HamiltonianGaussianMomenta(Hamiltonian):
    def __init__(self, target: Density):
        self.target = target

    @property
    def dim(self) -> int:
        return self.target.dim

    def action(self, x: Tensor) -> Tensor:
        return self.target.log_density(x).negative()

    def hamiltonian(self, x: Tensor, p: Tensor) -> Tensor:
        return (1 / 2) * batched_dot(p, p) + self.action(x)

    def grad_wrt_p(self, p: Tensor) -> Tensor:
        return p.clone()

    def grad_wrt_x(self, x: Tensor) -> Tensor:
        Px = orthogonal_projection(x)
        grad_action = self.target.grad_log_density(x).negative()
        return batched_mv(Px, grad_action)

    def sample_momentum(self, x: Tensor) -> None:
        Px = orthogonal_projection(x)
        p = torch.empty_like(x).normal_()
        return batched_mv(Px, p)


class HamiltonianCauchyMomenta(Hamiltonian):
    max_abs_momentum: float = 1e6

    def __init__(self, target: Density, gamma: float):
        self.target = target
        self.gamma = gamma
        self.gamma_sq = gamma**2

    @property
    def dim(self) -> int:
        return self.target.dim

    def action(self, x: Tensor) -> Tensor:
        return self.target.log_density(x).negative()

    def hamiltonian(self, x: Tensor, p: Tensor) -> Tensor:
        return ((self.dim + 1) / 2) * (
            1 + batched_dot(p, p) / self.gamma_sq
        ).log() + self.action(x)

    def grad_wrt_p(self, p: Tensor) -> Tensor:
        return (
            (self.dim + 1)
            / (self.gamma_sq + batched_dot(p, p, keepdim=True))
            * p
        )

    def grad_wrt_x(self, x: Tensor) -> Tensor:
        Px = orthogonal_projection(x)
        grad_action = self.target.grad_log_density(x).negative()
        return batched_mv(Px, grad_action)

    def sample_momentum(self, x: Tensor) -> None:
        Px = orthogonal_projection(x)
        u = torch.empty_like(x).normal_(0, self.gamma)
        u = batched_mv(Px, u)

        v = torch.empty(*x.shape[:-1], 1, device=x.device).normal_(0, 1)

        p = (u / v).nan_to_num(
            posinf=self.max_abs_momentum, neginf=-self.max_abs_momentum
        )

        return p


def leapfrog_integrator(
    x0: Tensor,
    p0: Tensor,
    hamiltonian: Hamiltonian,
    step_size: float,
    traj_length: float,
    on_step_func: Optional[Callable[[Tensor, Tensor, float], None]] = None,
) -> tuple[Tensor, Tensor, float]:
    n_steps = max(1, round(traj_length / abs(step_size)))
    if not isclose(n_steps * step_size, traj_length):
        # TODO log warn
        pass

    x = x0.clone()
    p = p0.clone()
    t = 0
    ε = step_size

    F = hamiltonian.grad_wrt_x(x).negative()

    for _ in range(n_steps):
        if on_step_func is not None:
            on_step_func(x, p, t)

        # NOTE: avoid in-place here in case p stored in on_step_func
        p = p + (ε / 2) * F

        v = hamiltonian.grad_wrt_p(p)

        mod_p = LA.vector_norm(p, dim=-1, keepdim=True)
        mod_v = LA.vector_norm(v, dim=-1, keepdim=True)
        cos_εv = (ε * mod_v).cos()
        sin_εv = (ε * mod_v).sin()

        p = cos_εv * p - (sin_εv * mod_p) * x
        x = cos_εv * x + (sin_εv / mod_v) * v

        mod_x = LA.vector_norm(x, dim=-1, keepdim=True)
        x /= mod_x

        F = hamiltonian.grad_wrt_x(x).negative()

        p += (ε / 2) * F

        t += ε

    return x, p, t


class HMCSampler:
    def __init__(
        self,
        hamiltonian: Hamiltonian,
        step_size: float,
        traj_length: float = 1.0,
        n_replicas: int = 1,
    ) -> None:
        self.hamiltonian = hamiltonian

        self._traj_length = traj_length
        self._n_steps = round(traj_length / step_size)
        self._step_size = traj_length / self._n_steps
        self._n_replicas = n_replicas

        self._n_traj = 0
        self._n_accepted = torch.zeros(n_replicas)

        self._delta_H_history = []

        self._current_state, _ = next(
            uniform_prior(hamiltonian.dim, n_replicas)
        )

    @property
    def step_size(self) -> float:
        return self._step_size

    @property
    def n_traj(self) -> int:
        return self._n_traj

    @property
    def acceptance_rate(self) -> Tensor:
        return self._n_accepted / self._n_traj

    @property
    def exp_delta_H(self) -> Tensor:
        return (
            torch.stack(self._delta_H_history, dim=0)
            .negative()
            .exp()
            .mean(dim=0)
        )

    @torch.no_grad()
    def sample(self, n: int) -> Tensor:
        x0 = self._current_state
        outputs = torch.empty(
            (n, self._n_replicas, self.hamiltonian.dim + 1)
        ).type_as(x0)

        for i in trange(n):
            x0 = x0.clone()
            p0 = self.hamiltonian.sample_momentum(x0)
            H0 = self.hamiltonian.hamiltonian(x0, p0)

            xT, pT, T = leapfrog_integrator(
                x0,
                p0,
                hamiltonian=self.hamiltonian,
                step_size=self._step_size,
                traj_length=self._traj_length,
            )

            HT = self.hamiltonian.hamiltonian(xT, pT)

            # NOTE: torch.exp(large number) yields 'inf' and
            # 'inf' > x for float x yields True
            accepted = (H0 - HT).exp() > torch.rand_like(H0)

            x0[accepted] = xT[accepted]

            self._n_traj += 1
            self._n_accepted += accepted.int()
            self._delta_H_history.append(HT - H0)

            outputs[i] = x0

        self._current_state = x0

        return outputs


class FlowedDensity(Density):
    def __init__(self, flow: Flow, target: Density):
        assert flow.dim == target.dim or flow.dim is None

        self.flow = flow
        self.target = target

    @property
    def dim(self) -> int:
        return self.target.dim

    @torch.no_grad()
    def density(self, x: Tensor) -> Tensor:
        fx, ldj = self.flow(x)
        p_fx = self.target.density(fx)
        return p_fx * ldj.exp()

    @torch.no_grad()
    def log_density(self, x: Tensor) -> Tensor:
        fx, ldj = self.flow(x)
        return self.target.log_density(fx) + ldj

    def grad_density(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @torch.enable_grad()
    def grad_log_density(self, x: Tensor) -> Tensor:
        x = x.clone().detach()  # creates a new leaf tensor
        x.requires_grad_(True)
        fx, ldj = self.flow(x)
        log_density_fx = self.target.log_density(fx) + ldj
        log_density_fx.backward(gradient=torch.ones_like(log_density_fx))
        return x.grad

def add_hmc_hooks(flow: Flow, target: Density):
    def forward_pre_hook(module, inputs: tuple[Tensor]) -> None:
        (x,) = inputs
        x.requires_grad_(True)
        x.grad = None

    def forward_post_hook(
        module, inputs: Tensor, outputs: tuple[Tensor, Tensor]
    ) -> None:
        (x,) = inputs
        fx, ldj = outputs
        log_density_fx = target.log_density(fx) + ldj
        log_density_fx.backward(gradient=torch.ones_like(log_density_fx))

    pre_hook_handle = flow.register_forward_pre_hook(forward_pre_hook)
    post_hook_handle = flow.register_forward_hook(forward_post_hook)

    return pre_hook_handle, post_hook_handle
