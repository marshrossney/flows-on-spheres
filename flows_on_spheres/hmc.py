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


class HamiltonianGaussianMomenta(Hamiltonian):
    def __init__(self, target: Density):
        self.target = target

    @property
    def dim(self) -> int:
        return self.target.dim

    def action(self, x: Tensor) -> Tensor:
        return self.target.log_density(x).negative()

    def hamiltonian(self, x: Tensor, p: Tensor) -> Tensor:
        return dot(p, p) / 2 + self.action(x)

    def grad_wrt_p(self, p: Tensor) -> Tensor:
        return p.clone()

    def grad_wrt_x(self, x: Tensor) -> Tensor:
        return self.target.grad_log_density(x).negative()

    def sample_momentum(self, x: Tensor) -> None:
        p = torch.empty_like(x).normal_()
        return orthogonal_projection(p, x)


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
            1 + dot(p, p) / self.gamma_sq
        ).log() + self.action(x)

    def grad_wrt_p(self, p: Tensor) -> Tensor:
        return (self.dim + 1) / (self.gamma_sq + dot_keepdim(p, p)) * p

    def grad_wrt_x(self, x: Tensor) -> Tensor:
        return self.target.grad_log_density(x).negative()

    def sample_momentum(self, x: Tensor) -> None:
        u = torch.empty_like(x).normal_(0, self.gamma)
        u = orthogonal_projection(u, x)

        v = x.new_empty((*x.shape[:-1], 1)).normal_(0, 1)

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
    assert p0.dtype == x0.dtype
    if not x0.dtype == torch.float64:
        log.warn("Not using 64 bit precision. This may be a problem")

    n_steps = max(1, round(traj_length / abs(step_size)))
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

        self._current_state, _ = uniform_prior(
            hamiltonian.dim, device="cpu", dtype=torch.double
        )(n_replicas)

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
        return torch.stack(self._delta_H_history, dim=0).negative().exp()

    @torch.no_grad()
    def sample(self, n_traj: int, n_therm: int) -> Tensor:
        x0 = self._current_state
        outputs = torch.empty(
            (n_traj, self._n_replicas, self.hamiltonian.dim + 1)
        ).type_as(x0)

        sampling = False
        with trange(n_therm + n_traj, desc="Thermalising") as pbar:
            for step in pbar:
                if step == n_therm:
                    sampling = True
                    pbar.set_description_str("Sampling")

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

                if sampling:
                    self._n_traj += 1
                    self._n_accepted += accepted.int()
                    self._delta_H_history.append(HT - H0)

                    outputs[step - n_therm] = x0

        self._current_state = x0

        return outputs


class FlowedDensity(nn.Module, Density):
    def __init__(self, flow: Flow, target: Density):
        super().__init__()
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
        x.requires_grad_(True)
        fx, ldj = self.flow(x)

        # This is a scalar field on the latent manifold - function of x
        log_density_flowed = self.target.log_density(fx) + ldj

        (gradient,) = torch.autograd.grad(
            outputs=log_density_flowed,
            inputs=x,
            grad_outputs=torch.ones_like(log_density_flowed),
        )

        return orthogonal_projection(gradient, x)


# useless
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
