from typing import Optional

import torch

from flows_on_spheres.nn import TransformModule
from flows_on_spheres.linalg import dot, norm
from flows_on_spheres.utils import mod_2pi

Tensor = torch.Tensor


class _MobiusTransform:
    def __init__(self, omega: Tensor):
        assert omega.dim() == 2
        assert omega.shape[1] == 2
        self.omega = omega

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        assert x.dim() == 2
        assert x.shape[1] == 2
        ω = self.omega
        dydx = ((1 - dot(ω, ω)) / dot(x - ω, x - ω)).unsqueeze(-1)
        y = dydx * (x - ω) - ω
        ldj = dydx.log().sum(dim=1)
        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        assert y.dim() == 2
        assert y.shape[1] == 2
        ω = self.omega
        dxdy = ((1 - dot(ω, ω)) / dot(y + ω, y + ω)).unsqueeze(-1)
        x = dxdy * (y + ω) + ω
        ldj = dxdy.log().sum(dim=1)
        return x, ldj


class _MobiusMixtureTransform:
    def __init__(
        self,
        omega: Tensor,
        weights: Tensor | None = None,
    ):
        ω, ρ = omega, weights
        assert ω.dim() == 3
        assert (norm(ω) < 1).all()

        n_batch, n_mixture, n_coords = ω.shape
        assert n_coords == 2  # only valid for circle

        if ρ is not None:
            assert ρ.shape == (n_batch, n_mixture, 1), f"{ρ.shape}"
            assert torch.allclose(ρ.sum(dim=1), ρ.new_ones(1))
        else:
            ρ = 1 / n_mixture

        self.omega = ω
        self.weights = ρ

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        assert x.dim() == 2
        assert x.shape[1] == 2
        ω, ρ = self.omega, self.weights

        x0 = torch.tensor([1, 0]).type_as(x).expand_as(x)
        x_x0 = torch.stack([x, x0], dim=0).unsqueeze(-2)  # add mixture dim

        dydx = ((1 - dot(ω, ω)) / dot(x_x0 - ω, x_x0 - ω)).unsqueeze(-1)
        y_y0 = dydx * (x_x0 - ω) - ω

        # Now rotate s.t (1, 0) -> (1, 0)
        ϕ, ϕ0 = torch.atan2(*reversed(y_y0.split(1, dim=-1)))
        ϕ = mod_2pi(ϕ - ϕ0)

        Σρϕ = (ρ * ϕ).sum(dim=1)
        y = torch.cat([torch.cos(Σρϕ), torch.sin(Σρϕ)], dim=-1)

        dydx, _ = dydx
        dydx = (ρ * dydx).sum(dim=1)
        ldj = dydx.log().sum(dim=1)

        return y, ldj

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class MobiusModule(TransformModule):
    def __init__(
        self,
        *,
        n_mixture: int,
        weighted: bool = False,
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: Optional[str] = None,
        epsilon: float = 1e-2,
    ):
        assert not (n_mixture == 1 and weighted)
        super().__init__(
            n_params=(2 + int(weighted)) * n_mixture,
            net_hidden_shape=net_hidden_shape,
            net_activation=net_activation,
        )
        self.n_mixture = n_mixture
        self.weighted = weighted
        self.epsilon = epsilon

    def _constrain_omega(self, omega: Tensor) -> Tensor:
        omega = torch.tanh(omega) * (1 - self.epsilon)
        omega1, omega2 = omega.split(1, dim=-1)
        omega = torch.cat(
            [omega1, omega2 * (1 - omega1.pow(2)).sqrt()], dim=-1
        )
        return omega

    def forward(
        self, k: Tensor | None = None
    ) -> _MobiusTransform | _MobiusMixtureTransform:
        params = self.params(k)

        if self.n_mixture == 1:
            omega = self._constrain_omega(params)
            return _MobiusTransform(omega)

        if not self.weighted:
            params = params.unflatten(1, (self.n_mixture, 2))
            omega = self._constrain_omega(params)
            return _MobiusMixtureTransform(omega)

        else:
            params = params.unflatten(1, (self.n_mixture, 3))
            omega, weights = params.split([2, 1], dim=2)
            omega = self._constrain_omega(omega)
            weights = torch.softmax(weights, dim=1)
            return _MobiusMixtureTransform(omega, weights)
