"""
Classes defining target densities on the D-spheres.
"""
from math import exp, isclose, log, pi as π, sqrt
from typing import Optional, TypeAlias

import torch
from scipy.special import iv

from flows_on_spheres.abc import Density
from flows_on_spheres.linalg import orthogonal_projection

Tensor: TypeAlias = torch.Tensor


class _MixtureDensity(Density):
    def __init__(
        self,
        components: list[Density],
        weights: Optional[list[float]] = None,
    ):
        self.components = components
        if weights is None:
            self.weights = [1 / len(components) for _ in components]
        else:
            exp_weights = [exp(weight) for weight in weights]
            sum_exp_weights = sum(exp_weights)
            self.weights = [
                exp_weight / sum_exp_weights for exp_weight in exp_weights
            ]

        assert len(self.weights) == len(self.components)
        assert isclose(sum(self.weights), 1)
        assert len(set([component.dim for component in self.components])) == 1

        self._dim = self.components[0].dim

    @property
    def dim(self) -> int:
        return self._dim

    def density(self, x: Tensor) -> Tensor:
        total = torch.zeros(x.shape[0], device=x.device)
        for ρ, f in zip(self.weights, self.components):
            total += ρ * f.density(x)
        return total

    def log_density(self, x: Tensor) -> Tensor:
        return self.density(x).log()

    def grad_density(self, x: Tensor) -> Tensor:
        total = torch.zeros_like(x)
        for ρ, f in zip(self.weights, self.components):
            total += ρ * f.grad_density(x)
        return total

    def grad_log_density(self, x: Tensor) -> Tensor:
        return self.grad_density(x).divide(self.density(x).unsqueeze(1))


class VonMisesFisherDensity(Density):
    def __init__(self, κ: float, μ: list[float]):
        if κ == 0:
            raise ValueError(
                "κ must be nonzero. For a uniform density please use `CircularUniformDistribuion`"
            )
        elif κ < 0:
            raise ValueError("κ must be positive.")

        mod_μ = sqrt(sum([pow(μi, 2) for μi in μ]))
        μ = [μi / mod_μ for μi in μ]

        D = len(μ) - 1
        v = (D - 1) / 2

        self.κ = κ
        self.μ = μ
        self._dim = D
        self._norm = pow(self.κ, -v) * pow(2 * π, v + 1) * iv(v, self.κ)
        self._log_norm = log(self._norm)

    @property
    def dim(self) -> int:
        return self._dim

    def density(self, x: Tensor) -> Tensor:
        μ = torch.tensor(self.μ, device=x.device, dtype=x.dtype)
        return x.mv(μ).mul(self.κ).exp().divide(self._norm)

    def log_density(self, x: Tensor) -> Tensor:
        μ = torch.tensor(self.μ, device=x.device, dtype=x.dtype)
        return x.mv(μ).mul(self.κ).subtract(self._log_norm)

    def grad_density(self, x: Tensor) -> Tensor:
        κμ = torch.tensor(self.μ, device=x.device, dtype=x.dtype).mul(self.κ)
        gradient = torch.outer(self.density(x), κμ)
        return orthogonal_projection(gradient, x)

    def grad_log_density(self, x: Tensor) -> Tensor:
        κμ = torch.tensor(self.μ, device=x.device, dtype=x.dtype).mul(self.κ)
        return orthogonal_projection(κμ, x)


class VonMisesFisherMixtureDensity(_MixtureDensity):
    def __init__(
        self,
        κ: list[int],
        μ: list[list[float]],
        weights: Optional[list[float]] = None,
    ):
        components = [VonMisesFisherDensity(κi, μi) for κi, μi in zip(κ, μ)]
        super().__init__(components, weights)
