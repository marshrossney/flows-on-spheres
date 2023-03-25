"""
Classes defining von Mises-Fisher and uniform distributions on the D-spheres.
"""
from dataclasses import dataclass, field
from math import exp, gamma, isclose, lgamma, log, pi as π, sqrt
from typing import Optional, TypeAlias

import torch
import torch.linalg as LA
from scipy.special import iv

Tensor: TypeAlias = torch.Tensor
IterableDataset: TypeAlias = torch.utils.data.IterableDataset

@dataclass
class VonMisesFisherDensity:
    κ: float
    μ: list[float]
    dim: int = field(init=False)
    log_norm: float = field(init=False)
    
    def __post_init__(self):
        if self.κ == 0:
            raise ValueError("κ must be nonzero. For a uniform density please use `CircularUniformDistribuion`")
        elif self.κ < 0:
            raise ValueError("κ must be positive.")
            
        mod_μ = sqrt(sum([pow(μi, 2) for μi in self.μ]))
        self.μ = [μi / mod_μ for μi in self.μ]
        
        D = len(self.μ) - 1
        v = (D - 1) / 2
        
        self.dim = D
        self.norm = pow(self.κ, -v) * pow(2 * π, v + 1) * iv(v, self.κ)
        self.log_norm = log(self.norm)
        
    def compute(self, x: Tensor) -> Tensor:
        μ = torch.tensor(self.μ, device=x.device)
        return x.mv(μ).mul(self.κ).exp().divide(self.norm)
    
    def compute_log(self, x: Tensor) -> Tensor:
        μ = torch.tensor(self.μ, device=x.device)
        return x.mv(μ).mul(self.κ).subtract(self.log_norm)
        
@dataclass
class VonMisesFisherMixtureDensity:
    densities: list[VonMisesFisherDensity]
    weights: Optional[list[float]] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = [1 / len(self.densities) for _ in self.densities]
        else:
            exp_weights = [exp(weight) for weight in self.weights]
            sum_exp_weights = sum(exp_weights)
            self.weights = [exp_weight / sum_exp_weights for exp_weight in exp_weights]
        
        assert len(self.weights) == len(self.densities)
        assert isclose(sum(self.weights), 1)
        
    def compute(self, x: Tensor) -> Tensor:
        total = torch.zeros(x.shape[0], device=x.device)
        for ρ, density in zip(self.weights, self.densities):
            total += (ρ * density.compute(x))
        return total
    
    def compute_log(self, x: Tensor) -> Tensor:
        return self.compute(x).log()

        
def uniform_prior(dim: int, batch_size: int) -> IterableDataset:
    """Returns an IterableDataset that generates uniform samples on the sphere."""
    if dim == 1:
        return CircularUniformPrior(batch_size)
    elif dim == 2:
        return SphericalUniformPrior(batch_size)
    elif dim > 2:
        return HyperSphericalUniformPrior(dim, batch_size)
    else:
        raise ValueError("dim must be an integer greater than or equal to 1")
    

class CircularUniformPrior(IterableDataset):
    """
    Uniform density on the unit circle.
    """
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def __iter__(self):
        return self
    
    def __next__(self):
        ϕ = torch.empty(self.batch_size).uniform_(0, 2 * π)
        x = torch.stack([ϕ.cos(), ϕ.sin()], dim=1)
        log_p = torch.full((self.batch_size,), fill_value=-log(2 * π))
        return x, log_p


class SphericalUniformPrior(IterableDataset):
    """
    Uniform density on the unit sphere.
    """
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        θ = (1 - torch.empty(self.batch_size).uniform_(0, 2)).acos()
        ϕ = torch.empty(self.batch_size).uniform_(0, 2 * π)

        sinθ, cosθ, sinϕ, cosϕ = θ.sin(), θ.cos(), ϕ.sin(), ϕ.cos()
        x = torch.stack([
            sinθ * cosϕ,
            sinθ * sinϕ,
            cosθ,
        ], dim=1
        )
        log_p = torch.full((self.batch_size,), fill_value=-log(4 * π))

        return x, log_p
    
def marsaglia(D: int, size: torch.Size = torch.Size([1]), device: Optional[torch.device] = None) -> Tensor:
    device = torch.device("cpu") if device is None else device
    
    x = torch.empty(*size, D, dtype=torch.float64, device=device).normal_()
    x.divide_(LA.vector_norm(x, dim=-1, keepdim=True))

    # Drop nans and infs
    isfinite = x.isfinite().flatten(start_dim=1).all(dim=1)
    if not isfinite.all():
        x = x[isfinite]

    return x.to(torch.float32)

class HyperSphericalUniformPrior(torch.utils.data.IterableDataset):
    def __init__(self, dim: int, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        
        assert isinstance(dim, int)
        assert dim > 2

        self.dim = dim
        self._log_surface_area = (
            log(2) + (dim / 2) * log(π) - lgamma(dim / 2)
        )

    def __iter__(self):
        return self

    def __next__(self):
        x = marsaglia(self.dim, self.shape)
        log_p = torch.full((self.batch_size,), fill_value=-self.log_surface_area)
        return x, log_p
