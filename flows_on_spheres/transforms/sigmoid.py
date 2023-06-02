from functools import partial
from typing import Any, Callable, Optional

import torch

from flows_on_spheres.nn import TransformModule
from flows_on_spheres.linalg import dot, norm
from flows_on_spheres.utils import mod_2pi

Tensor = torch.Tensor

# Credit to https://arxiv.org/pdf/2110.00351.pdf
# and https://github.com/noegroup/bgflow for this transformation


def exponential_ramp(
    x: Tensor, log_scale: Tensor, power: int, eps: float = 1e-9
) -> Tensor:
    assert isinstance(power, int) and power > 0
    α, β = log_scale.exp(), power
    # TODO:
    # Why use where and not clamp?
    # Why does all this avoid NaN?
    # Why divide by exp(-α) rather than use single exponential?
    z = torch.where(x > eps, x, torch.full_like(x, eps))
    return torch.where(
        x > eps,
        torch.exp(-α * z.pow(-β)) / torch.exp(-α),
        torch.zeros_like(x),
    )


def monomial_ramp(x: Tensor, order: int) -> Tensor:
    assert isinstance(order, int) and order > 0
    return x.pow(order)


def generalised_sigmoid(
    ramp: Callable[[Tensor, Any, ...], Tensor]
) -> Callable[[Tensor, Any, ...], Tensor]:
    def _sigmoid(x: Tensor, **ramp_kwargs):
        numer = ramp(x, **ramp_kwargs)
        denom = numer + ramp(1 - x, **ramp_kwargs)
        return numer / denom

    return _sigmoid


class _BumpTransform:
    def __init__(
        self,
        affine_log_scale: Tensor,
        affine_shift: Tensor,
        linear_weight: Tensor,
        ramp_log_scale: Tensor,
        ramp_power: int = 1,
    ):
        assert (affine_shift > 0).all()
        assert (affine_shift < 1).all()
        assert (linear_weight >= 0).all()
        assert (linear_weight <= 1).all()
        self.affine_log_scale = affine_log_scale
        self.affine_shift = affine_shift
        self.linear_weight = linear_weight

        self.ramp = partial(
            exponential_ramp, log_scale=ramp_log_scale, power=ramp_power
        )
        self.ramp_log_scale = ramp_log_scale
        self.ramp_power = ramp_power

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = (x + 1) / 2

        assert (x >= 0).all()
        assert (x <= 1).all()

        (
            loga,
            b,
            c,
        ) = (
            self.affine_log_scale,
            self.affine_shift,
            self.linear_weight,
        )
        a = loga.exp()

        logα, β = self.ramp_log_scale, self.ramp_power
        α = logα.exp()

        x10 = torch.stack([x, torch.zeros_like(x), torch.ones_like(x)])

        h_x10 = a * (x10 - b) + (1 / 2)  # affine transform of x

        ε = 1e-9

        ρ_x10 = self.ramp(h_x10)
        ρ1m_x10 = self.ramp(1 - h_x10)

        # The numerically stable version is much better!
        # ρ_x10 = torch.exp(-1 / (α * h_x10.pow(β) + ε)) / torch.exp(-1 / α)
        # ρ1m_x10 = torch.exp(-1 / (α * (1 - h_x10).pow(β) + ε)) / torch.exp(-1 / α)

        σ_x10 = ρ_x10 / (ρ_x10 + ρ1m_x10)

        σx, σ0, σ1 = σ_x10

        y = c * (σx - σ0) / (σ1 - σ0) + (1 - c) * x

        dσdx = (
            a
            * σx
            * (1 - σx)
            * (β / α)
            * ((x.pow(β + 1) + (1 - x).pow(β + 1)) / (x * (1 - x)).pow(β + 1))
        )

        dydx = c * dσdx / (σ1 - σ0) + (1 - c)

        ldj = dydx.log().squeeze(1)

        y = y * 2 - 1

        return y, ldj


class _BumpMixtureTransform:
    pass


class BumpModule(TransformModule):
    def __init__(
        self,
        *,
        n_mixture: int = 1,
        weighted: bool = False,
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: Optional[str] = None,
        epsilon: float = 1e-2,
        ramp_power: int = 2,
    ):
        assert not (n_mixture == 1 and weighted)
        super().__init__(
            n_params=(4 + int(weighted)) * n_mixture,
            net_hidden_shape=net_hidden_shape,
            net_activation=net_activation,
        )
        self.n_mixture = n_mixture
        self.weighted = weighted
        self.epsilon = epsilon

        self.ramp_power = ramp_power

    def forward(
        self, k: Tensor | None = None
    ) -> _BumpTransform | _BumpMixtureTransform:
        params = self.params(k)

        if self.n_mixture == 1:
            loga, b, c, logα = params.split(1, dim=1)
            c = torch.sigmoid(c)
            return _BumpTransform(
                affine_log_scale=logα,
                affine_shift=torch.sigmoid(b),
                linear_weight=torch.sigmoid(c),
                ramp_log_scale=logα,
                ramp_power=self.ramp_power,
            )
