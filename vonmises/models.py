from itertools import chain
import math
from pathlib import Path
from typing import TypeAlias, Callable, Optional
from random import random

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import seaborn as sns
import torch
import torch.linalg as LA
import torch.nn.functional as F

from vonmises.distributions import Density, uniform_prior
from vonmises.utils import (
    apply_global_rotation,
    circle_vectors_to_angles,
    mod_2pi,
    metropolis_acceptance,
    effective_sample_size,
)

Tensor: TypeAlias = torch.Tensor
Module: TypeAlias = torch.nn.Module
Parameter: TypeAlias = torch.nn.Parameter

π = math.pi

DEFAULT_VAL_BATCH_SIZE = pow(2, 12)
DEFAULT_TEST_BATCH_SIZE = pow(2, 14)


def make_net(
    in_features: int,
    out_features: int,
    hidden_shape: list[int],
    activation: str = "Tanh",
):
    layers = [
        torch.nn.Linear(f_in, f_out)
        for f_in, f_out in zip(
            [in_features, *hidden_shape], [*hidden_shape, out_features]
        )
    ]
    activation = getattr(torch.nn, activation)
    activations = [activation() for _ in hidden_shape] + [torch.nn.Identity()]

    return torch.nn.Sequential(*list(chain(*zip(layers, activations))))


class BaseFlow(pl.LightningModule):
    def __init__(
        self,
        target: Density,
        n_layers: int,
        batch_size: int,
        val_batch_size: int = DEFAULT_VAL_BATCH_SIZE,
        test_batch_size: int = DEFAULT_TEST_BATCH_SIZE,
        init_lr: float = 1e-3,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.init_lr = init_lr

        self.target = target

    def training_step(self, inputs: tuple[Tensor, Tensor], *_) -> Tensor:
        x_in, log_prior_density = inputs
        x_out, delta_log_vol = self(x_in)
        log_target_density = self.target.log_density(x_out)
        log_model_density = log_prior_density - delta_log_vol
        weights = log_target_density - log_model_density

        kl_div = weights.mean().negative()
        self.log("loss", kl_div)

        return kl_div

    def validation_step(self, inputs: tuple[Tensor, Tensor], *_) -> None:
        x_in, log_prior_density = inputs
        x_out, delta_log_vol = self(x_in)
        log_model_density = log_prior_density - delta_log_vol
        log_target_density = self.target.log_density(x_out)
        weights = log_target_density - log_model_density

        self.log("acceptance", metropolis_acceptance(weights))
        self.log("ess", effective_sample_size(weights))
        self.log("kl_div", weights.mean().negative())

    def test_step(self, inputs: tuple[Tensor, Tensor], *_) -> None:
        x_in, log_prior_density = inputs
        x_out, delta_log_vol = self(x_in)
        log_model_density = log_prior_density - delta_log_vol
        log_target_density = self.target.log_density(x_out)
        weights = log_target_density - log_model_density

        self.log("test/acceptance", metropolis_acceptance(weights))
        self.log("test/ess", effective_sample_size(weights))
        self.log("test/kl_div", weights.mean().negative())

    @torch.no_grad()
    def sample(self, sample_size: int = 1) -> tuple[Tensor, Tensor, Tensor]:
        prior = uniform_prior(self.target.dim, sample_size)
        x_in, log_prior_density = next(iter(prior))
        x_out, delta_log_vol = self(x_in)
        log_model_density = log_prior_density - delta_log_vol
        log_target_density = self.target.log_density(x_out)
        return x_out, log_model_density, log_target_density

    def forward_with_forces(self, x_in: Tensor) -> Tensor:
        x_in.requires_grad_(True)
        x_in.grad = None
        with torch.enable_grad():
            x_out, delta_log_vol = self(x_in)
            log_p_vMF = self.target.compute_log(x_out)
            log_p_eff = log_p_vMF + delta_log_vol
            assert S.shape == torch.Size([batch_size])
            log_p_eff.backward(gradient=torch.ones_like(log_p_eff))
        force = x_in.grad
        x_in.requires_grad_(False)
        x_in.grad = None
        return force

    def train_dataloader(self):
        return uniform_prior(self.target.dim, self.batch_size)

    def val_dataloader(self):
        return uniform_prior(self.target.dim, self.val_batch_size)

    def test_dataloader(self):
        return uniform_prior(self.target.dim, self.test_batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.init_lr)


class CircularFlow(BaseFlow):
    def __init__(
        self,
        transformer: Callable,
        target: Density,
        n_layers: int,
        batch_size: int,
        val_batch_size: int = DEFAULT_VAL_BATCH_SIZE,
        test_batch_size: int = DEFAULT_TEST_BATCH_SIZE,
        init_lr: float = 1e-3,
    ):
        super().__init__(
            target=target,
            n_layers=n_layers,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            init_lr=init_lr,
        )
        self.save_hyperparameters()

        self.transformer = transformer
        self.transformer_parameters = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.rand_like(transformer.identity_params))
                for _ in range(n_layers)
            ]
        )
        self.rotations = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.empty(1).uniform_(0, 2 * π))
                for _ in range(n_layers)
            ]
        )

        if self.logger is not None and type(self.logger) is TensorBoardLogger:
            self.logger.log_hyperparams(self.hparams)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        xy = inputs
        ldj_full = torch.zeros(inputs.shape[0], device=inputs.device)

        for params, θ in zip(self.transformer_parameters, self.rotations):
            xy, ldj_this = self.transformer(xy, params.expand(*xy.shape[:-1], -1))
            xy = apply_global_rotation(xy, θ)
            ldj_full += ldj_this

        return xy, ldj_full

    @torch.no_grad()
    def visualise(self, polar: bool = False) -> "Figure":
        x_in, log_prior_density = next(self.val_dataloader())
        x_out, delta_log_vol = self(x_in)
        log_target_density = self.target.log_density(x_out)
        target_density = self.target.density(x_out)
        ϕ = circle_vectors_to_angles(x_out)

        fig, ax = plt.subplots(subplot_kw={"polar": True} if polar else {})
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel("density")
        ax.scatter(ϕ, (log_prior_density - delta_log_vol).exp(), s=0.4, label="model")
        ax.scatter(ϕ, target_density, s=0.4, label="target")
        ax.legend()
        return fig


# class RecursiveFlowS2(pl.LightningModule):
#     def __init__(
#         self,
#         κ: float,
#         μ: Tensor,
#         z_transformer: Callable,
#         xy_transformer: Callable,
#         n_layers: int,
#         batch_size: int,
#         val_batch_size: int = DEFAULT_VAL_BATCH_SIZE,
#         test_batch_size: int = DEFAULT_TEST_BATCH_SIZE,
#         net_hidden_shape: Optional[list[int]] = None,
#         net_activation: str = "Tanh",
#         init_lr: float = 1e-3,
#         softmax_beta: float = 1e8,
#     ):
#         super().__init__()
#         self.save_hyperparameters()

#         self.κ = κ
#         μ = μ if isinstance(μ, torch.Tensor) else torch.tensor(μ, dtype=torch.float32)
#         μ.div_(LA.vector_norm(μ))
#         self.register_buffer("μ", μ)
#         self.log_norm = math.log(κ / (4 * π * math.sinh(κ)))

#         self.n_layers = n_layers
#         self.batch_size = batch_size
#         self.val_batch_size = val_batch_size
#         self.test_batch_size = test_batch_size
#         self.init_lr = init_lr
#         self.softmax_beta = softmax_beta

#         assert isinstance(z_transformer, RQSplineTransform) or isinstance(
#             z_transformer, BSplineTransform
#         )
#         assert isinstance(
#             xy_transformer, RQSplineTransformCircularDomain
#         ) or isinstance(xy_transformer, MobiusMixtureTransform)
#         self.z_transformer = z_transformer
#         self.xy_transformer = xy_transformer
#         self.polar_mode = isinstance(xy_transformer, RQSplineTransformCircularDomain)

#         self.z_params = torch.nn.ParameterList(
#             [
#                 torch.nn.Parameter(torch.tensor(z_transformer.identity_params))
#                 for _ in range(n_layers)
#             ]
#         )
#         if net_hidden_shape is None:
#             self.xy_params = torch.nn.ParameterList(
#                 [
#                     torch.nn.Parameter(torch.tensor(xy_transformer.identity_params))
#                     for _ in range(n_layers)
#                 ]
#             )
#         else:
#             self.xy_params = torch.nn.ModuleList(
#                 [
#                     make_net(1, xy_transformer.n_params, net_hidden_shape, net_activation)
#                     for _ in range(n_layers)
#                 ]
#             )

#         # Random global rotations applied after each layer (potentially trainable)
#         self.rotations = torch.nn.ParameterList(
#             [
#                 torch.nn.Parameter(torch.empty(1).uniform_(0, 2 * π))
#                 for _ in range(n_layers)
#             ]
#         )

#         if self.logger is not None and type(self.logger) is TensorBoardLogger:
#             self.logger.log_hyperparams(self.hparams)

#     def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
#         xy, z = inputs.split([2, 1], dim=-1)
#         ldj = torch.zeros(inputs.shape[0], device=inputs.device)

#         # Map sphere to cylinder
#         ρ = F.softplus(1 - z**2, beta=self.softmax_beta).sqrt()
#         xy = xy / ρ

#         if self.polar_mode:
#             x, y = xy.split(1, dim=-1)
#             xy_or_ϕ = mod_2pi(torch.atan2(y, x))  # [0, 2π)
#         else:
#             xy_or_ϕ = xy

#         for layer_idx, (z_params, xy_params, θ) in enumerate(
#             zip(self.z_params, self.xy_params, self.rotations)
#         ):
#             z, ldj_z = self.z_transformer(z, z_params.expand(*z.shape, -1))

#             xy_params = (
#                 xy_params.expand(*xy_or_ϕ.shape[:-1], -1)
#                 if isinstance(xy_params, Parameter)
#                 else xy_params(z)
#             )
#             xy_or_ϕ, ldj_xy = self.xy_transformer(
#                 xy_or_ϕ, xy_params.view(*xy_or_ϕ.shape, -1)
#             )

#             xy_or_ϕ = self._apply_global_rotation(xy_or_ϕ, θ)

#             ldj += ldj_z + ldj_xy

#         if self.polar_mode:
#             xy = torch.cat([torch.cos(xy_or_ϕ), torch.sin(xy_or_ϕ)], dim=-1)
#         else:
#             xy = xy_or_ϕ

#         ρ = F.softplus(1 - z**2, beta=self.softmax_beta).sqrt()
#         xy = ρ * xy

#         outputs = torch.cat([xy, z], dim=-1)
#         outputs.squeeze_(dim=1)

#         return outputs, ldj

#     def training_step(self, batch, *_):
#         z, logq = batch
#         x, ldj = self(z)
#         logp = self.log_norm + self.κ * torch.mv(x, self.μ)

#         kl_div = torch.mean(logq - ldj - logp)
#         self.log("loss", kl_div)

#         return kl_div

#     def validation_step(self, batch, *_):
#         z, logq = batch
#         x, ldj = self(z)
#         logp = self.log_norm + self.κ * torch.mv(x, self.μ)

#         weights = logp - (logq - ldj)
#         self.log("acceptance", metropolis_acceptance(weights))
#         self.log("ess", effective_sample_size(weights))

#         # Only TensorBoardLogger supported
#         if self.logger is None or type(self.logger) is not TensorBoardLogger:
#             return

#         x = x.to("cpu").detach()

#         fig = scatter(x)
#         self.logger.experiment.add_figure(
#             "3d_projection", fig, self.trainer.global_step
#         )

#         grid = pairplot(x)
#         self.logger.experiment.add_figure(
#             "pairplot", grid.figure, self.trainer.global_step
#         )

#     def test_step(self, batch, *_):
#         z, logq = batch
#         x, ldj = self(z)
#         logp = self.log_norm + self.κ * torch.mv(x, self.μ)

#         weights = logp - (logq - ldj)

#         self.log("hp/acceptance", metropolis_acceptance(weights))
#         self.log("hp/ess", effective_sample_size(weights))
#         self.log("hp/kl_div", -torch.mean(weights))

#     def train_dataloader(self):
#         return SphericalUniformPrior3D(self.batch_size)

#     def val_dataloader(self):
#         return SphericalUniformPrior3D(self.val_batch_size)

#     def test_dataloader(self):
#         return SphericalUniformPrior3D(self.test_batch_size)

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.init_lr)


# class DummyNormalizingFlow(pl.LightningModule):
#     def __init__(self):
#         super().__init__()

#     def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
#         return inputs, torch.zeros(inputs.shape[0], device=inputs.device)
