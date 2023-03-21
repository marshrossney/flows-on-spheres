import math
from typing import TypeAlias

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import seaborn as sns
import torch
import torch.linalg as LA

from distributions import SphericalUniformPrior3D
from transforms import (
    MobiusMixtureTransform,
    RQSplineTransform,
    RQSplineTransformCircularDomain,
    BSplineTransform,
)
from utils import (
    metropolis_acceptance,
    effective_sample_size,
    spherical_mesh,
    simple_fnn_conditioner,
    mod_2pi,
)
from visualisations import scatter, pairplot

Tensor: TypeAlias = torch.Tensor

π = math.pi


class NormalizingFlowBase(pl.LightningModule):
    def __init__(
        self,
        κ: float,
        μ: Tensor,
        n_layers: int,
        batch_size: int,
        val_batch_size: int,
        init_lr: float = 0.001,
    ):
        super().__init__()

        self.κ = κ
        μ = μ if isinstance(μ, torch.Tensor) else torch.tensor(μ, dtype=torch.float32)
        μ.div_(LA.vector_norm(μ))
        self.register_buffer("μ", μ)

        self.n_layers = n_layers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.init_lr = init_lr

        self.log_norm = math.log(κ / (4 * π * math.sinh(κ)))

        self.mesh = spherical_mesh(25)

        # Random global rotations applied after each layer (potentially trainable)
        self.rotations = torch.nn.Parameter(torch.empty(n_layers).uniform_(0, 2 * π))

        self.save_hyperparameters()
        if self.logger is not None and type(self.logger) is TensorBoardLogger:
            self.logger.log_hyperparams(self.hparams)

    def apply_global_rotation(self, xy_or_ϕ: Tensor, layer: int) -> Tensor:
        *data_shape, coord_dims = xy_or_ϕ.shape

        θ = self.rotations[layer]

        if coord_dims == 1:  # working with angles
            ϕ = xy_or_ϕ
            return mod_2pi(ϕ + θ)
        elif coord_dims == 2:  # working with euclidean coords
            xy = xy_or_ϕ
            cosθ, sinθ = θ.cos(), θ.sin()
            # counter-clockwise rotation
            R = torch.tensor(
                [[cosθ, -sinθ], [sinθ, cosθ]],
                device=xy.device,
            ).view(*[1 for _ in data_shape], 2, 2)
            xy.unsqueeze_(dim=-2)
            return (R * xy).sum(dim=-1)
        else:
            raise ValueError("Wrong number of dimensions")

    def training_step(self, batch, *_):
        z, logq = batch
        x, ldj = self(z)
        logp = self.log_norm + self.κ * torch.mv(x, self.μ)

        kl_div = torch.mean(logq - ldj - logp)
        self.log("loss", kl_div)

        return kl_div

    def validation_step(self, batch, *_):
        z, logq = batch
        x, ldj = self(z)
        logp = self.log_norm + self.κ * torch.mv(x, self.μ)

        weights = logp - (logq - ldj)
        self.log("acceptance", metropolis_acceptance(weights))
        self.log("ess", effective_sample_size(weights))

        # Only TensorBoardLogger supported
        if self.logger is None or type(self.logger) is not TensorBoardLogger:
            return

        x = x.to("cpu").detach()

        fig = scatter(x)
        self.logger.experiment.add_figure(
            "3d_projection", fig, self.trainer.global_step
        )

        grid = pairplot(x)
        self.logger.experiment.add_figure(
            "pairplot", grid.figure, self.trainer.global_step
        )

    def test_step(self, batch, *_):
        z, logq = batch
        x, ldj = self(z)
        logp = self.log_norm + self.κ * torch.mv(x, self.μ)

        weights = logp - (logq - ldj)

        self.log("hp/acceptance", metropolis_acceptance(weights))
        self.log("hp/ess", effective_sample_size(weights))
        self.log("hp/kl_div", -torch.mean(weights))

    def train_dataloader(self):
        return SphericalUniformPrior3D(self.batch_size)

    def val_dataloader(self):
        return SphericalUniformPrior3D(self.val_batch_size)

    def test_dataloader(self):
        return self.val_dataloader()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.init_lr)


class NormalizingFlowRQSRQS(NormalizingFlowBase):
    def __init__(
        self,
        κ: float,
        μ: Tensor,
        n_layers: int,
        n_spline: int,
        hidden_shape: list[int],
        activation: str,
        batch_size: int,
        val_batch_size: int,
        init_lr: float,
    ):
        super().__init__(κ, μ, n_layers, batch_size, val_batch_size, init_lr)

        self.z_transform = RQSplineTransform([-1, 1], n_spline)
        self.ϕ_transform = RQSplineTransformCircularDomain(n_spline)

        self.z_params = torch.nn.Parameter(
            torch.stack(
                [self.z_transform.identity_params for _ in range(n_layers)],
                dim=0,
            )
        )
        self.ϕ_params = torch.nn.ModuleList(
            [
                simple_fnn_conditioner(
                    in_features=1,
                    out_features=self.ϕ_transform.n_params,
                    hidden_shape=hidden_shape,
                    activation=activation,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        x, y, z = inputs.split(1, dim=-1)
        ldj = torch.zeros(inputs.shape[0], device=inputs.device)

        ρ = (1 - z**2).clamp(min=1e-12, max=1.0).sqrt()
        u = x / ρ
        v = y / ρ
        ϕ = torch.atan2(v, u)  # [-π, π)
        ϕ = torch.fmod(ϕ + 2 * π, 2 * π)  # [0, 2π)

        for i, (z_params, ϕ_params) in enumerate(
            zip(
                self.z_params.split(1, dim=0),
                self.ϕ_params,
            )
        ):
            z, ldj_z = self.z_transform(z, z_params.expand(*z.shape, -1))
            ϕ, ldj_ϕ = self.ϕ_transform(ϕ, ϕ_params(z).view(*ϕ.shape, -1))

            ldj += ldj_z + ldj_ϕ

            ϕ = self.apply_global_rotation(ϕ, i)

        u = torch.cos(ϕ)
        v = torch.sin(ϕ)

        ρ = (1 - z**2).clamp(min=1e-12, max=1.0).sqrt()
        x = ρ * u
        y = ρ * v

        outputs = torch.cat([x, y, z], dim=-1)
        outputs.squeeze_(dim=1)

        return outputs, ldj


class NormalizingFlowRQSMobius(NormalizingFlowBase):
    def __init__(
        self,
        κ: float,
        μ: Tensor,
        n_layers: int,
        n_spline: int,
        n_mobius: int,
        hidden_shape: list[int],
        activation: str,
        batch_size: int,
        val_batch_size: int,
        init_lr: float,
    ):
        super().__init__(κ, μ, n_layers, batch_size, val_batch_size, init_lr)

        self.z_transform = RQSplineTransform([-1, 1], n_spline)
        self.xy_transform = MobiusMixtureTransform(
            n_mobius, weighted=True, epsilon=1e-3
        )

        self.z_params = torch.nn.Parameter(
            torch.stack(
                [self.z_transform.identity_params for _ in range(n_layers)],
                dim=0,
            )
        )
        self.xy_params = torch.nn.ModuleList(
            [
                simple_fnn_conditioner(
                    in_features=1,
                    out_features=self.xy_transform.n_params,
                    hidden_shape=hidden_shape,
                    activation=activation,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        xy, z = inputs.tensor_split([2], dim=-1)
        ldj = torch.zeros(inputs.shape[0], device=inputs.device)

        ρ = (1 - z**2).clamp(min=1e-12, max=1.0).sqrt()
        xy = xy / ρ

        for i, (z_params, xy_params) in enumerate(
            zip(
                self.z_params.split(1, dim=0),
                self.xy_params,
            )
        ):
            z, ldj_z = self.z_transform(z, z_params.expand(*z.shape, -1))
            xy, ldj_xy = self.xy_transform(xy, xy_params(z).view(*xy.shape[:-1], -1))

            ldj += ldj_z + ldj_xy

            xy = self.apply_global_rotation(xy, i)

        ρ = (1 - z**2).clamp(min=1e-12, max=1.0).sqrt()
        xy = ρ * xy

        outputs = torch.cat([xy, z], dim=-1)
        outputs.squeeze_(dim=1)

        return outputs, ldj


class NormalizingFlowBSRQS(NormalizingFlowBase):
    def __init__(
        self,
        κ: float,
        μ: Tensor,
        n_layers: int,
        n_spline: int,
        hidden_shape: list[int],
        activation: str,
        batch_size: int,
        val_batch_size: int,
        init_lr: float,
    ):
        super().__init__(κ, μ, n_layers, batch_size, val_batch_size, init_lr)

        self.z_transform = BSplineTransform([-1, 1], n_spline)
        self.ϕ_transform = RQSplineTransformCircularDomain(n_spline)

        self.z_params = torch.nn.Parameter(
            torch.stack(
                [self.z_transform.identity_params for _ in range(n_layers)],
                dim=0,
            )
        )
        self.ϕ_params = torch.nn.ModuleList(
            [
                simple_fnn_conditioner(
                    in_features=1,
                    out_features=self.ϕ_transform.n_params,
                    hidden_shape=hidden_shape,
                    activation=activation,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        x, y, z = inputs.split(1, dim=-1)
        ldj = torch.zeros(inputs.shape[0], device=inputs.device)

        ρ = (1 - z**2).clamp(min=1e-12, max=1.0).sqrt()
        u = x / ρ
        v = y / ρ
        ϕ = torch.atan2(v, u)  # [-π, π)
        ϕ = torch.fmod(ϕ + 2 * π, 2 * π)  # [0, 2π)

        for i, (z_params, ϕ_params) in enumerate(
            zip(
                self.z_params.split(1, dim=0),
                self.ϕ_params,
            )
        ):
            z, ldj_z = self.z_transform(z, z_params.expand(*z.shape, -1))
            ϕ, ldj_ϕ = self.ϕ_transform(ϕ, ϕ_params(z).view(*ϕ.shape, -1))

            ldj += ldj_z + ldj_ϕ

            ϕ = self.apply_global_rotation(ϕ, i)

        u = torch.cos(ϕ)
        v = torch.sin(ϕ)

        ρ = (1 - z**2).clamp(min=1e-12, max=1.0).sqrt()
        x = ρ * u
        y = ρ * v

        outputs = torch.cat([x, y, z], dim=-1)
        outputs.squeeze_(dim=1)

        return outputs, ldj


class NormalizingFlowBSMobius(NormalizingFlowBase):
    def __init__(
        self,
        κ: float,
        μ: Tensor,
        n_layers: int,
        n_spline: int,
        n_mobius: int,
        hidden_shape: list[int],
        activation: str,
        batch_size: int,
        val_batch_size: int,
        init_lr: float,
    ):
        super().__init__(κ, μ, n_layers, batch_size, val_batch_size, init_lr)

        self.z_transform = BSplineTransform([-1, 1], n_spline)
        self.xy_transform = MobiusMixtureTransform(
            n_mobius, weighted=True, epsilon=1e-3
        )

        self.z_params = torch.nn.Parameter(
            torch.stack(
                [self.z_transform.identity_params for _ in range(n_layers)],
                dim=0,
            )
        )
        self.xy_params = torch.nn.ModuleList(
            [
                simple_fnn_conditioner(
                    in_features=1,
                    out_features=self.xy_transform.n_params,
                    hidden_shape=hidden_shape,
                    activation=activation,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        xy, z = inputs.tensor_split([2], dim=-1)
        ldj = torch.zeros(inputs.shape[0], device=inputs.device)

        ρ = (1 - z**2).clamp(min=1e-12, max=1.0).sqrt()
        xy = xy / ρ

        for i, (z_params, xy_params) in enumerate(
            zip(
                self.z_params.split(1, dim=0),
                self.xy_params,
            )
        ):
            z, ldj_z = self.z_transform(z, z_params.expand(*z.shape, -1))
            xy, ldj_xy = self.xy_transform(xy, xy_params(z).view(*xy.shape[:-1], -1))

            ldj += ldj_z + ldj_xy

            xy = self.apply_global_rotation(xy, i)

        ρ = (1 - z**2).clamp(min=1e-12, max=1.0).sqrt()
        xy = ρ * xy

        outputs = torch.cat([xy, z], dim=-1)
        outputs.squeeze_(dim=1)

        return outputs, ldj


class DummyNormalizingFlow(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        return inputs, torch.zeros(inputs.shape[0], device=inputs.device)
