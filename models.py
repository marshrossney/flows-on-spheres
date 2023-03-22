import math
from typing import TypeAlias, Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import seaborn as sns
import torch
import torch.linalg as LA
import torch.nn.functional as F

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
Module: TypeAlias = torch.nn.Module
Parameter: TypeAlias = torch.nn.Parameter

π = math.pi


class RecursiveFlowS2(pl.LightningModule):
    def __init__(
        self,
        κ: float,
        μ: Tensor,
        z_transformer: Callable,
        xy_transformer: Callable,
        n_layers: int,
        batch_size: int,
        val_batch_size: int = pow(2, 12),
        test_batch_size: int = pow(2, 14),
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: str = "Tanh",
        init_lr: float = 0.001,
        softmax_beta: float = 1e8,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.κ = κ
        μ = μ if isinstance(μ, torch.Tensor) else torch.tensor(μ, dtype=torch.float32)
        μ.div_(LA.vector_norm(μ))
        self.register_buffer("μ", μ)
        self.log_norm = math.log(κ / (4 * π * math.sinh(κ)))

        self.n_layers = n_layers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.init_lr = init_lr
        self.softmax_beta = softmax_beta

        assert isinstance(z_transformer, RQSplineTransform) or isinstance(
            z_transformer, BSplineTransform
        )
        assert isinstance(
            xy_transformer, RQSplineTransformCircularDomain
        ) or isinstance(xy_transformer, MobiusMixtureTransform)
        self.z_transformer = z_transformer
        self.xy_transformer = xy_transformer
        self.polar_mode = isinstance(xy_transformer, RQSplineTransformCircularDomain)

        self.z_params = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.tensor(z_transformer.identity_params))
                for _ in range(n_layers)
            ]
        )
        if net_hidden_shape is None:
            self.xy_params = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.tensor(xy_transformer.identity_params))
                    for _ in range(n_layers)
                ]
            )
        else:
            self.xy_params = torch.nn.ModuleList(
                [
                    simple_fnn_conditioner(
                        in_features=1,
                        out_features=xy_transformer.n_params,
                        hidden_shape=net_hidden_shape,
                        activation=net_activation,
                    )
                    for _ in range(n_layers)
                ]
            )

        # Random global rotations applied after each layer (potentially trainable)
        self.rotations = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.empty(1).uniform_(0, 2 * π))
                for _ in range(n_layers)
            ]
        )

        if self.logger is not None and type(self.logger) is TensorBoardLogger:
            self.logger.log_hyperparams(self.hparams)

    def _apply_global_rotation(self, xy_or_ϕ: Tensor, θ: Tensor) -> Tensor:
        *data_shape, coord_dims = xy_or_ϕ.shape

        if self.polar_mode:  # working with angles
            assert coord_dims == 1
            ϕ = xy_or_ϕ
            return mod_2pi(ϕ + θ)
        else:  # working with euclidean coords
            assert coord_dims == 2
            xy = xy_or_ϕ
            cosθ, sinθ = θ.cos(), θ.sin()
            # counter-clockwise rotation
            R = torch.tensor(
                [[cosθ, -sinθ], [sinθ, cosθ]],
                device=xy.device,
            ).view(*[1 for _ in data_shape], 2, 2)
            xy.unsqueeze_(dim=-2)
            return (R * xy).sum(dim=-1)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        xy, z = inputs.split([2, 1], dim=-1)
        ldj = torch.zeros(inputs.shape[0], device=inputs.device)

        # Map sphere to cylinder
        ρ = F.softplus(1 - z**2, beta=self.softmax_beta).sqrt()
        xy = xy / ρ

        if self.polar_mode:
            x, y = xy.split(1, dim=-1)
            xy_or_ϕ = mod_2pi(torch.atan2(y, x))  # [0, 2π)
        else:
            xy_or_ϕ = xy

        for layer_idx, (z_params, xy_params, θ) in enumerate(
            zip(self.z_params, self.xy_params, self.rotations)
        ):
            z, ldj_z = self.z_transformer(z, z_params.expand(*z.shape, -1))

            xy_params = (
                xy_params.expand(*xy_or_ϕ.shape[:-1], -1)
                if isinstance(xy_params, Parameter)
                else xy_params(z)
            )
            xy_or_ϕ, ldj_xy = self.xy_transformer(
                xy_or_ϕ, xy_params.view(*xy_or_ϕ.shape, -1)
            )

            xy_or_ϕ = self._apply_global_rotation(xy_or_ϕ, θ)

            ldj += ldj_z + ldj_xy

        if self.polar_mode:
            xy = torch.cat([torch.cos(xy_or_ϕ), torch.sin(xy_or_ϕ)], dim=-1)
        else:
            xy = xy_or_ϕ

        ρ = F.softplus(1 - z**2, beta=self.softmax_beta).sqrt()
        xy = ρ * xy

        outputs = torch.cat([xy, z], dim=-1)
        outputs.squeeze_(dim=1)

        return outputs, ldj

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
        return SphericalUniformPrior3D(self.test_batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.init_lr)


class DummyNormalizingFlow(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        return inputs, torch.zeros(inputs.shape[0], device=inputs.device)
