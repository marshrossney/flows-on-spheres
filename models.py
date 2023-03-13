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
from transforms import MobiusTransform, RQSplineTransform, RQSplineTransformCircularDomain, C2SplineTransform
from utils import metropolis_acceptance, effective_sample_size, spherical_mesh, simple_fnn_conditioner
from visualisations import scatter, pairplot

Tensor: TypeAlias = torch.Tensor

π = math.pi


class NormalizingFlowBase(pl.LightningModule):
    def __init__(
        self,
        κ: float,
        μ: Tensor,
        batch_size: int,
        val_batch_size: int,
        init_lr: float = 0.001,
    ):
        super().__init__()
        
        self.κ = κ
        μ = μ if isinstance(μ, torch.Tensor) else torch.tensor(μ, dtype=torch.float32)
        μ.div_(LA.vector_norm(μ))
        self.register_buffer("μ", μ)
        
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.init_lr = init_lr

        self.log_norm = math.log(κ / (4 * π * math.sinh(κ)))

        self.mesh = spherical_mesh(25)
        
        self.save_hyperparameters()
        if self.logger is not None and type(self.logger) is TensorBoardLogger:
            self.logger.log_hyperparams(self.hparams)

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
        
    def train_dataloader(self):
        return SphericalUniformPrior3D(self.batch_size)
    
    def val_dataloader(self):
        return SphericalUniformPrior3D(self.val_batch_size)
        
    def test_dataloader(self):
        return self.val_dataloader()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.init_lr)


class NormalizingFlowRQS(NormalizingFlowBase):
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
        super().__init__(κ, μ, batch_size, val_batch_size, init_lr)
        
        if n_layers % 2 == 1:
            print("An even number of layers is recommended")

        self.z_spline = RQSplineTransform([-1, 1], n_spline)
        self.ϕ_spline = RQSplineTransformCircularDomain(n_spline)

        self.z_spline_params = torch.nn.Parameter(
            torch.stack(
                [self.z_spline.identity_params for _ in range(n_layers)],
                dim=0,
            )
        )
        self.ϕ_spline_params = torch.nn.ModuleList(
            [
                simple_fnn_conditioner(
                    in_features=1,
                    out_features=self.ϕ_spline.n_params,
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

        for z_spline_params, ϕ_spline_params in zip(
            self.z_spline_params.split(1, dim=0),
            self.ϕ_spline_params,
        ):
            z, ldj_z = self.z_spline(z, z_spline_params.expand(*z.shape, -1))
            ϕ, ldj_ϕ = self.ϕ_spline(ϕ, ϕ_spline_params(z).view(*ϕ.shape, -1))

            ldj += ldj_z + ldj_ϕ
            
            ϕ = torch.fmod(ϕ + π, 2 * π)  # effecively shift spline fixed points by π

        u = torch.cos(ϕ)
        v = torch.sin(ϕ)

        ρ = (1 - z**2).clamp(min=1e-12, max=1.0).sqrt()
        x = ρ * u
        y = ρ * v

        outputs = torch.cat([x, y, z], dim=-1)
        outputs.squeeze_(dim=1)

        return outputs, ldj
    

class NormalizingFlowC2(NormalizingFlowBase):
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
        super().__init__(κ, μ, batch_size, val_batch_size, init_lr)

        self.spline_transform = C2SplineTransform([-1, 1], n_spline)
        self.mobius_transform = MobiusTransform()

        self.z_params = torch.nn.Parameter(
            torch.stack(
                [self.spline_transform.identity_params for _ in range(n_layers)],
                dim=0,
            )
        )
        self.xy_params = torch.nn.ModuleList(
            [
                simple_fnn_conditioner(
                    in_features=1,
                    out_features=self.mobius_transform.n_params,
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

        for z_params, xy_params in zip(
            self.z_params.split(1, dim=0),
            self.xy_params,
        ):
            z, ldj_z = self.spline_transform(z, z_params.expand(*z.shape, -1))
            
            # [-1, 1]
            omega_x, omega_y = torch.tanh(xy_params(z)).clamp(-1 + 1e-3, 1 - 1e-3).view(*xy.shape).split(1, dim=-1)
            omega_y = omega_y * torch.sqrt(1 - omega_x.pow(2))  # [-sqrt(1 - x^2), sqrt(1 - x^2)]
            omega = torch.cat([omega_x, omega_y], dim=-1)
            xy, ldj_xy = self.mobius_transform(xy, omega)
            
            #print(omega.pow(2).sum(dim=-1).abs().max())

            #print("z: ", z.isnan().any())
            #print("xy: ", xy.isnan().any())
            #print("ldj_z: ", ldj_z.isnan().any())
            #print("ldj_xy: ", ldj_xy.isnan().any())

            ldj += ldj_z + ldj_xy

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