from abc import ABC, abstractmethod
from math import pi as π
from typing import TypeAlias, Optional

import torch
import torch.nn.functional as F

from vonmises.geometry import apply_global_rotation
from vonmises.utils import make_net
from vonmises.transforms import Transformer

Tensor: TypeAlias = torch.Tensor
Module: TypeAlias = torch.nn.Module
Parameter: TypeAlias = torch.nn.Parameter


class Flow(Module, ABC):
    @abstractmethod
    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        ...


class DummyFlow(Flow):
    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        return inputs, torch.zeros(inputs.shape[0], device=inputs.device)


class CircularFlow(Flow):
    def __init__(
        self,
        transformer: Transformer,
        n_layers: int,
    ):
        super().__init__()
        self.transformer = transformer
        self.transformer_parameters = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.rand_like(transformer.identity_params)
                )
                for _ in range(n_layers)
            ]
        )
        self.rotations = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.empty(1).uniform_(0, 2 * π))
                for _ in range(n_layers)
            ]
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        xy = inputs
        ldj_full = torch.zeros(inputs.shape[0], device=inputs.device)

        for params, θ in zip(self.transformer_parameters, self.rotations):
            xy, ldj_this = self.transformer(
                xy, params.expand(*xy.shape[:-1], -1)
            )
            xy = apply_global_rotation(xy, θ)
            ldj_full += ldj_this

        return xy, ldj_full


class RecursiveFlowS2(Flow):
    def __init__(
        self,
        z_transformer: Transformer,
        xy_transformer: Transformer,
        n_layers: int,
        net_hidden_shape: Optional[list[int]] = None,
        net_activation: str = "Tanh",
        softmax_beta: float = 1e8,
    ):
        super().__init__()

        self.softmax_beta = softmax_beta
        self.z_transformer = z_transformer
        self.xy_transformer = xy_transformer
        self.z_params = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.rand_like(z_transformer.identity_params)
                )
                for _ in range(n_layers)
            ]
        )
        if net_hidden_shape is None:
            self.xy_params = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.rand_like(xy_transformer.identity_params)
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.xy_params = torch.nn.ModuleList(
                [
                    make_net(
                        1,
                        xy_transformer.n_params,
                        net_hidden_shape,
                        net_activation,
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

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        xy, z = inputs.split([2, 1], dim=-1)
        delta_log_vol = torch.zeros(inputs.shape[0], device=inputs.device)

        # Map sphere to cylinder
        ρ = F.softplus(1 - z**2, beta=self.softmax_beta).sqrt()
        xy = xy / ρ

        for layer_idx, (z_params, xy_params, θ) in enumerate(
            zip(self.z_params, self.xy_params, self.rotations)
        ):
            z, ldj_z = self.z_transformer(
                z, z_params.expand(*z.shape[:-1], -1)
            )

            xy_params = (
                xy_params.expand(*xy.shape[:-1], -1)
                if isinstance(xy_params, Parameter)
                else xy_params(z)
            )
            xy, ldj_xy = self.xy_transformer(
                xy, xy_params.view(*xy.shape[:-1], -1)
            )

            xy = apply_global_rotation(xy, θ)

            delta_log_vol += ldj_z + ldj_xy

        ρ = F.softplus(1 - z**2, beta=self.softmax_beta).sqrt()
        xy = ρ * xy

        xyz = torch.cat([xy, z], dim=-1)

        return xyz, delta_log_vol
