from itertools import chain
from math import pi as π
from typing import Optional, TypeAlias

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary

Tensor: TypeAlias = torch.Tensor


def get_trainer(
    steps: Optional[int] = None, device: str = "auto"
) -> pl.Trainer:
    config = dict(
        max_epochs=1,
        accelerator=device,
        limit_train_batches=steps,
        limit_val_batches=1,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        callbacks=[ModelSummary(2)],
    )
    return pl.Trainer(**config)


def get_tester() -> pl.Trainer:
    config = dict(
        max_epochs=1,
        accelerator="cpu",
        limit_train_batches=1,  # not for training
        limit_val_batches=1,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    return pl.Trainer(**config)


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


def mod_2pi(angles: Tensor) -> Tensor:
    return torch.remainder(angles, 2 * π)


def batched_dot(x: Tensor, y: Tensor) -> Tensor:
    return (x * y).sum(dim=-1)


def batched_outer(x: Tensor, y: Tensor) -> Tensor:
    return x.unsqueeze(dim=-2) * y.unsqueeze(dim=-1)


def batched_mv(M: Tensor, v: Tensor) -> Tensor:
    return (M * v.unsqueeze(dim=-2)).sum(dim=-1)


def batched_vm(v: Tensor, M: Tensor) -> Tensor:
    return (v.unsqueeze(dim=-1) * M).sum(dim=-2)


def batched_mm(M: Tensor, N: Tensor) -> Tensor:
    return torch.matmul(M, N)
