from math import exp, pi as π
from os import PathLike
from random import random
from typing import Optional, TypeAlias

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

Tensor: TypeAlias = torch.Tensor


def mod_2pi(angles: Tensor) -> Tensor:
    return torch.remainder(angles, 2 * π)


def batched_dot(x: Tensor, y: Tensor) -> Tensor:
    return (x * y).sum(dim=-1)


def batched_outer(x: Tensor, y: Tensor) -> Tensor:
    return x.unsqueeze(dim=-2) * y.unsqueeze(dim=-1)


def batched_mv(M: Tensor, v: Tensor) -> Tensor:
    return (M * v.unsqueeze(dim=-2)).sum(dim=-1)


# M = torch.rand(3, 3)
# v = torch.rand(3)
# res = batched_mv(M.view(1, 3, 3), v.view(1, 3))
# assert torch.allclose(M @ v, res)

# ---------- Geometry


# --------- Training


DEFAULT_TRAINER_CONFIG = dict(
    accelerator="auto",
    check_val_every_n_epoch=None,
    limit_val_batches=1,
    limit_test_batches=1,
    num_sanity_val_steps=0,
    logger=False,
    enable_checkpointing=False,
)


def Trainer(
    steps: int,
    output_path: Optional[str | PathLike] = None,
    logging: bool = False,
) -> pl.Trainer:
    config = DEFAULT_TRAINER_CONFIG.copy()

    config |= {"max_steps": steps}

    if output_path is not None:
        config |= {
            "default_root_dir": output_path,
            "enable_checkpointing": True,
            "callbacks": [ModelCheckpoint(save_last=True)],
        }

    if logging:
        config |= ({"logger": TensorBoardLogger(), "val_check_interval": 200},)

    return pl.Trainer(**config)


# ------Metrics


def metropolis_acceptance(log_weights: Tensor) -> float:
    log_weights = log_weights.tolist()
    current = log_weights.pop(0)

    idx = 0
    indices = []

    for proposal in log_weights:
        # Deal with this case separately to avoid overflow
        if proposal > current:
            current = proposal
            idx += 1
        elif random() < min(1, exp(proposal - current)):
            current = proposal
            idx += 1

        indices.append(idx)

    transitions = set(indices)
    transitions.discard(0)  # there was no transition *to* 0th state

    return len(transitions) / len(log_weights)


def effective_sample_size(log_weights: Tensor) -> float:
    ess = torch.exp(log_weights.logsumexp(0).mul(2) - log_weights.mul(2).logsumexp(0))
    ess.div_(len(log_weights))
    return float(ess)
