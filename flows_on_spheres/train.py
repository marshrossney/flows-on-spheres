import pandas as pd
import torch
from tqdm import trange
from typing import TypeAlias

from flows_on_spheres.flows import Flow
from flows_on_spheres.prior import uniform_prior
from flows_on_spheres.metrics import LogWeightMetrics
from flows_on_spheres.target import Density

Tensor: TypeAlias = torch.Tensor

# torch.autograd.set_detect_anomaly(True)


def train(
    flow: Flow,
    target: Density,
    steps: int,
    batch_size: int,
    init_lr: float = 1e-2,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    validation_interval: int = 100,
) -> Flow:
    """
    Train a model.

    Parameters
    ----------
    flow:
        a Normalizing Flow
    target:
        a target
    steps:
        number of training steps
    batch_size:
        size of each training batch
    init_lr:
        initial learning rate for the Adam optimizer
    device:
        device to train on (cpu or cuda)
    validation_interval:
        how often to compute validation metrics

    """
    flow = flow.to(device=device, dtype=dtype)
    prior = uniform_prior(target.dim, device=device, dtype=dtype)
    optimizer = torch.optim.Adam(flow.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps
    )

    flow.target = target

    with trange(steps + 1, desc="Training") as pbar:
        pbar.write(
            "   Step  |  Acc    |  ESS    |      log(p_target) - log(p_model)       "
        )
        pbar.write(
            "         |         |         |   min     |  max    |   mean   |  var   "
        )
        for step in pbar:
            x, log_prior_density = prior(batch_size)

            fx, log_det_jacobian = flow(x)

            log_model_density = log_prior_density - log_det_jacobian
            log_target_density = target.log_density(fx)

            loss = (log_model_density - log_target_density).mean()

            if step % validation_interval == 0:
                log_weights = log_target_density - log_model_density
                metrics = LogWeightMetrics(log_weights)

                output = "  |  ".join(
                    [
                        f"  {step:5d}",
                        f"{metrics.metropolis_acceptance:.3f}",
                        f"{metrics.effective_sample_size:.3f}",
                        f"{metrics.min:07.3f}",
                        f"{metrics.max:.3f}",
                        f"{metrics.mean:.3f}",
                        f"{metrics.variance:.3f}",
                    ]
                )
                pbar.write(output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    return flow.to("cpu")


def test(
    flow: Flow, target: Density, sample_size: int, repeats: int = 1
) -> pd.DataFrame:
    flow = flow.to("cpu")
    prior = uniform_prior(flow.dim, device="cpu", dtype=torch.float32)

    metrics = []
    for _ in range(repeats):
        x, log_prior_density = prior(sample_size)
        fx, log_det_jacobian = flow(x)
        log_model_density = log_prior_density - log_det_jacobian
        log_target_density = target.log_density(fx)

        log_weights = log_target_density - log_model_density

        metrics_ = {"sample_size": sample_size} | LogWeightMetrics(
            log_weights
        ).as_dict()
        metrics.append(metrics_)

    return pd.DataFrame(metrics)
