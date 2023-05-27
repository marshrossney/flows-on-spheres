import torch
from tqdm import trange
import pandas as pd

from flows_on_spheres.abc import Flow, Density
from flows_on_spheres.prior import uniform_prior
from flows_on_spheres.metrics import LogWeightMetrics

Tensor = torch.Tensor


def train(
    flow: Flow,
    target: Density,
    steps: int,
    batch_size: int,
    init_lr: float = 1e-2,
    device: str = "cpu",
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

    flow = flow.to(device)
    prior = uniform_prior(target.dim, device=device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps
    )

    flow.target = target

    with trange(steps, desc="Training") as pbar:
        pbar.write("   Step  |  KL     |  Var    |  Acc    |  ESS")
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
                        f"{metrics.kl_divergence:2.3f}",
                        f"{metrics.variance:2.3f}",
                        f"{metrics.metropolis_acceptance:1.3f}",
                        f"{metrics.effective_sample_size:1.3f}",
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
    prior = uniform_prior(flow.dim, device="cpu")

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
