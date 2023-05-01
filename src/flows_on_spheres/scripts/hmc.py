from pathlib import Path
from typing import Optional

from jsonargparse import ArgumentParser, Namespace
from jsonargparse.typing import PositiveInt, PositiveFloat, Path_dw
import pandas as pd
import torch

from flows_on_spheres.model import FlowBasedModel
from flows_on_spheres.hmc import (
    HamiltonianGaussianMomenta,
    HamiltonianCauchyMomenta,
    HMCSampler,
    FlowedDensity,
)
from flows_on_spheres.metrics import integrated_autocorrelation

CHECKPOINT_FNAME = "trained_model.ckpt"
HMC_METRICS_FNAME = "hmc_metrics.csv"


def hmc(
    model: FlowBasedModel,
    step_size: PositiveFloat,
    n_traj: PositiveInt,
    traj_length: PositiveFloat = 1.0,
    cauchy_gamma: Optional[PositiveFloat] = None,
    n_replicas: PositiveInt = 1,
) -> pd.DataFrame:
    target = FlowedDensity(model.flow, model.target)

    if cauchy_gamma is None:
        hamiltonian = HamiltonianGaussianMomenta(target)
    else:
        hamiltonian = HamiltonianCauchyMomenta(target, cauchy_gamma)

    sampler = HMCSampler(
        hamiltonian=hamiltonian,
        step_size=step_size,
        traj_length=traj_length,
        n_replicas=n_replicas,
    )

    x = sampler.sample(n_traj)  # dims n_traj, n_replicas, D+1

    acc_mean = sampler.acceptance_rate.mean()
    acc_std = sampler.acceptance_rate.std(dim=0, correction=1)

    edh_mean = sampler.exp_delta_H.mean()
    edh_std = sampler.exp_delta_H.std(dim=0, correction=1)

    tau = []
    for x_rep in x.split(1, dim=1):
        log_density = hamiltonian.target.log_density(x_rep.squeeze(1))
        tau.append(integrated_autocorrelation(log_density))
    tau = torch.tensor(tau)
    tau_50pc = tau.quantile(0.5)
    tau_75pc = tau.quantile(0.75)
    tau_90pc = tau.quantile(0.9)

    summary = {
        "n_traj": n_traj,
        "step_size": sampler.step_size,  # may have been adjusted
        "traj_length": traj_length,
        "cauchy_gamma": cauchy_gamma,
        "acceptance_mean": float(acc_mean),
        "acceptance_std": float(acc_std) if not acc_std.isnan() else None,
        "exp_delta_H_mean": float(edh_mean),
        "exp_delta_H_std": float(edh_std) if not edh_std.isnan() else None,
        "tau_50pc": float(tau_50pc),
        "tau_75pc": float(tau_75pc),
        "tau_90pc": float(tau_90pc),
        "n_replicas": n_replicas,
    }

    return pd.DataFrame([summary])


parser = ArgumentParser()
parser.add_argument(
    "model",
    type=Path_dw,
    help="path to trained model",
)
parser.add_argument("-n", "--n_traj", type=PositiveInt, required=True)
parser.add_argument("-s", "--step_size", type=PositiveFloat, required=True)
parser.add_argument("-T", "--traj_length", type=PositiveFloat, default=1.0)
parser.add_argument(
    "-g", "--gamma", type=Optional[PositiveFloat], default=None
)
parser.add_argument("-r", "--n_replicas", type=PositiveInt, default=1)


def main(config: Namespace) -> None:
    model_path = Path(config.model)
    model = FlowBasedModel.load_from_checkpoint(model_path / CHECKPOINT_FNAME)

    metrics = hmc(
        model=model,
        n_traj=config.n_traj,
        step_size=config.step_size,
        traj_length=config.traj_length,
        cauchy_gamma=config.gamma,
        n_replicas=config.n_replicas,
    )
    print(metrics)

    metrics_file = model_path / HMC_METRICS_FNAME

    if metrics_file.exists():
        existing_metrics = pd.read_csv(metrics_file)
        metrics = pd.concat([existing_metrics, metrics])

    metrics.to_csv(metrics_file, index=False)
