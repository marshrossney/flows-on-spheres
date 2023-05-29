from pathlib import Path
from typing import Optional

from jsonargparse import ArgumentParser, Namespace, ActionYesNo
from jsonargparse.typing import PositiveInt, PositiveFloat, Path_dw
import pandas as pd
import torch

from flows_on_spheres.abc import Flow, Density
from flows_on_spheres.hmc import (
    HamiltonianGaussianMomenta,
    HamiltonianCauchyMomenta,
    HMCSampler,
    FlowedDensity,
)
from flows_on_spheres.metrics import integrated_autocorrelation
from flows_on_spheres.scripts import (
    CHECKPOINT_FNAME,
    HMC_METRICS_FNAME,
    DUMMY_HMC_METRICS_FNAME,
)


def hmc(
    target: Density,
    step_size: PositiveFloat,
    n_traj: PositiveInt,
    flow: Optional[Flow] = None,
    traj_length: PositiveFloat = 1.0,
    cauchy_gamma: Optional[PositiveFloat] = None,
    n_replicas: PositiveInt = 1,
    n_therm: PositiveInt = 100,
) -> pd.DataFrame:
    if flow is not None:
        target = FlowedDensity(flow, target).to(torch.double)

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

    x = sampler.sample(n_traj, n_therm)  # dims n_traj, n_replicas, D+1

    acc = sampler.acceptance_rate.mean()

    edh = sampler.exp_delta_H.flatten()
    edh_mean = edh.mean()
    edh_stderr = (edh.var(correction=1) / len(edh)).sqrt()

    x = x.transpose(0, 1)  # n_replicas, n_traj, D+1
    log_density = torch.stack(
        [hamiltonian.target.log_density(x_r) for x_r in x],
    )

    tau = integrated_autocorrelation(log_density)
    tau_10pc = tau.quantile(0.1)
    tau_50pc = tau.quantile(0.5)
    tau_90pc = tau.quantile(0.9)

    # TODO Autocorrelation time of true target? (Not flowed density)

    summary = {
        "n_traj": n_traj,
        "step_size": sampler.step_size,  # may have been adjusted
        "traj_length": traj_length,
        "cauchy_gamma": cauchy_gamma,
        "exp_dH_minus_1": float(edh_mean) - 1,
        "exp_dH_stderr": float(edh_stderr),
        "acceptance": float(acc),
        "tau_10pc": float(tau_10pc),
        "tau_50pc": float(tau_50pc),
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
parser.add_argument("--n_therm", type=PositiveInt, default=100)
parser.add_argument("--dummy", action=ActionYesNo)


def main(config: Namespace) -> None:
    model_path = Path(config.model)
    flow = torch.load(model_path / CHECKPOINT_FNAME)

    metrics = hmc(
        flow=None if config.dummy else flow,
        target=flow.target,
        n_traj=config.n_traj,
        step_size=config.step_size,
        traj_length=config.traj_length,
        cauchy_gamma=config.gamma,
        n_replicas=config.n_replicas,
        n_therm=config.n_therm,
    )
    print(metrics)

    metrics_file = model_path / (
        DUMMY_HMC_METRICS_FNAME if config.dummy else HMC_METRICS_FNAME
    )

    if metrics_file.exists():
        existing_metrics = pd.read_csv(metrics_file)
        metrics = pd.concat([existing_metrics, metrics])

    metrics.to_csv(metrics_file, index=False)
