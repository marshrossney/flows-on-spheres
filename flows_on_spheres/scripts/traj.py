from datetime import datetime
from pathlib import Path
from typing import Optional

from jsonargparse import ArgumentParser, Namespace, ActionYesNo
from jsonargparse.typing import PositiveFloat, Path_dw
from matplotlib.pyplot import Figure
import torch

from flows_on_spheres.abc import Density, Flow
from flows_on_spheres.hmc import (
    HamiltonianGaussianMomenta,
    HamiltonianCauchyMomenta,
    FlowedDensity,
)
from flows_on_spheres.visualise import TrajectoryVisualiser
from flows_on_spheres.scripts import CHECKPOINT_FNAME


def visualise_trajectory(
    target: Density,
    step_size: PositiveFloat,
    traj_length: PositiveFloat,
    flow: Optional[Flow] = None,
    cauchy_gamma: Optional[PositiveFloat] = None,
) -> dict[str, Figure]:
    if flow is not None:
        target = FlowedDensity(flow, target)

    if cauchy_gamma is None:
        hamiltonian = HamiltonianGaussianMomenta(target)
    else:
        hamiltonian = HamiltonianCauchyMomenta(target, cauchy_gamma)

    visualiser = TrajectoryVisualiser(hamiltonian)
    visualiser.forward(step_size=step_size, traj_length=traj_length)

    dict_of_figs = {name: fig for name, fig in visualiser.figures()}

    return dict_of_figs


parser = ArgumentParser()
parser.add_argument(
    "model",
    type=Path_dw,
    help="path to trained model",
)
parser.add_argument("-s", "--step_size", type=PositiveFloat, required=True)
parser.add_argument("-T", "--traj_length", type=PositiveFloat, default=1.0)
parser.add_argument(
    "-g", "--gamma", type=Optional[PositiveFloat], default=None
)
parser.add_argument("--dummy", action=ActionYesNo)


def main(config: Namespace) -> None:
    model_path = Path(config.model)

    flow = torch.load(model_path / CHECKPOINT_FNAME)

    dict_of_figs = visualise_trajectory(
        flow=None if config.dummy else flow,
        target=flow.target,
        step_size=config.step_size,
        traj_length=config.traj_length,
        cauchy_gamma=config.gamma,
    )

    figures_dir = (
        "dummy_traj_" if config.dummy else "traj_"
    ) + datetime.now().strftime("%Y%m%d%H%M%S")
    figures_dir = model_path / "figures" / figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    for name, fig in dict_of_figs.items():
        fig.savefig(figures_dir / f"{name}.png", dpi=200, bbox_inches="tight")
