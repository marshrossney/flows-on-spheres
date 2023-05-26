from datetime import datetime
from pathlib import Path
from typing import Optional

from jsonargparse import ArgumentParser, Namespace
from jsonargparse.typing import PositiveFloat, Path_dw
from matplotlib.pyplot import Figure

from flows_on_spheres.model import FlowBasedModel
from flows_on_spheres.hmc import (
    HamiltonianGaussianMomenta,
    HamiltonianCauchyMomenta,
    FlowedDensity,
)
from flows_on_spheres.visualise import TrajectoryVisualiser

from flows_on_spheres.flows import DummyFlow

CHECKPOINT_FNAME = "trained_model.ckpt"


def trajectory(
    model: FlowBasedModel,
    step_size: PositiveFloat,
    traj_length: PositiveFloat,
    cauchy_gamma: Optional[PositiveFloat] = None,
) -> dict[str, Figure]:
    target = FlowedDensity(model.flow, model.target)
    
    #target = FlowedDensity(DummyFlow(), model.target)

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


def main(config: Namespace) -> None:
    model_path = Path(config.model)
    model = FlowBasedModel.load_from_checkpoint(model_path / CHECKPOINT_FNAME)

    dict_of_figs = trajectory(
        model=model,
        step_size=config.step_size,
        traj_length=config.traj_length,
        cauchy_gamma=config.gamma,
    )

    figures_dir = (
        model_path
        / "figures"
        / f"traj_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    for name, fig in dict_of_figs.items():
        fig.savefig(figures_dir / f"{name}.png", dpi=200, bbox_inches="tight")
