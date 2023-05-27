import logging
from pathlib import Path

from jsonargparse import ArgumentParser, Namespace
from jsonargparse.typing import PositiveInt, Path_dw
from matplotlib.pyplot import Figure
import torch

from flows_on_spheres.visualise import (
    # Visualiser,  # TODO: decide whether it's worth making this class configurable
    CircularFlowVisualiser,
    SphericalFlowVisualiser,
)
from flows_on_spheres.scripts import CHECKPOINT_FNAME

log = logging.getLogger(__name__)


def visualise(
    model: Path_dw,
    sample_size: PositiveInt,
) -> dict[str, Figure]:
    model_path = Path(model)

    flow = torch.load(model_path / CHECKPOINT_FNAME)

    if flow.dim == 1:
        visualiser = CircularFlowVisualiser(flow, flow.target, sample_size)
    elif flow.dim == 2:
        visualiser = SphericalFlowVisualiser(flow, flow.target, sample_size)
    else:
        raise NotImplementedError("Visualisations not implemented for dim > 2")

    dict_of_figs = {name: fig for name, fig in visualiser.figures()}

    return dict_of_figs


parser = ArgumentParser()
parser.add_argument(
    "model",
    type=Path_dw,
    help="path to trained model",
)
parser.add_argument("-n", "--sample_size", type=PositiveInt, default=100000)


def main(config: Namespace):
    dict_of_figs = visualise(config.model, config.sample_size)

    figures_dir = Path(config.model) / "figures"
    if figures_dir.is_dir():
        log.warning(
            "Figures directory already exists: figures will be overwritten"
        )
    figures_dir.mkdir(exist_ok=True)

    for name, fig in dict_of_figs.items():
        fig.savefig(figures_dir / f"{name}.png", dpi=200, bbox_inches="tight")
