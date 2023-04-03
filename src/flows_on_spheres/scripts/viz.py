import logging
from pathlib import Path

from jsonargparse import ArgumentParser, Namespace
from jsonargparse.typing import PositiveInt, Path_dw
from matplotlib.pyplot import Figure

from flows_on_spheres.model import FlowBasedModel
from flows_on_spheres.visualise import (
    # Visualiser,  # TODO: decide whether it's worth making this class configurable
    CircularFlowVisualiser,
    SphericalFlowVisualiser,
)

log = logging.getLogger(__name__)

CHECKPOINT_FNAME = "trained_model.ckpt"


def visualise(
    model: Path_dw,
    sample_size: PositiveInt,
) -> dict[str, Figure]:
    model_path = Path(model)

    trained_model = FlowBasedModel.load_from_checkpoint(
        model_path / CHECKPOINT_FNAME, test_sample_size=sample_size
    )

    if trained_model.target.dim == 1:
        visualiser = CircularFlowVisualiser(trained_model, sample_size)
    elif trained_model.target.dim == 2:
        visualiser = SphericalFlowVisualiser(trained_model, sample_size)
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
