from pathlib import Path
from typing import Optional

from jsonargparse import ArgumentParser, Namespace
from jsonargparse.typing import PositiveInt, PositiveFloat, Path_dw, Path_dc
import pandas as pd

from vonmises.model import FlowBasedModel
import vonmises.hmc

CHECKPOINT_FNAME = "trained_model.ckpt"
HMC_METRICS_FNAME = "hmc_metrics.csv"


def fhmc(
    model: Path_dw,
    step_size: PositiveFloat,
    sample_size: PositiveInt,
    traj_length: PositiveFloat = 1.0,
    repeat: PositiveInt = 1,
):
    model_path = Path(model)

    model = FlowBasedModel.load_from_checkpoint(model_path / CHECKPOINT_FNAME)

    cfg = {
        "sample_size": sample_size,
        "step_size": step_size,
        "traj_length": traj_length,
    }

    metrics = []
    for _ in range(repeat):
        _, acceptance = vonmises.hmc.fhmc(
            model.flow,
            model.target,
            sample_size=sample_size,
            step_size=step_size,
            traj_length=traj_length,
        )
        metrics.append(cfg | {"acceptance": acceptance})

    return pd.DataFrame(metrics)


parser = ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    type=Optional[Path_dc],
    default=None,
    help="path to trained model",
)
parser.add_argument("-n", "--sample_size", type=PositiveInt, required=True)
parser.add_argument("-s", "--step_size", type=PositiveFloat, required=True)
parser.add_argument("-T", "--traj_length", type=PositiveFloat, default=1.0)
parser.add_argument("-r", "--repeat", type=PositiveInt, default=1)


def main(config: Namespace) -> None:
    metrics = fhmc(
        model=config.model,
        sample_size=config.sample_size,
        step_size=config.step_size,
        traj_length=config.traj_length,
        repeat=config.repeat,
    )

    model_path = Path(config.model)

    metrics_file = model_path / HMC_METRICS_FNAME

    if metrics_file.exists():
        existing_metrics = pd.read_csv(metrics_file)
        metrics = pd.concat([existing_metrics, metrics])

    metrics.to_csv(metrics_file, index=False)
