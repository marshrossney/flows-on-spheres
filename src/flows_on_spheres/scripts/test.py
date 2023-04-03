from pathlib import Path

from jsonargparse import ArgumentParser, Namespace
from jsonargparse.typing import PositiveInt, Path_dw
import pandas as pd

from flows_on_spheres.model import FlowBasedModel
from flows_on_spheres.utils import get_tester

CHECKPOINT_FNAME = "trained_model.ckpt"
METRICS_FNAME = "metrics.csv"


def test(
    model: Path_dw,
    sample_size: PositiveInt,
    repeat: PositiveInt = 1,
) -> pd.DataFrame:
    model_path = Path(model)

    trained_model = FlowBasedModel.load_from_checkpoint(
        model_path / CHECKPOINT_FNAME, test_sample_size=sample_size
    )

    tester = get_tester()

    metrics = []
    for _ in range(repeat):
        (metrics_,) = tester.test(trained_model)
        metrics.append({"sample_size": sample_size} | metrics_)

    return pd.DataFrame(metrics)


parser = ArgumentParser()
# NOTE: this would be nice but prevents use of flags -m -n -r
# test_parser.add_function_arguments(test, nested_key=None, as_group=False)
parser.add_argument(
    "model",
    type=Path_dw,
    help="path to trained model",
)
parser.add_argument("-n", "--sample_size", type=PositiveInt, default=100000)
parser.add_argument("-r", "--repeat", type=PositiveInt, default=1)


def main(config: Namespace) -> None:
    metrics = test(
        model=config.model,
        sample_size=config.sample_size,
        repeat=config.repeat,
    )

    model_path = Path(config.model)
    metrics_file = model_path / METRICS_FNAME

    if metrics_file.exists():
        existing_metrics = pd.read_csv(metrics_file)
        metrics = pd.concat([existing_metrics, metrics])

    metrics.to_csv(metrics_file, index=False)
