from pathlib import Path

from jsonargparse import ArgumentParser, Namespace
from jsonargparse.typing import PositiveInt, Path_dw
import pandas as pd

from flows_on_spheres.model import FlowBasedModel, get_tester

CHECKPOINT_FNAME = "trained_model.ckpt"
METRICS_FNAME = "metrics.csv"


def test(model: FlowBasedModel, repeats: PositiveInt = 1) -> pd.DataFrame:
    tester = get_tester()

    metrics = []
    for _ in range(repeats):
        (metrics_,) = tester.test(model)
        metrics.append({"sample_size": model.test_sample_size} | metrics_)

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
parser.add_argument("-r", "--repeats", type=PositiveInt, default=1)


def main(config: Namespace) -> None:
    model_path = Path(config.model)

    model = FlowBasedModel.load_from_checkpoint(
        model_path / CHECKPOINT_FNAME, test_sample_size=config.sample_size
    )

    metrics = test(model=model, repeats=config.repeats)

    metrics_file = model_path / METRICS_FNAME

    if metrics_file.exists():
        existing_metrics = pd.read_csv(metrics_file)
        metrics = pd.concat([existing_metrics, metrics])

    metrics.to_csv(metrics_file, index=False)
