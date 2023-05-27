from pathlib import Path

from jsonargparse import ArgumentParser, Namespace
from jsonargparse.typing import PositiveInt, Path_dw
import pandas as pd
import torch

from flows_on_spheres.scripts import (
    CHECKPOINT_FNAME,
    METRICS_FNAME,
)
from flows_on_spheres.train import test


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

    flow = torch.load(model_path / CHECKPOINT_FNAME)

    metrics = test(
        flow,
        target=flow.target,
        sample_size=config.sample_size,
        repeats=config.repeats,
    )
    print(metrics)

    metrics_file = model_path / METRICS_FNAME

    if metrics_file.exists():
        existing_metrics = pd.read_csv(metrics_file)
        metrics = pd.concat([existing_metrics, metrics])

    metrics.to_csv(metrics_file, index=False)
