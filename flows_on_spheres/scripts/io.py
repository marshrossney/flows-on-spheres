from pathlib import Path

import torch

from flows_on_spheres.flows import Flow
from flows_on_spheres.target import Density

CHECKPOINT_FNAME = "checkpoint.pt"
CONFIG_FNAME = "config.yaml"
METRICS_FNAME = "metrics.csv"
HMC_METRICS_FNAME = "hmc_metrics.csv"
DUMMY_HMC_METRICS_FNAME = "dummy_hmc_metrics.csv"

def load(path: str | Path) -> tuple[Flow, Density]:
    path = path if isinstance(path, Path) else Path(path)

    checkpoint = torch.load(path / CHECKPOINT_FNAME)

    from flows_on_spheres.scripts.train import parser

    config = parser.parse_path(str(path / CONFIG_FNAME))
    config = parser.instantiate_classes(config)
    flow = config.flow
    target = config.target

    flow.load_state_dict(checkpoint)

    return flow, target

