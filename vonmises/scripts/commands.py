from pathlib import Path

from jsonargparse.typing import PositiveInt, PositiveFloat, Path_dw
import pandas as pd
from matplotlib.pyplot import Figure

from vonmises.distributions import Density
from vonmises.flows import Flow
from vonmises.model import FlowBasedModel
from vonmises.utils import get_trainer, get_tester
from vonmises.hmc import fhmc

CHECKPOINT_FNAME = "trained_model.ckpt"


def train(
    flow: Flow,
    target: Density,
    steps: PositiveInt,
    batch_size: PositiveInt,
    init_lr: PositiveFloat = 0.01,
) -> FlowBasedModel:
    model = FlowBasedModel(
        flow=flow,
        target=target,
        batch_size=batch_size,
        init_lr=init_lr,
    )

    trainer = get_trainer(steps)

    trainer.fit(model)

    trainer.test(model)

    return model


def test(
    model: Path_dw,
    sample_size: PositiveInt = pow(2, 14),
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


def viz(
    model: Path_dw,
    sample_size: PositiveInt = pow(2, 14),
) -> dict[Figure]:
    model_path = Path(model)

    trained_model = FlowBasedModel.load_from_checkpoint(
        model_path / CHECKPOINT_FNAME, test_sample_size=sample_size
    )

    dict_of_figs = {}

    #

    return dict_of_figs


def hmc(
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
        _, acceptance = fhmc(
            model.flow,
            model.target,
            sample_size=sample_size,
            step_size=step_size,
            traj_length=traj_length,
        )
        metrics.append(cfg | {"acceptance": acceptance})

    return pd.DataFrame(metrics)
