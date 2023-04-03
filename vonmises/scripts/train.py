from pathlib import Path
from typing import Optional

from jsonargparse import ArgumentParser, ActionConfigFile, Namespace
from jsonargparse.typing import PositiveInt, PositiveFloat, Path_dc

from vonmises.distributions import Density
from vonmises.flows import Flow
from vonmises.model import FlowBasedModel
from vonmises.utils import get_trainer

CHECKPOINT_FNAME = "trained_model.ckpt"
CONFIG_FNAME = "config.yaml"


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


parser = ArgumentParser()
parser.add_function_arguments(train, nested_key=None, as_group=False)
parser.add_argument(
    "-o",
    "--output",
    type=Optional[Path_dc],
    default=None,
    help="location to save trained model and metrics",
)
parser.add_argument("-c", "--config", action=ActionConfigFile)


def main(config: Namespace) -> None:
    config_yaml = parser.dump(config, skip_none=False)

    config = parser.instantiate_classes(config)

    trained_model = train(
        flow=config.flow,
        target=config.target,
        steps=config.steps,
        batch_size=config.batch_size,
        init_lr=config.init_lr,
    )

    if config.output is None:
        return

    output_path = Path(config.output)

    trainer = trained_model.trainer
    trainer.save_checkpoint(output_path / CHECKPOINT_FNAME)

    with (output_path / CONFIG_FNAME).open("w") as file:
        file.write(config_yaml)
