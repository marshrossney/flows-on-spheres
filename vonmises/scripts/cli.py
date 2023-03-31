from pathlib import Path
from typing import Optional

import pandas as pd
from jsonargparse import ArgumentParser, ActionConfigFile, Namespace
from jsonargparse.typing import PositiveInt, PositiveFloat, Path_dc

from vonmises.scripts.commands import train, test, hmc

CHECKPOINT_FNAME = "trained_model.ckpt"
CONFIG_FNAME = "config.yaml"
METRICS_FNAME = "metrics.csv"
HMC_METRICS_FNAME = "hmc_metrics.csv"


train_parser = ArgumentParser()
train_parser.add_function_arguments(train, nested_key=None, as_group=False)
train_parser.add_argument(
    "-o",
    "--output",
    type=Optional[Path_dc],
    default=None,
    help="location to save trained model and metrics",
)
train_parser.add_argument("-c", "--config", action=ActionConfigFile)

test_parser = ArgumentParser()
# NOTE: this would be nice but prevents use of flags -m -n -r
# test_parser.add_function_arguments(test, nested_key=None, as_group=False)
test_parser.add_argument(
    "-m",
    "--model",
    type=Optional[Path_dc],
    default=None,
    help="path to trained model",
)
test_parser.add_argument(
    "-n", "--sample_size", type=PositiveInt, default=pow(2, 14)
)
test_parser.add_argument("-r", "--repeat", type=PositiveInt, default=1)

hmc_parser = ArgumentParser()
# hmc_parser.add_function_arguments(hmc, nested_key=None, as_group=False)
hmc_parser.add_argument(
    "-m",
    "--model",
    type=Optional[Path_dc],
    default=None,
    help="path to trained model",
)
hmc_parser.add_argument("-n", "--sample_size", type=PositiveInt, required=True)
hmc_parser.add_argument("-s", "--step_size", type=PositiveFloat, required=True)
hmc_parser.add_argument("-T", "--traj_length", type=PositiveFloat, default=1.0)
hmc_parser.add_argument("-r", "--repeat", type=PositiveInt, default=1)

parser = ArgumentParser(prog="cli")
subcommands = parser.add_subcommands()
subcommands.add_subcommand("train", train_parser)
subcommands.add_subcommand("test", test_parser)
subcommands.add_subcommand("hmc", hmc_parser)


def _train(config: Namespace) -> None:
    config_yaml = train_parser.dump(config, skip_none=False)

    config = train_parser.instantiate_classes(config)

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


def _test(config: Namespace) -> None:
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


def _hmc(config: Namespace) -> None:
    metrics = hmc(
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


def cli(config: Optional[dict] = None):
    config = (
        parser.parse_object(config)
        if config is not None
        else parser.parse_args()
    )

    if config.subcommand == "train":
        _train(config.train)
    elif config.subcommand == "test":
        _test(config.test)
    elif config.subcommand == "hmc":
        _hmc(config.hmc)
    else:
        raise ValueError


if __name__ == "__main__":
    cli()
