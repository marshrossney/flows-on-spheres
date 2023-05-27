import logging
from pathlib import Path
from typing import Optional

import torch
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    Namespace,
    namespace_to_dict,
)
from jsonargparse.typing import Path_dc

from flows_on_spheres.train import train
from flows_on_spheres.scripts import CHECKPOINT_FNAME, CONFIG_FNAME

# Imports that just allow you to specify class rather than full class path in config file
from flows_on_spheres.abc import Density, Flow  # noqa: F401
from flows_on_spheres.target import (  # noqa: F401
    VonMisesFisherDensity,
    VonMisesFisherMixtureDensity,
)
from flows_on_spheres.transforms import (  # noqa: F401
    MobiusModule,
    RQSplineModule,
    ProjectedAffineModule,
)


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

    config = namespace_to_dict(config)
    _ = config.pop("config")

    output = config.pop("output")
    if output is None:
        logging.getLogger().warning(
            "No output directory specified: trained model will not be saved!"
        )
    if output is not None:
        output_path = Path(output)
        if output_path.exists():
            logging.getLogger().warning(
                "Output directory exists: its contents will be overwritten!"
            )

    trained_model = train(**config)

    if output is None:
        return

    output_path.mkdir(exist_ok=True, parents=True)

    torch.save(trained_model, output_path / CHECKPOINT_FNAME)

    with (output_path / CONFIG_FNAME).open("w") as file:
        file.write(config_yaml)
