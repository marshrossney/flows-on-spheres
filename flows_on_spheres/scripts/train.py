import logging
from pathlib import Path
from typing import Optional

import torch
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionYesNo,
    Namespace,
    namespace_to_dict,
)
from jsonargparse.typing import Path_dc

from flows_on_spheres.train import train
from flows_on_spheres.scripts.io import CHECKPOINT_FNAME, CONFIG_FNAME

# Imports that just allow you to specify class rather than full class path in config file
from flows_on_spheres.target import (  # noqa: F401
    Density,
    VonMisesFisherDensity,
    VonMisesFisherMixtureDensity,
)
from flows_on_spheres.transforms import (  # noqa: F401
    MobiusModule,
    RQSplineModule,
    CircularSplineModule,
    ProjectedAffineModule,
    SigmoidModule,
    CircularSigmoidModule,
    BSplineModule,
)
from flows_on_spheres.flows import (  # noqa: F401
    Flow,
    CircularFlow,
    RecursiveFlowS2,
    RecursiveFlowSD,
    Composition,
)


parser = ArgumentParser()
parser.add_function_arguments(
    train, nested_key=None, as_group=False, skip=["device", "dtype"]
)
parser.add_argument("--cuda", action=ActionYesNo)
parser.add_argument("--double", action=ActionYesNo)
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

    device_dtype = {
        "device": "cuda" if config.pop("cuda") else "cpu",
        "dtype": torch.float64 if config.pop("double") else torch.float32,
    }
    config |= device_dtype

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

    torch.save(trained_model.state_dict(), output_path / CHECKPOINT_FNAME)

    with (output_path / CONFIG_FNAME).open("w") as file:
        file.write(config_yaml)
