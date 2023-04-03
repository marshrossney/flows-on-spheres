from typing import Optional

from jsonargparse import ArgumentParser

import flows_on_spheres.scripts.train as train
import flows_on_spheres.scripts.test as test
import flows_on_spheres.scripts.fhmc as fhmc
import flows_on_spheres.scripts.viz as viz


parser = ArgumentParser(prog="cli")
subcommands = parser.add_subcommands()
subcommands.add_subcommand("train", train.parser)
subcommands.add_subcommand("test", test.parser)
subcommands.add_subcommand("fhmc", fhmc.parser)
subcommands.add_subcommand("viz", viz.parser)


def cli(config: Optional[dict] = None):
    config = (
        parser.parse_object(config)
        if config is not None
        else parser.parse_args()
    )

    if config.subcommand == "train":
        train.main(config.train)
    elif config.subcommand == "test":
        test.main(config.test)
    elif config.subcommand == "fhmc":
        fhmc.main(config.fhmc)
    elif config.subcommand == "viz":
        viz.main(config.viz)
    else:
        raise ValueError("Invalid subcommand")


if __name__ == "__main__":
    cli()
