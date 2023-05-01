from typing import Optional

from jsonargparse import ArgumentParser

import flows_on_spheres.scripts.train as train
import flows_on_spheres.scripts.test as test
import flows_on_spheres.scripts.hmc as hmc
import flows_on_spheres.scripts.traj as traj
import flows_on_spheres.scripts.viz as viz


parser = ArgumentParser(prog="cli")
subcommands = parser.add_subcommands()
subcommands.add_subcommand("train", train.parser)
subcommands.add_subcommand("test", test.parser)
subcommands.add_subcommand("hmc", hmc.parser)
subcommands.add_subcommand("traj", traj.parser)
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
    elif config.subcommand == "hmc":
        hmc.main(config.hmc)
    elif config.subcommand == "traj":
        traj.main(config.traj)
    elif config.subcommand == "viz":
        viz.main(config.viz)
    else:
        raise ValueError("Invalid subcommand")


if __name__ == "__main__":
    cli()
