#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import click
from gnn_tracking.metrics.losses import PotentialLoss

from gte.cli import test_option
from gte.config import read_json
from gte.trainable import TCNTrainable


class ThisTrainable(TCNTrainable):
    def post_setup_hook(self):
        self.trainer.pt_thlds = [1.5]

    def get_potential_loss_function(self):
        return PotentialLoss(q_min=self.tc.get("q_min", 0.01), attr_pt_thld=0.0)

    def get_lr_scheduler(self):
        return None


@click.command()
@test_option
@click.option(
    "--config",
    "config_file",
    help="Read config values from file",
)
def main(
    *,
    test=False,
    config_file: None | str = None,
):
    config = {}
    if test:
        config["test"] = True
    if config_file:
        config = read_json(Path(config_file))

    trainable = ThisTrainable()
    trainable.setup(config)
    while True:
        trainable.step()


if __name__ == "__main__":
    main()
