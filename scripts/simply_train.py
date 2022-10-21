#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from gnn_tracking.metrics.losses import EdgeWeightBCELoss
from tune import TCNTrainable
from util import read_json


class ThisTrainable(TCNTrainable):
    def get_cluster_functions(self) -> dict[str, Any]:
        return {}

    def post_setup_hook(self):
        self.trainer.pt_thlds = [0.0]

    def get_lr_scheduler(self):
        return None

    def get_edge_loss_function(self):
        return EdgeWeightBCELoss()


@click.command()
@click.option(
    "--test",
    help="As-fast-as-possible run to test the setup",
    is_flag=True,
    default=False,
)
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
