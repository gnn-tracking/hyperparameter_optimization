#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import click
from tune import TCNTrainable
from util import read_json


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

    trainable = TCNTrainable()
    trainable.setup(config)
    while True:
        trainable.step()
