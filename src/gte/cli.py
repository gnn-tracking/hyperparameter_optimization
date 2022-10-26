from __future__ import annotations

import functools

import click

test_option = click.option(
    "--test",
    help="As-fast-as-possible run to test the setup",
    is_flag=True,
    default=False,
)
gpu_option = click.option(
    "--gpu",
    help="Run on a GPU. This will also assume that you are on a batch node without "
    "internet access and will set wandb mode to offline.",
    is_flag=True,
    default=False,
)
enqueue_option = click.option(
    "--enqueue",
    help="Read trials from this file and enqueue them",
    multiple=True,
)


def wandb_options(f):
    """To be used as a decorator. Add command line options for wandb metadata."""

    @click.option("--tags", multiple=True, help="Tags for wandb")
    @click.option(
        "--group",
        help="Wandb group name",
    )
    @click.option("--note", help="Wandb note")
    @functools.wraps(f)
    def wrapper_common_options(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_common_options
