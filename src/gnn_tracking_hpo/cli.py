from __future__ import annotations

from argparse import ArgumentParser


def add_test_option(parser: ArgumentParser):
    parser.add_argument(
        "--test",
        help="As-fast-as-possible run to test the setup.",
        is_flag=True,
    )


def add_gpu_option(parser: ArgumentParser):
    parser.add_argument(
        "--gpu",
        help="Run on a GPU",
        is_flag=True,
    )


def add_enqueue_option(parser: ArgumentParser):
    parser.add_argument(
        "--enqueue",
        help="Read trials from these json files and enqueue "
        "them. Json files can either contain dictionary "
        "(single config) or a list thereof.",
        multiple=True,
    )


def add_wandb_options(parser: ArgumentParser):
    """To be used as a decorator. Add command line options for wandb metadata."""

    parser.add_argument("--tags", multiple=True, help="Tags for wandb")
    parser.add_argument(
        "--group",
        help="Wandb group name",
    )
    parser.add_argument("--note", help="Wandb note")
