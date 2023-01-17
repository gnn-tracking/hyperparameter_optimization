from __future__ import annotations

from argparse import ArgumentParser


def add_test_option(parser: ArgumentParser):
    parser.add_argument(
        "--test",
        help="As-fast-as-possible run to test the setup.",
        action="store_true",
    )


def add_gpu_option(parser: ArgumentParser):
    parser.add_argument(
        "--gpu",
        help="Run on a GPU",
        action="store_true",
    )


def add_enqueue_option(parser: ArgumentParser):
    parser.add_argument(
        "--enqueue",
        help="Read trials from these json files and enqueue "
        "them. Json files can either contain dictionary "
        "(single config) or a list thereof.",
        nargs="+",
    )


def add_wandb_options(parser: ArgumentParser):
    """To be used as a decorator. Add command line options for wandb metadata."""

    parser.add_argument("--tags", nargs="+", help="Tags for wandb")
    parser.add_argument(
        "--group",
        help="Wandb group name",
    )
    parser.add_argument("--note", help="Wandb note")


def add_truth_cut_options(parser: ArgumentParser):
    parser.add_argument(
        "--training-pt-thld",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--training-without-noise",
        action="store_true",
    )
    parser.add_argument(
        "--training-without-non-reconstructable",
        action="store_true",
    )
