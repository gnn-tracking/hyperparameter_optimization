from __future__ import annotations

from argparse import ArgumentParser


def add_local_option(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--local",
        help="Run locally, not with head node",
        action="store_true",
    )


def add_test_option(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--test",
        help="As-fast-as-possible run to test the setup.",
        action="store_true",
    )


def add_cpu_option(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--cpu",
        help="Do not run on a GPU, only use CPU",
        action="store_true",
    )


def add_enqueue_option(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--enqueue",
        help="Read trials from these json files and enqueue "
        "them. Json files can either contain dictionary "
        "(single config) or a list thereof.",
        nargs="+",
    )


def add_wandb_options(parser: ArgumentParser) -> None:
    """To be used as a decorator. Add command line options for wandb metadata."""

    parser.add_argument("--tags", nargs="+", help="Tags for wandb")
    parser.add_argument(
        "--group",
        help="Wandb group name",
    )
    parser.add_argument("--note", help="Wandb note")
    parser.add_argument("--wandb-project", default="gnn_tracking", help="Wandb project")


def add_truth_cut_options(parser: ArgumentParser) -> None:
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


def add_restore_options(
    parser: ArgumentParser, required=False, prefix="ec", name="edge classifier"
) -> None:
    """Add options for model restoring.

    Args:
        parser: The parser to add the options to.
        required: Whether hash and project are required
        prefix: Prefix for the options
        name: Name of the model to load (e.g., "edge classifier")
    """
    parser.add_argument(
        f"--{prefix}-hash",
        required=required,
        type=str,
        help=f"Hash of the {name} to load",
    )
    parser.add_argument(
        f"--{prefix}-project",
        required=required,
        type=str,
        help=f"Name of the folder that the {name} to load belongs to",
    )
    parser.add_argument(
        f"--{prefix}-epoch",
        type=int,
        default=-1,
        help=f"Epoch of the {name} to load. Defaults to -1 (last epoch).",
    )
