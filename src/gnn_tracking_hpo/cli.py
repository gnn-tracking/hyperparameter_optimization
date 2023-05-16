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


def add_ec_restore_options(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--ec-hash", required=True, type=str, help="Hash of the edge classifier to load"
    )
    parser.add_argument(
        "--ec-project",
        required=True,
        type=str,
        help="Name of the folder that the edge classifier to load belongs to",
    )
    parser.add_argument(
        "--ec-epoch",
        type=int,
        default=-1,
        help="Epoch of the edge classifier to load. Defaults to -1 (last epoch).",
    )


def add_tc_restore_options(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--tc-hash", required=True, type=str, help="Hash of the TCN to load"
    )
    parser.add_argument(
        "--tc-project",
        required=True,
        type=str,
        help="Name of the folder that the TCN to load belongs to",
    )
    parser.add_argument(
        "--tc-epoch",
        type=int,
        default=-1,
        help="Epoch of the TCN to load. Defaults to -1 (last epoch).",
    )
