#!/usr/bin/env python3

from __future__ import annotations

import functools
from functools import partial
from pathlib import Path

import click
import pytimeparse
from ray import tune
from ray.air import CheckpointConfig, FailureConfig, RunConfig
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune import SyncConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.stopper import CombinedStopper, TimeoutStopper, TrialPlateauStopper
from util import (
    della,
    enqueue_option,
    get_points_to_evaluate,
    gpu_option,
    maybe_run_distributed,
    read_json,
    run_wandb_offline,
    test_option,
)

server = della


def common_options(f):
    @test_option
    @gpu_option
    @click.option(
        "--restore",
        help="Restore previous training state from this directory",
        default=None,
    )
    @enqueue_option
    @click.option(
        "--only-enqueued",
        help="Only run enqueued points, do not tune any parameters",
        is_flag=True,
    )
    @click.option(
        "--fixed",
        help="Read config values from file and fix these values in all trials.",
    )
    @click.option(
        "--timeout",
        help="Stop all trials after certain time. Natural time specifications "
        "supported.",
    )
    @functools.wraps(f)
    def wrapper_common_options(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_common_options


def main(
    trainable,
    suggest_config,
    *,
    test=False,
    gpu=False,
    restore=None,
    enqueue: None | list[str] = None,
    only_enqueued=False,
    fixed: None | str = None,
    grace_period=3,
    timeout=None,
):
    """
    For most argument, see corresponding command line interface.

    Args:
        trainable: The trainable to run.
        suggest_config: A function that returns a config dictionary.
        grace_period: Grace period for ASHA scheduler.
    """
    if gpu:
        run_wandb_offline()

    maybe_run_distributed()

    timeout_seconds = pytimeparse.parse(timeout) if timeout else None
    del timeout

    points_to_evaluate = get_points_to_evaluate(enqueue)

    fixed_config = None
    if fixed:
        fixed_config = read_json(Path(fixed))

    optuna_search = OptunaSearch(
        partial(suggest_config, test=test, fixed=fixed_config),
        metric="trk.double_majority_pt1.5",
        mode="max",
        points_to_evaluate=points_to_evaluate,
    )
    if restore:
        print(f"Restoring previous state from {restore}")
        optuna_search.restore_from_dir(restore)

    num_samples = 50
    if test:
        num_samples = 1
    if only_enqueued:
        num_samples = len(points_to_evaluate)

    stoppers = [
        TrialPlateauStopper(
            metric="total",
        )
    ]
    if timeout_seconds is not None:
        stoppers.append(TimeoutStopper(timeout_seconds))
    stopper = CombinedStopper(*stoppers)

    tuner = tune.Tuner(
        tune.with_resources(
            trainable,
            {"gpu": 1 if gpu else 0, "cpu": server.cpus_per_gpu if not test else 1},
        ),
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(
                metric="trk.double_majority_pt1.5",
                mode="max",
                grace_period=grace_period,
            ),
            num_samples=num_samples,
            search_alg=optuna_search,
        ),
        run_config=RunConfig(
            name="tcn",
            callbacks=[
                WandbLoggerCallback(
                    api_key_file="~/.wandb_api_key", project="gnn_tracking"
                ),
            ],
            sync_config=SyncConfig(syncer=None),
            stop=stopper,
            checkpoint_config=CheckpointConfig(checkpoint_at_end=True),
            log_to_file=True,
            # verbose=1,  # Only status reports, no results
            failure_config=FailureConfig(
                fail_fast=True,
            ),
        ),
    )
    tuner.fit()
