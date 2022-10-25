#!/usr/bin/env python3

from __future__ import annotations

import functools
from functools import partial
from pathlib import Path

import click
from ray import tune
from ray.air import CheckpointConfig, FailureConfig, RunConfig
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune import SyncConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
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
        "--fixed",
        help="Fix config values to these values",
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
    fixed: None | str = None,
    grace_period=3,
):
    """ """
    if gpu:
        run_wandb_offline()

    maybe_run_distributed()

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
            num_samples=50 if not test else 1,
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
            stop={"training_iteration": 40 if not test else 1},
            checkpoint_config=CheckpointConfig(checkpoint_at_end=True),
            log_to_file=True,
            # verbose=1,  # Only status reports, no results
            failure_config=FailureConfig(
                fail_fast=True,
            ),
        ),
    )
    tuner.fit()
