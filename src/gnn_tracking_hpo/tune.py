#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import optuna
import pytimeparse
from ray import tune
from ray.air import CheckpointConfig, FailureConfig, RunConfig
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune import Callback, Stopper, SyncConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.stopper import (  # TrialPlateauStopper,
    CombinedStopper,
    MaximumIterationStopper,
    TimeoutStopper,
)
from rt_stoppers_contrib.no_improvement import NoImprovementTrialStopper
from wandb_osh.ray_hooks import TriggerWandbSyncRayHook

from gnn_tracking_hpo.cli import (
    add_enqueue_option,
    add_gpu_option,
    add_test_option,
    add_wandb_options,
)
from gnn_tracking_hpo.config import della, get_points_to_evaluate, read_json
from gnn_tracking_hpo.orchestrate import maybe_run_distributed, maybe_run_wandb_offline

server = della


def add_common_options(parser: ArgumentParser):
    add_test_option(parser)
    add_gpu_option(parser)
    parser.add_argument(
        "--restore",
        help="Restore previous training state from this directory",
        default=None,
    )
    add_enqueue_option(parser)
    parser.add_argument(
        "--only-enqueued",
        help="Only run enqueued points, do not tune any parameters",
        action="store_true",
    )
    parser.add_argument(
        "--fixed",
        help="Read config values from file and fix these values in all trials.",
    )
    parser.add_argument(
        "--timeout",
        help="Stop all trials after certain time. Natural time specifications "
        "supported.",
    )
    parser.add_argument(
        "--fail-slow",
        help="Do not abort tuning after trial fails.",
        action="store_true",
    )
    parser.add_argument(
        "--dname",
        help="Name of ray output directory",
        default="tcn",
    )
    parser.add_argument(
        "--no-tune",
        help="Do not run tuner, simply train (useful for debugging)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--n-trials",
        dest="num_samples",
        help="Maximum number of trials to run",
        default=None,
        type=int,
    )
    add_wandb_options(parser)


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
    tags=None,
    group=None,
    note=None,
    fail_slow=False,
    dname="tcn",
    metric="trk.double_majority_pt1.5",
    no_improvement_patience=10,
    no_tune=False,
    num_samples=None,
    additional_stoppers=None,
):
    """
    For most arguments, see corresponding command line interface.

    Args:
        trainable: The trainable to run.
        suggest_config: A function that returns a config dictionary.
        grace_period: Grace period for ASHA scheduler.
        no_improvement_patience: Number of iterations without improvement before
            stopping
        thresholds: Thresholds for stopping: Mapping of epoch -> expected FOM
    """
    if additional_stoppers is None:
        additional_stoppers = []
    if no_tune:
        assert test
        study = optuna.create_study()
        trial = study.ask()
        config = suggest_config(trial, test=test)
        config = {**config, **trial.params}
        assert config["test"]
        train_instance = trainable(config)
        for _ in range(2):
            train_instance.trainer.step(max_batches=1)
        raise SystemExit(0)

    maybe_run_wandb_offline()

    maybe_run_distributed()

    if timeout is None:
        timeout_seconds = None
    else:
        timeout_seconds = pytimeparse.parse(timeout)
        if timeout_seconds is None:
            raise ValueError(
                "Could not parse timeout. Try specifying a unit, " "e.g., 1h13m"
            )
    del timeout

    points_to_evaluate = get_points_to_evaluate(enqueue)

    fixed_config = None
    if fixed:
        fixed_config = read_json(Path(fixed))

    optuna_search = OptunaSearch(
        partial(suggest_config, test=test, fixed=fixed_config),
        metric=metric,
        mode="max",
        points_to_evaluate=points_to_evaluate,
    )
    if restore:
        print(f"Restoring previous state from {restore}")
        optuna_search.restore_from_dir(restore)

    num_samples = num_samples or 20
    if test:
        num_samples = 1
    if only_enqueued:
        num_samples = len(points_to_evaluate)

    stoppers: list[Stopper] = [
        NoImprovementTrialStopper(
            metric=metric,
            patience=no_improvement_patience,
            mode="max",
            grace_period=grace_period,
        ),
        *additional_stoppers,
    ]
    if timeout_seconds is not None:
        stoppers.append(TimeoutStopper(timeout_seconds))
    if test:
        stoppers.append(MaximumIterationStopper(1))
    stopper = CombinedStopper(*stoppers)

    callbacks: list[Callback] = []
    if not test:
        callbacks = [
            WandbLoggerCallback(
                api_key_file="~/.wandb_api_key",
                project="gnn_tracking",
                tags=tags,
                group=group,
                notes=note,
            ),
            TriggerWandbSyncRayHook(),
        ]

    if test:
        dname += "_test"

    tuner = tune.Tuner(
        tune.with_resources(
            trainable,
            {"gpu": 1 if gpu else 0, "cpu": server.cpus_per_gpu if not test else 1},
        ),
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(
                metric=metric,
                mode="max",
                grace_period=grace_period,
            ),
            num_samples=num_samples,
            search_alg=optuna_search,
        ),
        run_config=RunConfig(
            name=dname,
            callbacks=callbacks,
            sync_config=SyncConfig(syncer=None),
            stop=stopper,
            checkpoint_config=CheckpointConfig(
                checkpoint_score_attribute=metric,
                checkpoint_score_order="max",
                num_to_keep=5,
            ),
            log_to_file=True,
            # verbose=1,  # Only status reports, no results
            failure_config=FailureConfig(
                fail_fast=not fail_slow,
            ),
        ),
    )
    tuner.fit()
