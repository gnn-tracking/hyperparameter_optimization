#!/usr/bin/env python3

"""Population based training"""

from __future__ import annotations

import os

import click
import ray
from gnn_tracking.utils.log import logger
from ray import air
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune import SyncConfig
from ray.tune.schedulers import PopulationBasedTraining
from ray.util.joblib import register_ray
from tune import TCNTrainable


def get_param_space():
    return {
        "q_min": ray.tune.loguniform(1e-3, 1),
        "sb": ray.tune.uniform(0, 1),
        "lr": ray.tune.loguniform(2e-6, 1e-3),
        "focal_gamma": ray.tune.uniform(0, 20),
        "focal_alpha": ray.tune.uniform(0, 1),
        "lw_edge": ray.tune.uniform(0.001, 500),
        "lw_potential_attractive": ray.tune.uniform(1, 500),
        "lw_potential_repulsive": ray.tune.uniform(1e-2, 1e2),
    }


@click.command()
@click.option(
    "--test",
    help="As-fast-as-possible run to test the setup",
    is_flag=True,
    default=False,
)
@click.option(
    "--gpu",
    help="Run on a GPU. This will also assume that you are on a batch node without "
    "internet access and will set wandb mode to offline.",
    is_flag=True,
    default=False,
)
def main(
    test=False,
    gpu=False,
):
    """ """
    if gpu:
        logger.warning(
            "Setting wandb mode to offline because we assume you don't have internet"
            " on a GPU node."
        )
        os.environ["WANDB_MODE"] = "offline"

    if "redis_password" in os.environ:
        # We're running distributed
        ray.init(address="auto", _redis_password=os.environ["redis_password"])
        register_ray()

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=5,
        hyperparam_mutations=get_param_space(),
    )

    tuner = ray.tune.Tuner(
        TCNTrainable,
        run_config=air.RunConfig(
            name="pbt_test",
            callbacks=[
                WandbLoggerCallback(
                    api_key_file="~/.wandb_api_key", project="gnn_tracking"
                ),
            ],
            sync_config=SyncConfig(syncer=None),
            stop={"training_iteration": 40 if not test else 1},
            verbose=1,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute="mean_accuracy",
                num_to_keep=4,
            ),
        ),
        tune_config=ray.tune.TuneConfig(
            scheduler=scheduler,
            metric="mean_accuracy",
            mode="max",
            num_samples=4,
        ),
        param_space=get_param_space(),
    )
    tuner.fit()


if __name__ == "__main__":
    main()
