#!/usr/bin/env python3

"""Population based training"""

from __future__ import annotations

import pprint
from pathlib import Path
from typing import Any

import click
import ray
from gnn_tracking.metrics.losses import EdgeWeightBCELoss
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped
from gnn_tracking.utils.log import logger
from ray import air
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune import SyncConfig
from ray.tune.schedulers import PopulationBasedTraining
from torch.optim import SGD
from tune import TCNTrainable
from util import (
    della,
    get_fixed_config,
    maybe_run_distributed,
    read_json,
    run_wandb_offline,
)

server = della


def get_param_space():
    return {
        "q_min": ray.tune.loguniform(1e-3, 1),
        "sb": ray.tune.uniform(0, 1),
        "lr": ray.tune.loguniform(2e-6, 1e-3),
        # "focal_gamma": ray.tune.uniform(0, 20),
        # "focal_alpha": ray.tune.uniform(0, 1),
        "lw_edge": ray.tune.uniform(1, 500),
        "lw_potential_attractive": ray.tune.uniform(1, 500),
        "lw_potential_repulsive": ray.tune.uniform(1e-2, 1e2),
    }


class PBTTrainable(TCNTrainable):
    def get_optimizer(self):
        return SGD

    def get_lr_scheduler(self):
        return None

    def get_edge_loss_function(self):
        return EdgeWeightBCELoss()

    def post_setup_hook(self):
        self.trainer.pt_thlds = [1.5]

    def reset_config(self, new_config: dict[str, Any]):
        logger.debug("Reset config called with\n%s", pprint.pformat(new_config))
        self.tc = new_config
        self.trainer.loss_functions = self.get_loss_functions()
        self.trainer.optimizer.lr = self.tc["lr"]
        self.trainer.loss_weights = subdict_with_prefix_stripped(self.tc, "lw_")


def get_trainable(test=False):
    class FixedConfigTCNTrainable(PBTTrainable):
        def setup(self, config):
            fixed_config = get_fixed_config(test=test)
            config.update(fixed_config)
            super().setup(config)

    return FixedConfigTCNTrainable


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
@click.option(
    "--enqueue-trials",
    help="Read trials from this file and enqueue them",
    multiple=True,
)
def main(
    *,
    test=False,
    gpu=False,
    enqueue_trials: None | list[str] = None,
):
    """ """
    if gpu:
        run_wandb_offline()

    maybe_run_distributed()

    points_to_evaluate = [read_json(Path(path)) for path in enqueue_trials or []]
    if points_to_evaluate:
        logger.info("Enqueued trials:\n%s", pprint.pformat(points_to_evaluate))

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=5,
        hyperparam_mutations=get_param_space(),
    )
    if test:
        logger.warning("Running in test mode")

    tuner = ray.tune.Tuner(
        ray.tune.with_resources(
            get_trainable(test),
            {"gpu": 1 if gpu else 0, "cpu": server.cpus_per_gpu if not test else 1},
        ),
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
                checkpoint_score_attribute="trk.double_majority_pt1.5",
                num_to_keep=4,
            ),
        ),
        tune_config=ray.tune.TuneConfig(
            scheduler=scheduler,
            metric="trk.double_majority_pt1.5",
            mode="max",
            num_samples=4,
            reuse_actors=True,
        ),
        param_space=get_param_space(),
    )
    tuner.fit()


if __name__ == "__main__":
    main()
