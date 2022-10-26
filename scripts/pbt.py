#!/usr/bin/env python3

"""Population based training"""

from __future__ import annotations

import pprint
from typing import Any

import click
import ray
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped
from gnn_tracking.utils.log import logger
from ray import air
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune import SyncConfig
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search import BasicVariantGenerator
from torch.optim import SGD

from gte.cli import enqueue_option, gpu_option, test_option
from gte.config import della, get_fixed_config, get_points_to_evaluate
from gte.orchestrate import maybe_run_distributed, maybe_run_wandb_offline
from gte.trainable import TCNTrainable

server = della


def get_param_space():
    return {
        "q_min": ray.tune.uniform(1e-3, 0.01),
        "sb": ray.tune.uniform(0, 1),
        # "lr": ray.tune.choice([1e-5]),
        # "unnec_param": ray.tune.choice([1, 2, 3, 4, 5, 6]),
        "lr": ray.tune.loguniform(2e-6, 1e-3),
        # "focal_gamma": ray.tune.uniform(0, 20),
        # "focal_alpha": ray.tune.uniform(0, 1),
        "lw_edge": ray.tune.choice([100]),
        "lw_potential_attractive": ray.tune.uniform(1, 500),
        "lw_potential_repulsive": ray.tune.uniform(1e-2, 1e2),
    }


class PBTTrainable(TCNTrainable):
    def get_optimizer(self):
        return SGD

    def get_lr_scheduler(self):
        return None

    def reset_config(self, new_config: dict[str, Any]):
        logger.debug("Reset config called with\n%s", pprint.pformat(new_config))
        self.tc = new_config
        self.trainer.loss_functions = self.get_loss_functions()
        # self.trainer.optimizer.lr = self.tc.get("lr", 5e-4)
        self.trainer.loss_weights = subdict_with_prefix_stripped(self.tc, "lw_")


def get_trainable(test=False):
    logger.debug("Returning FCTCNT")

    class FixedConfigTCNTrainable(PBTTrainable):
        def setup(self, config):
            logger.debug("Original config:\n%s", pprint.pformat(config))
            fixed_config = get_fixed_config(test=test)
            config.update(fixed_config)
            super().setup(config)
            self.post_setup_hook()

    return FixedConfigTCNTrainable


@click.command()
@test_option
@gpu_option
@enqueue_option
def main(
    *,
    test=False,
    gpu=False,
    enqueue: None | list[str] = None,
):
    """ """
    maybe_run_wandb_offline()

    maybe_run_distributed()

    points_to_evaluate = get_points_to_evaluate(enqueue)

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
            reuse_actors=False,
            search_alg=BasicVariantGenerator(points_to_evaluate=points_to_evaluate)
            if points_to_evaluate
            else None,
        ),
        param_space=get_param_space(),
    )
    tuner.fit()


if __name__ == "__main__":
    main()
