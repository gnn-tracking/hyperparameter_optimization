#!/usr/bin/env python3

from __future__ import annotations

import pprint
from functools import partial
from pathlib import Path
from typing import Any

import click
import optuna
from gnn_tracking.metrics.losses import (
    BackgroundLoss,
    EdgeWeightFocalLoss,
    PotentialLoss,
)
from gnn_tracking.models.track_condensation_networks import GraphTCN
from gnn_tracking.postprocessing.dbscanscanner import dbscan_scan
from gnn_tracking.training.tcn_trainer import TCNTrainer
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped
from gnn_tracking.utils.log import logger
from gnn_tracking.utils.seeds import fix_seeds
from ray import tune
from ray.air import CheckpointConfig, FailureConfig, RunConfig
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune import SyncConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch.optim import Adam
from util import (
    della,
    get_fixed_config,
    get_graphs,
    get_loaders,
    maybe_run_distributed,
    read_json,
    run_wandb_offline,
    suggest_if_not_fixed,
)

server = della


class TCNTrainable(tune.Trainable):
    # Do not add blank self.tc or self.trainer to __init__, because it will be called
    # after setup when setting ``reuse_actor == True`` and overwriting your values
    # from set
    def setup(self, config: dict[str, Any]):
        logger.debug("Got config\n%s", pprint.pformat(config))
        self.tc = config
        fix_seeds()
        self.trainer = self.get_trainer()
        logger.debug(f"Trainer: {self.trainer}")

    def get_model(self) -> GraphTCN:
        return GraphTCN(
            node_indim=6, edge_indim=4, **subdict_with_prefix_stripped(self.tc, "m_")
        )

    def get_edge_loss_function(self):
        return EdgeWeightFocalLoss(
            alpha=self.tc.get("focal_alpha", 0.25),
            gamma=self.tc.get("focal_gamma", 2),
        )

    def get_potential_loss_function(self):
        return PotentialLoss(q_min=self.tc.get("q_min", 0.01))

    def get_background_loss_function(self):
        return BackgroundLoss(sb=self.tc.get("sb", 0.1))

    def get_loss_functions(self) -> dict[str, Any]:
        return {
            "edge": self.get_edge_loss_function(),
            "potential": self.get_potential_loss_function(),
            "background": self.get_background_loss_function(),
        }

    def get_cluster_functions(self) -> dict[str, Any]:
        return {
            "dbscan": partial(
                dbscan_scan,
                n_trials=100 if not self.tc.get("test", False) else 1,
                n_jobs=server.cpus_per_gpu if not self.tc.get("test", False) else 1,
            )
        }

    def get_lr_scheduler(self):
        return None

    def get_optimizer(self):
        return Adam

    def get_trainer(self) -> TCNTrainer:
        test = self.tc.get("test", False)
        trainer = TCNTrainer(
            model=self.get_model(),
            loaders=get_loaders(get_graphs(test=test), test=test),
            loss_functions=self.get_loss_functions(),
            loss_weights=subdict_with_prefix_stripped(self.tc, "lw_"),
            lr=self.tc.get("lr", 5e-4),
            lr_scheduler=self.get_lr_scheduler(),
            cluster_functions=self.get_cluster_functions(),  # type: ignore
            optimizer=self.get_optimizer(),
        )
        trainer.max_batches_for_clustering = 100 if not test else 10
        return trainer

    def step(self):
        return self.trainer.step(max_batches=self.tc.get("max_batches", None))

    def save_checkpoint(
        self,
        checkpoint_dir,
    ):
        return self.trainer.save_checkpoint(
            Path(checkpoint_dir) / "checkpoint.pt",
        )

    def load_checkpoint(self, checkpoint_path, **kwargs):
        self.trainer.load_checkpoint(checkpoint_path, **kwargs)


def suggest_config(
    trial: optuna.Trial, *, test=False, fixed: dict[str, Any] | None = None
) -> dict[str, Any]:
    # Everything with prefix "m_" is passed to the model
    # Everything with prefix "lw_" is treated as loss weight kwarg
    fixed_config = get_fixed_config(test=test)
    if fixed is not None:
        fixed_config.update(fixed)

    def sinf_float(key, *args, **kwargs):
        suggest_if_not_fixed(trial.suggest_float, key, fixed_config, *args, **kwargs)

    def sinf_int(key, *args, **kwargs):
        suggest_if_not_fixed(trial.suggest_int, key, fixed_config, *args, **kwargs)

    sinf_float("q_min", 1e-3, 1, log=True)
    sinf_float("sb", 0, 1)
    sinf_float("lr", 2e-6, 1e-3, log=True)
    sinf_int("m_hidden_dim", 64, 256)
    sinf_int("m_L_ec", 1, 7)
    sinf_int("m_L_hc", 1, 7)
    sinf_float("focal_gamma", 0, 20)  # 5 might be a good default
    sinf_float("focal_alpha", 0, 1)  # 0.95 might be a good default
    sinf_float("lw_edge", 0.001, 500)
    sinf_float("lw_potential_attractive", 1, 500)
    sinf_float("lw_potential_repulsive", 1e-2, 1e2)
    return fixed_config


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
    "--restore",
    help="Restore previous training state from this directory",
    default=None,
)
@click.option(
    "--enqueue-trials",
    help="Read trials from this file and enqueue them",
    multiple=True,
)
@click.option(
    "--fixed",
    help="Fix config values to these values",
)
def main(
    *,
    test=False,
    gpu=False,
    restore=None,
    enqueue_trials: None | list[str] = None,
    fixed: None | str = None,
):
    """ """
    if gpu:
        run_wandb_offline()

    maybe_run_distributed()

    points_to_evaluate = [read_json(Path(path)) for path in enqueue_trials or []]
    if points_to_evaluate:
        logger.info("Enqueued trials: %s", pprint.pformat(points_to_evaluate))

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
            TCNTrainable,
            {"gpu": 1 if gpu else 0, "cpu": server.cpus_per_gpu if not test else 1},
        ),
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(
                metric="trk.double_majority_pt1.5",
                mode="max",
                grace_period=5,
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


if __name__ == "__main__":
    main()
