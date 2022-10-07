#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import pprint
from functools import partial
from pathlib import Path
from typing import Any

import click
import optuna
import ray
import sklearn.model_selection
from gnn_tracking.graph_construction.graph_builder import GraphBuilder
from gnn_tracking.models.track_condensation_networks import GraphTCN
from gnn_tracking.postprocessing.dbscanscanner import dbscan_scan
from gnn_tracking.training.tcn_trainer import TCNTrainer
from gnn_tracking.utils.log import logger
from gnn_tracking.utils.losses import BackgroundLoss, EdgeWeightFocalLoss, PotentialLoss
from gnn_tracking.utils.seeds import fix_seeds
from gnn_tracking.utils.training import subdict_with_prefix_stripped
from ray import tune
from ray.air import CheckpointConfig, FailureConfig, RunConfig
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune import SyncConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.util.joblib import register_ray
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader


def get_loaders(test=False) -> tuple[GraphBuilder, dict[str, DataLoader]]:
    logger.info("Loading data")
    graph_builder = GraphBuilder(
        str(Path("~/data/gnn_tracking/point_clouds").expanduser()),
        str(Path("~/data/gnn_tracking/graphs").expanduser()),
        redo=False,
    )
    n_graphs = 100 if test else None
    graph_builder.process(n=n_graphs)

    # partition graphs into train, test, val splits
    graphs = graph_builder.data_list
    _train_graphs, test_graphs = sklearn.model_selection.train_test_split(
        graphs, test_size=0.2
    )
    train_graphs, val_graphs = sklearn.model_selection.train_test_split(
        _train_graphs, test_size=0.15
    )

    # build graph loaders
    params = {"batch_size": 1, "num_workers": 1}
    train_loader = DataLoader(list(train_graphs), **params, shuffle=True)
    test_loader = DataLoader(list(test_graphs), **params)
    val_loader = DataLoader(list(val_graphs), **params)
    loaders = {"train": train_loader, "test": test_loader, "val": val_loader}
    return graph_builder, loaders


def get_model(graph_builder, config: dict[str, Any]) -> GraphTCN:
    # use reference graph to get relevant dimensions
    g = graph_builder.data_list[0]
    node_indim = g.x.shape[1]
    edge_indim = g.edge_attr.shape[1]
    hc_outdim = 2  # output dim of latent space
    model = GraphTCN(node_indim, edge_indim, hc_outdim, **config)
    return model


class TCNTrainable(tune.Trainable):
    def setup(self, config: dict[str, Any]):
        test = config.get("test", False)
        self.config = config
        fix_seeds()
        graph_builder, loaders = get_loaders(test=test)

        loss_functions = {
            "edge": EdgeWeightFocalLoss(),
            "potential": PotentialLoss(q_min=config["q_min"]),
            "background": BackgroundLoss(sb=config["sb"]),
        }

        model = get_model(
            graph_builder, config=subdict_with_prefix_stripped(config, "m_")
        )
        cluster_functions = {
            "dbscan": partial(dbscan_scan, n_trials=100 if not test else 1)
        }
        scheduler = partial(StepLR, gamma=0.95, step_size=4)
        self.trainer = TCNTrainer(
            model=model,
            loaders=loaders,
            loss_functions=loss_functions,
            loss_weights=subdict_with_prefix_stripped(config, "lw_"),
            lr=config["lr"],
            lr_scheduler=scheduler,
            cluster_functions=cluster_functions,  # type: ignore
        )
        self.trainer.max_batches_for_clustering = 100 if not test else 10

    def step(self):
        return self.trainer.step(max_batches=self.config.get("max_batches", None))

    def save_checkpoint(
        self,
        checkpoint_dir,
    ):
        return self.trainer.save_checkpoint(
            Path(checkpoint_dir) / "checkpoint.pt",
        )

    def load_checkpoint(self, checkpoint_path, **kwargs):
        self.trainer.load_checkpoint(checkpoint_path, **kwargs)


def suggest_config(trial: optuna.Trial, *, test=False) -> dict[str, Any]:
    # Everything with prefix "m_" is passed to the model
    # Everything with prefix "lw_" is treated as loss weight
    trial.suggest_float("q_min", 1e-3, 1, log=True)
    trial.suggest_float("sb", 0, 1)
    trial.suggest_float("lr", 2e-6, 1e-3, log=True)
    trial.suggest_int("m_hidden_dim", 64, 256)
    trial.suggest_int("m_L_ec", 1, 7)
    trial.suggest_int("m_L_hc", 1, 7)
    trial.suggest_float("lw_potential_attractive", 1, 500)
    trial.suggest_float("lw_potential_repulsive", 1e-2, 1e2)
    trial.suggest_float("focal_gamma", 0, 20)  # 5 might be a good default
    trial.suggest_float("focal_alpha", 0, 1)  # 0.95 might be a good default
    fixed_config = {
        "lw_edge": 500,
        "lw_background": 0.05,
        "test": test,
    }
    return fixed_config


def read_config_from_file(path: Path) -> dict[str, Any]:
    with path.open() as f:
        config = json.load(f)
    return config


@click.command()
@click.option("--test", is_flag=True, default=False)
@click.option("--gpu", is_flag=True, default=False)
@click.option(
    "--restore",
    help="Restore previous training state from this directory",
    default=None,
)
@click.option(
    "--enqueue-trials", help="Read trials from this file and enqueue them", nargs="+"
)
def main(test=False, gpu=False, restore=None, enqueue_trials: None | list[str] = None):
    """

    Args:
        test: Speed up for testing (will only use limited data/epochs)
        gpu: Run on GPUs

    Returns:

    """
    if enqueue_trials is None:
        enqueue_trials = []

    ray.init(address="auto", _redis_password=os.environ["redis_password"])
    register_ray()

    points_to_evaluate = [read_config_from_file(Path(path)) for path in enqueue_trials]
    if points_to_evaluate:
        logger.info("Enqueued trials: %s", pprint.pformat(points_to_evaluate))

    optuna_search = OptunaSearch(
        partial(suggest_config, test=test),
        metric="trk.double_majority",
        mode="max",
        points_to_evaluate=points_to_evaluate,
    )
    if restore:
        print(f"Restoring previous state from {restore}")
        optuna_search.restore_from_dir(restore)

    tuner = tune.Tuner(
        tune.with_resources(
            TCNTrainable,
            {"gpu": 1 if gpu else 0, "cpu": 1},
        ),
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(
                metric="trk.double_majority", mode="max", grace_period=3
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
            stop={"training_iteration": 20 if not test else 1},
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
