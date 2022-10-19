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
    graph_builder.process(stop=n_graphs)

    # partition graphs into train, test, val splits
    graphs = graph_builder.data_list
    _train_graphs, test_graphs = sklearn.model_selection.train_test_split(
        graphs, test_size=0.2
    )
    train_graphs, val_graphs = sklearn.model_selection.train_test_split(
        _train_graphs, test_size=0.15
    )

    # build graph loaders
    params = {"batch_size": 21 if not test else 1, "num_workers": 6 if not test else 1}
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


def _sinf(f, key, config, *args, **kwargs):
    """Suggest if not fixed"""
    if key not in config:
        f(key, *args, **kwargs)


def suggest_config(
    trial: optuna.Trial, *, test=False, fixed: dict[str, Any] | None = None
) -> dict[str, Any]:
    # Everything with prefix "m_" is passed to the model
    # Everything with prefix "lw_" is treated as loss weight kwarg
    fixed_config = {
        "test": test,
        "lw_edge": 500,
    }
    if fixed is not None:
        fixed_config.update(fixed)

    def sinf(f, key, *args, **kwargs):
        _sinf(f, key, fixed_config, *args, **kwargs)

    def sinf_float(key, *args, **kwargs):
        sinf(trial.suggest_float, key, *args, **kwargs)

    def sinf_int(key, *args, **kwargs):
        sinf(trial.suggest_int, key, *args, **kwargs)

    sinf_float("q_min", 1e-3, 1, log=True)
    sinf_float("sb", 0, 1)
    sinf_float("lr", 2e-6, 1e-3, log=True)
    sinf_int("m_hidden_dim", 64, 256)
    sinf_int("m_L_ec", 1, 7)
    sinf_int("m_L_hc", 1, 7)
    sinf_float("focal_gamma", 0, 20)  # 5 might be a good default
    sinf_float("focal_alpha", 0, 1)  # 0.95 might be a good default
    sinf_float("lw_potential_attractive", 1, 500)
    sinf_float("lw_potential_repulsive", 1e-2, 1e2)
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
    "--enqueue-trials",
    help="Read trials from this file and enqueue them",
    multiple=True,
)
@click.option(
    "--fixed",
    help="Fix config values to these values",
)
def main(
    test=False,
    gpu=False,
    restore=None,
    enqueue_trials: None | list[str] = None,
    fixed: None | str = None,
):
    """

    Args:
        test: Speed up for testing (will only use limited data/epochs)
        gpu: Run on GPUs

    Returns:

    """
    if enqueue_trials is None:
        enqueue_trials = []

    if "redis_password" in os.environ:
        # We're running distributed
        ray.init(address="auto", _redis_password=os.environ["redis_password"])
        register_ray()

    points_to_evaluate = [read_config_from_file(Path(path)) for path in enqueue_trials]
    if points_to_evaluate:
        logger.info("Enqueued trials: %s", pprint.pformat(points_to_evaluate))

    fixed_config = None
    if fixed:
        fixed_config = read_config_from_file(Path(fixed))

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
            {"gpu": 1 if gpu else 0, "cpu": 12 if not test else 1},
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
