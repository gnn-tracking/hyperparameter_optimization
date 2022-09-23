#!/usr/bin/env python3

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import click
import optuna
import sklearn.model_selection
import torch
from gnn_tracking.graph_construction.graph_builder import GraphBuilder
from gnn_tracking.models.track_condensation_networks import GraphTCN
from gnn_tracking.postprocessing.dbscanscanner import dbscan_scan
from gnn_tracking.training.tcn_trainer import TCNTrainer
from gnn_tracking.utils.log import logger
from gnn_tracking.utils.losses import BackgroundLoss, EdgeWeightBCELoss, PotentialLoss
from gnn_tracking.utils.seeds import fix_seeds
from gnn_tracking.utils.training import subdict_with_prefix_stripped
from ray import tune
from ray.air import CheckpointConfig, FailureConfig, RunConfig
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune import SyncConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader


def get_loaders(test=False) -> tuple[GraphBuilder, dict[str, DataLoader]]:
    logger.info("Loading data")
    graph_builder = GraphBuilder(
        str(Path("~/data/gnn_tracking/point_clouds").expanduser()),
        str(Path("~/data/gnn_tracking/graphs").expanduser()),
        redo=False,
    )
    n_graphs = 3 if test else None
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
    params = {"batch_size": 1, "shuffle": True, "num_workers": 1}
    train_loader = DataLoader(list(train_graphs), **params)
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
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        graph_builder, loaders = get_loaders(test=test)

        print(f"Utilizing {device}")

        loss_functions = {
            "edge": EdgeWeightBCELoss().to(device),
            "potential": PotentialLoss(q_min=config["q_min"], device=device).to(device),
            "background": BackgroundLoss(sb=config["sb"], device=device).to(device),
        }

        model = get_model(
            graph_builder, config=subdict_with_prefix_stripped(config, "m_")
        )
        model.to(device)

        scheduler = partial(StepLR, gamma=0.95, step_size=4)
        self.trainer = TCNTrainer(
            model=model,
            loaders=loaders,
            loss_functions=loss_functions,
            loss_weights=subdict_with_prefix_stripped(config, "lw_"),
            lr=config["lr"],
            lr_scheduler=scheduler,
            cluster_functions={"dbscan": partial(dbscan_scan, n_trials=100 if not test else 1)},  # type: ignore
            device=device,
        )

    def step(self):
        return self.trainer.step(max_batches=self.config.get("max_batches", None))

    def save_checkpoint(self, checkpoint_dir):
        return self.trainer.save_checkpoint(Path(checkpoint_dir) / "checkpoint.pt")

    def load_checkpoint(self, checkpoint_path):
        self.trainer.load_checkpoint(checkpoint_path)


def suggest_config(trial: optuna.Trial, *, test=False) -> dict[str, Any]:
    # Everything with prefix "m_" is passed to the model
    # Everything with prefix "lw_" is treated as loss weight
    trial.suggest_loguniform("q_min", 1e-3, 1),
    trial.suggest_float("sb", 0, 1),
    trial.suggest_loguniform("lr", 2e-6, 1e-3),
    trial.suggest_categorical("m_hidden_dim", [64, 128, 256]),
    trial.suggest_categorical("m_L_ec", [3, 5, 7]),
    trial.suggest_categorical("m_L_hc", [1, 2, 3, 4]),
    trial.suggest_categorical("lw_potential_attractive", [100, 500]),
    fixed_config = {
        "lw_edge": 500,
        "lw_potential_repulsive": 5,
        "lw_background": 0.05,
        "test": test,
    }
    return fixed_config


@click.command()
@click.option("--test", is_flag=True, default=False)
@click.option("--gpu", is_flag=True, default=False)
def main(test=False, gpu=False):
    """

    Args:
        test: Speed up for testing (will only use limited data/epochs)
        gpu: Run on GPUs

    Returns:

    """

    optuna_search = OptunaSearch(
        partial(suggest_config, test=test),
        metric="trk.double_majority",
        mode="max",
    )

    tuner = tune.Tuner(
        tune.with_resources(
            TCNTrainable, {"gpu": 4 if gpu else 0, "cpu": 4 if gpu else 1}
        ),
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(metric="trk.double_majority", mode="max"),
            num_samples=10 if not test else 1,
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
            stop={"training_iteration": 10 if not test else 1},
            checkpoint_config=CheckpointConfig(checkpoint_at_end=True),
            log_to_file=True,
            verbose=1,  # Only status reports, no results
            failure_config=FailureConfig(
                fail_fast=True,
            ),
        ),
    )
    tuner.fit()


if __name__ == "__main__":
    main()
