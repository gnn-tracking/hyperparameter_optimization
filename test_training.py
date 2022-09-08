#!/usr/bin/env python3

from functools import partial
from pathlib import Path
from typing import Any

import click
import numpy as np
import sklearn.model_selection
import torch
from ray import tune
from ray.air import RunConfig
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from torch_geometric.loader import DataLoader
import random
from gnn_tracking.graph_construction.graph_builder import GraphBuilder
from gnn_tracking.models.track_condensation_networks import GraphTCN
from gnn_tracking.training.graph_tcn_trainer import GraphTCNTrainer
from gnn_tracking.utils.losses import PotentialLoss, BackgroundLoss, \
    EdgeWeightBCELoss
from gnn_tracking.utils.training import subdict_with_prefix_stripped
from hyperopt import hp


def get_loaders(test=False) -> tuple[GraphBuilder, dict[str, DataLoader]]:
    graph_builder = GraphBuilder(
        str(Path("~/data/gnn_tracking/point_clouds").expanduser()),
        str(Path("~/data/gnn_tracking/graphs").expanduser()),
        redo=False,
    )
    n_graphs = 1 if test else None
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

    params = {"batch_size": 1, "shuffle": False, "num_workers": 2}
    test_loader = DataLoader(list(test_graphs), **params)
    val_loader = DataLoader(list(val_graphs), **params)
    loaders = {"train": train_loader, "test": test_loader, "val": val_loader}
    print("Loader sizes:", [(k, len(v)) for k, v in loaders.items()])
    return graph_builder, loaders


def get_model(graph_builder, config: dict[str, Any]) -> GraphTCN:
    # use reference graph to get relevant dimensions
    g = graph_builder.data_list[0]
    node_indim = g.x.shape[1]
    edge_indim = g.edge_attr.shape[1]
    hc_outdim = 2  # output dim of latent space
    model = GraphTCN(node_indim, edge_indim, hc_outdim, **config)
    return model


def train(config: dict[str, Any], test=False):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    graph_builder, loaders = get_loaders(test=test)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Utilizing {device}")

    loss_functions = {
        "edge": EdgeWeightBCELoss().to(device),
        "potential": PotentialLoss(q_min=config["q_min"], device=device),
        "background": BackgroundLoss(device=device, sb=config["sb"]),
    }

    model = get_model(graph_builder, config=subdict_with_prefix_stripped(config, "model_"))
    trainer = GraphTCNTrainer(
        model=model,
        loaders=loaders,
        loss_functions=loss_functions,
        loss_weights=subdict_with_prefix_stripped(config, "loss_weight_"),
        lr=config["lr"]
    )

    epochs = 2 if test else 1000
    max_batches = 1 if test else None
    trainer.train(epochs=epochs, max_batches=max_batches)


@click.command()
@click.option("--test", is_flag=True, default=False)
def main(test=False):
    """

    Args:
        test: Speed up for testing (will only use limited data/epochs)

    Returns:

    """
    space = {
        "q_min": hp.loguniform("q_min", -3, 0),
        "sb": 1,
        "lr": hp.loguniform("lr", -11, -7),  # 2e-6 to 1e-3
        # Everything with prefix "model_" is passed to the model
        "model_hidden_dim": hp.choice("hidden_dim", [32, 64, 128, 256]),
        # Everything with prefix "loss_weight_" is treated as loss weight
        "loss_weight_potential_repulsive": hp.loguniform("loss_weight_potential_repulsive", -3, 5),
    }

    hyperopt_search = HyperOptSearch(
        space, metric="mean_accuracy", mode="max", n_initial_points=40
    )

    tuner = tune.Tuner(
        partial(train, test=test),
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
            num_samples=50,
            search_alg=hyperopt_search,
        ),
        run_config=RunConfig(
            callbacks=[
                WandbLoggerCallback(
                    api_key_file="~/.wandb_api_key", project="gnn_tracking"
                ),
            ]
        ),
    )
    tuner.fit()


if __name__ == "__main__":
    main()