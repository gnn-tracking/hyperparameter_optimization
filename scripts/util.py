from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gnn_tracking
import ray
import sklearn.model_selection
from gnn_tracking.graph_construction.graph_builder import GraphBuilder
from gnn_tracking.utils.log import logger
from gnn_tracking.utils.versioning import get_commit_hash
from ray.util.joblib import register_ray
from torch_geometric.loader import DataLoader


@dataclass
class ServerConfig:
    """Config values for server that we run on"""

    #: Total number of GPUs available per node
    gpus: int = 0
    #: Total number of cpus available per node
    cpus: int = 1
    #: Max batches that we can load into the GPU RAM
    max_batches: int = 1

    @property
    def cpus_per_gpu(self) -> int:
        return self.cpus // self.gpus


della = ServerConfig(gpus=4, cpus=48, max_batches=20)
server = della


def get_graphs(test=False) -> dict[str, list]:
    logger.info("Loading data to cpu memory")
    graph_builder = GraphBuilder(
        str(Path("~/data/gnn_tracking/point_clouds").expanduser()),
        str(Path("~/data/gnn_tracking/graphs").expanduser()),
        redo=False,
    )
    n_graphs = 100 if test else None
    logger.debug("Loading %s graphs", n_graphs)
    graph_builder.process(stop=n_graphs)

    # partition graphs into train, test, val splits
    graphs = graph_builder.data_list
    _train_graphs, test_graphs = sklearn.model_selection.train_test_split(
        graphs, test_size=0.2
    )
    train_graphs, val_graphs = sklearn.model_selection.train_test_split(
        _train_graphs, test_size=0.15
    )
    return {
        "train": train_graphs,
        "val": val_graphs,
        "test": test_graphs,
    }


def get_loaders(graph_dct: dict[str, list], test=False) -> dict[str, DataLoader]:
    # build graph loaders
    params = {
        "batch_size": server.max_batches if not test else 1,
        "num_workers": server.cpus_per_gpu if not test else 1,
    }
    loaders = {
        "train": DataLoader(list(graph_dct["train"]), **params, shuffle=True),
        "test": DataLoader(list(graph_dct["test"]), **params),
        "val": DataLoader(list(graph_dct["val"]), **params),
    }
    return loaders


def suggest_if_not_fixed(f, key, config, *args, **kwargs):
    """Suggest if not fixed"""
    if key not in config:
        f(key, *args, **kwargs)


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        config = json.load(f)
    return config


def get_fixed_config(*, test=False):
    return {
        "test": test,
        "max_batches": 1 if test else None,
        "gnn_tracking_hash": get_commit_hash(gnn_tracking),
        "gnn_tracking_experiments_hash": get_commit_hash(Path(__file__).parent),
    }


def run_wandb_offline():
    logger.warning(
        "Setting wandb mode to offline because we assume you don't have internet"
        " on a GPU node."
    )
    os.environ["WANDB_MODE"] = "offline"


def maybe_run_distributed():
    if "redis_password" in os.environ:
        # We're running distributed
        ray.init(address="auto", _redis_password=os.environ["redis_password"])
        register_ray()
