from __future__ import annotations

import functools
import http.client as httplib
import json
import os
import pprint
from dataclasses import dataclass
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any

import click
import gnn_tracking
import ray
import sklearn.model_selection
from gnn_tracking.graph_construction.graph_builder import GraphBuilder
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
from gnn_tracking.utils.versioning import get_commit_hash
from ray import tune
from ray.util.joblib import register_ray
from torch.optim import Adam
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


def get_loaders(
    graph_dct: dict[str, list], batch_size=1, test=False
) -> dict[str, DataLoader]:
    # build graph loaders
    params = {
        "batch_size": batch_size,
        "num_workers": server.cpus_per_gpu if not test else 1,
    }
    logger.debug("Parameters for data loaders: %s", params)
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


def read_json(path: PathLike | str) -> dict[str, Any]:
    with Path(path).open() as f:
        config = json.load(f)
    return config


def get_fixed_config(*, test=False):
    return {
        "test": test,
        "max_batches": 1 if test else None,
        "gnn_tracking_hash": get_commit_hash(gnn_tracking),
        "gnn_tracking_experiments_hash": get_commit_hash(Path(__file__).parent),
    }


def have_internet():
    conn = httplib.HTTPSConnection("8.8.8.8", timeout=5)
    try:
        conn.request("HEAD", "/")
        return True
    except Exception:
        return False
    finally:
        conn.close()


def maybe_run_wandb_offline():
    if not have_internet():
        logger.warning("Setting wandb mode to offline because you do not have internet")
        os.environ["WANDB_MODE"] = "offline"
    logger.debug("You seem to have internet, so directly syncing with wandb")


def maybe_run_distributed():
    if "redis_password" in os.environ:
        # We're running distributed
        ray.init(address="auto", _redis_password=os.environ["redis_password"])
        register_ray()


def get_points_to_evaluate(
    paths: None | list[str | PathLike] = None,
) -> list[dict[str, Any]]:
    """Read json files and return a list of dicts"""
    points_to_evaluate: list[dict[str, Any]] = []
    paths = paths or []
    for path in paths:
        obj = read_json(path)
        if isinstance(obj, list):
            points_to_evaluate.extend(obj)
        elif isinstance(obj, dict):
            points_to_evaluate.append(obj)
        else:
            raise ValueError("Decoding of json file failed")
    if points_to_evaluate:
        logger.info("Enqueued trials:\n%s", pprint.pformat(points_to_evaluate))
    return points_to_evaluate


def faster_dbscan_scan(*args, n_epoch=0, n_trials=100, **kwargs):
    """Skip scanning every second trial."""
    if n_epoch % 2 == 1 and n_epoch >= 4:
        logger.debug("Not reevaluating scanning of DBSCAN in epoch %d", n_epoch)
        n_trials = 1
    return dbscan_scan(*args, n_trials=n_trials, **kwargs)


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
        self.post_setup_hook()
        self.trainer.pt_thlds = [1.5]

    def post_setup_hook(self):
        pass

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
        return PotentialLoss(
            q_min=self.tc.get("q_min", 0.01),
            attr_pt_thld=self.tc.get("attr_pt_thld", 0.9),
        )

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
                faster_dbscan_scan,
                n_trials=100 if not self.tc.get("test", False) else 1,
                n_jobs=server.cpus_per_gpu if not self.tc.get("test", False) else 1,
            )
        }

    def get_lr_scheduler(self):
        return None

    def get_optimizer(self):
        return Adam

    def get_loss_weights(self):
        return subdict_with_prefix_stripped(self.tc, "lw_")

    def get_trainer(self) -> TCNTrainer:
        test = self.tc.get("test", False)
        trainer = TCNTrainer(
            model=self.get_model(),
            loaders=get_loaders(get_graphs(test=test), test=test),
            loss_functions=self.get_loss_functions(),
            loss_weights=self.get_loss_weights(),
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


test_option = click.option(
    "--test",
    help="As-fast-as-possible run to test the setup",
    is_flag=True,
    default=False,
)
gpu_option = click.option(
    "--gpu",
    help="Run on a GPU. This will also assume that you are on a batch node without "
    "internet access and will set wandb mode to offline.",
    is_flag=True,
    default=False,
)
enqueue_option = click.option(
    "--enqueue",
    help="Read trials from this file and enqueue them",
    multiple=True,
)


def wandb_options(f):
    @click.option("--tags", multiple=True, help="Tags for wandb")
    @click.option(
        "--group",
        help="Wandb group name",
    )
    @click.option("--note", help="Wandb note")
    @functools.wraps(f)
    def wrapper_common_options(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_common_options
