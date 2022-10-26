from __future__ import annotations

import pprint
from functools import partial
from pathlib import Path
from typing import Any

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
from torch.optim import SGD, Adam, lr_scheduler

from gte.config import server
from gte.load import get_graphs, get_loaders


def faster_dbscan_scan(*args, n_epoch=0, n_trials=100, **kwargs):
    """Skip scanning every second trial."""
    if n_epoch % 2 == 1 and n_epoch >= 4:
        logger.debug("Not reevaluating scanning of DBSCAN in epoch %d", n_epoch)
        n_trials = 1
    return dbscan_scan(*args, n_trials=n_trials, **kwargs)


def set_config_default_values(config: dict[str, Any]) -> dict[str, Any]:
    """Set all config values, so that everything gets recorded in the database, even
    if we do not change anything.
    """

    def d(k, v):
        config.setdefault(k, v)

    # Loss function parameters
    d("q_min", 0.01)
    d("attr_pt_thld", 0.9)
    d("focal_alpha", 0.25)
    d("focal_gamma", 2.0)
    d("sb", 0.1)

    # Optimizers
    d("lr", 5e-4)
    d("optimizer", "adam")
    if config["optimizer"] == "sgd":
        d("optim_momentum", 0.9)
        d("optim_weight_decay", 0.0)
        d("optim_nesterov", False)
        d("optim_dampening", 0.0)
    d("scheduler", None)
    if config["scheduler"] is not None:
        if config["optimizer"] == "adadm":
            raise ValueError("Don't use lr scheduler with Adam.")
    elif config["scheduler"] == "steplr":
        d("sched_step_size", 10)
        d("sched_gamma", 0.1)
    elif config["scheduler"] == "exponentiallr":
        d("sched_gamma", 0.9)
    else:
        raise ValueError(f"Unknown scheduler: {config['scheduler']}")

    # Model parameters
    d("m_h_dim", 5)
    d("m_e_dim", 4)
    d("m_h_outdim", 2)
    d("m_hidden_dim", 40)
    d("m_L_ec", 3)
    d("m_L_hc", 3)
    d("m_alpha_ec", 0.5)
    d("m_alpha_hc", 0.5)
    d("m_feed_edge_weights", False)

    return config


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

    def get_model(self) -> GraphTCN:
        return GraphTCN(
            node_indim=6, edge_indim=4, **subdict_with_prefix_stripped(self.tc, "m_")
        )

    def get_edge_loss_function(self):
        return EdgeWeightFocalLoss(
            alpha=self.tc["focal_alpha"],
            gamma=self.tc["focal_gamma"],
        )

    def get_potential_loss_function(self):
        return PotentialLoss(
            q_min=self.tc["q_min"],
            attr_pt_thld=self.tc["attr_pt_thld"],
        )

    def get_background_loss_function(self):
        return BackgroundLoss(sb=self.tc["sb"])

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
        if not self.tc["scheduler"]:
            return None
        elif self.tc["scheduler"] == "steplr":
            return partial(
                lr_scheduler.StepLR, **subdict_with_prefix_stripped(self.tc, "sched_")
            )
        elif self.tc["scheduler"] == "exponentiallr":
            return partial(
                lr_scheduler.ExponentialLR,
                **subdict_with_prefix_stripped(self.tc, "sched_"),
            )
        else:
            raise ValueError(f"Unknown scheduler {self.tc['scheduler']}")

    def get_optimizer(self):
        if self.tc["optimizer"] == "adam":
            return Adam
        elif self.tc["optimizer"] == "sgd":
            return partial(SGD, **subdict_with_prefix_stripped(self.tc, "optim_"))
        else:
            raise ValueError(f"Unknown optimizer {self.tc['optimizer']}")

    def get_loss_weights(self):
        return subdict_with_prefix_stripped(self.tc, "lw_")

    def get_loaders(self):
        return get_loaders(
            get_graphs(test=self.tc["test"]),
            test=self.tc["test"],
            batch_size=self.tc["batch_size"],
        )

    def get_trainer(self) -> TCNTrainer:
        test = self.tc.get("test", False)
        trainer = TCNTrainer(
            model=self.get_model(),
            loaders=self.get_loaders(),
            loss_functions=self.get_loss_functions(),
            loss_weights=self.get_loss_weights(),
            lr=self.tc["lr"],
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
