from __future__ import annotations

import pprint
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from gnn_tracking.metrics.cluster_metrics import common_metrics
from gnn_tracking.metrics.losses import (
    BackgroundLoss,
    EdgeWeightFocalLoss,
    PotentialLoss,
)
from gnn_tracking.models.track_condensation_networks import GraphTCN
from gnn_tracking.postprocessing.clusterscanner import ClusterScanResult
from gnn_tracking.postprocessing.dbscanscanner import DBSCANHyperParamScanner
from gnn_tracking.training.tcn_trainer import TCNTrainer
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped
from gnn_tracking.utils.log import logger
from gnn_tracking.utils.seeds import fix_seeds
from ray import tune
from torch.optim import SGD, Adam, lr_scheduler

from gnn_tracking_hpo.load import get_graphs, get_loaders


def fixed_dbscan_scan(
    graphs: np.ndarray,
    truth: np.ndarray,
    sectors: np.ndarray,
    pts: np.ndarray,
    reconstructable: np.ndarray,
    *,
    guide="trk.double_majority_pt1.5",
    epoch=None,
    start_params: dict[str, Any] | None = None,
) -> ClusterScanResult:
    """Convenience function for not scanning for DBSCAN hyperparameters at all."""
    if start_params is None:
        start_params = {
            "eps": 0.95,
            "min_samples": 1,
        }
    dbss = DBSCANHyperParamScanner(
        graphs=graphs,
        truth=truth,
        sectors=sectors,
        pts=pts,
        reconstructable=reconstructable,
        guide=guide,
        metrics=common_metrics,
    )
    return dbss.scan(
        n_jobs=1,
        n_trials=1,
        start_params=start_params,
    )


def reduced_dbscan_scan(
    graphs: np.ndarray,
    truth: np.ndarray,
    sectors: np.ndarray,
    pts: np.ndarray,
    reconstructable: np.ndarray,
    *,
    guide="trk.double_majority_pt1.5",
    epoch=None,
    start_params: dict[str, Any] | None = None,
) -> ClusterScanResult:
    """Convenience function for scanning DBSCAN hyperparameters with trial count
    that depends on the epoch (using many trials early on, then alternating between
    fixed and low samples in later epochs).
    """
    dbss = DBSCANHyperParamScanner(
        graphs=graphs,
        truth=truth,
        sectors=sectors,
        pts=pts,
        reconstructable=reconstructable,
        guide=guide,
        metrics=common_metrics,
        min_samples_range=(1, 1),
        eps_range=(0.95, 1.0),
    )
    if epoch < 8:
        n_trials = 12
    elif epoch % 4 == 0:
        n_trials = 12
    else:
        n_trials = 1
    return dbss.scan(
        n_jobs=min(12, n_trials),  # todo: make flexible
        n_trials=n_trials,
        start_params=start_params,
    )


def suggest_default_values(
    config: dict[str, Any],
    trial: None | optuna.Trial = None,
    perfect_ec=False,
    only_ec=False,
) -> None:
    """Set all config values, so that everything gets recorded in the database, even
    if we do not change anything.

    Args:
        config:
        trial:
        perfect_ec: Perfect (truth-based) edge classifier: Do not set parameters for
            edge classifier
        only_ec: We're only training the edge classifier, do not set any other
            parameters
    """
    if perfect_ec and only_ec:
        raise ValueError("perfect_ec and only_ec are mutually exclusive")

    c = {**config, **(trial.params if trial is not None else {})}

    def d(k, v):
        if trial is not None and k in trial.params:
            return
        if k in config:
            return
        config[k] = v
        c[k] = v

    n_graphs_default = 3200
    d("n_graphs_train", int(0.32 * n_graphs_default))
    d("n_graphs_test", int(0.2 * n_graphs_default))
    d("n_graphs_val", int(0.12 * n_graphs_default))
    assert (
        c["n_graphs_train"] + c["n_graphs_test"] + c["n_graphs_val"] <= n_graphs_default
    )

    d("training_pt_thld", 0.0)
    d("training_without_noise", False)

    d("sector", None)

    if perfect_ec:
        d("m_ec_tpr", 1.0)
        d("m_ec_tnr", 1.0)

    # Loss function parameters
    if not only_ec:
        d("q_min", 0.01)
        d("attr_pt_thld", 0.9)
        d("sb", 0.1)
    if not perfect_ec:
        d("focal_alpha", 0.25)
        d("focal_gamma", 2.0)

    # Optimizers
    d("lr", 5e-4)
    d("optimizer", "adam")
    if c["optimizer"] == "sgd":
        d("sgd_momentum", 0.0)
        d("sgd_weight_decay", 0.0)
        d("sgd_nesterov", False)
        d("sgd_dampening", 0.0)
    d("scheduler", None)
    if c["scheduler"] is not None:
        if c["optimizer"] == "adam":
            raise ValueError("Don't use lr scheduler with Adam.")

    # Schedulers
    if c["scheduler"] is None:
        pass
    elif c["scheduler"] == "steplr":
        d("steplr_step_size", 10)
        d("steplr_gamma", 0.1)
    elif c["scheduler"] == "exponentiallr":
        d("exponentiallr_gamma", 0.9)
    else:
        raise ValueError(f"Unknown scheduler: {c['scheduler']}")

    # Model parameters
    d("m_h_dim", 5)
    d("m_e_dim", 4)
    if not only_ec:
        d("m_h_outdim", 2)
    d("m_hidden_dim", 40)
    if not perfect_ec:
        d("m_L_ec", 3)
        d("m_alpha_ec", 0.5)
    if not only_ec:
        d("m_L_hc", 3)
        d("m_alpha_hc", 0.5)
    if not perfect_ec and not only_ec:
        d("m_feed_edge_weights", False)


class TCNTrainable(tune.Trainable):
    """A wrapper around `TCNTrainer` for use with Ray Tune."""

    # Do not add blank self.tc or self.trainer to __init__, because it will be called
    # after setup when setting ``reuse_actor == True`` and overwriting your values
    # from set
    def setup(self, config: dict[str, Any]):
        logger.debug("Got config\n%s", pprint.pformat(config))
        self.tc = config
        fix_seeds()
        self.trainer = self.get_trainer()
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
        return {"dbscan": reduced_dbscan_scan}

    def get_lr_scheduler(self):
        if not self.tc["scheduler"]:
            return None
        elif self.tc["scheduler"] == "steplr":
            return partial(
                lr_scheduler.StepLR, **subdict_with_prefix_stripped(self.tc, "steplr_")
            )
        elif self.tc["scheduler"] == "exponentiallr":
            return partial(
                lr_scheduler.ExponentialLR,
                **subdict_with_prefix_stripped(self.tc, "exponentiallr_"),
            )
        else:
            raise ValueError(f"Unknown scheduler {self.tc['scheduler']}")

    def get_optimizer(self):
        if self.tc["optimizer"] == "adam":
            return Adam
        elif self.tc["optimizer"] == "sgd":
            return partial(SGD, **subdict_with_prefix_stripped(self.tc, "sgd_"))
        else:
            raise ValueError(f"Unknown optimizer {self.tc['optimizer']}")

    def get_loss_weights(self):
        return subdict_with_prefix_stripped(self.tc, "lw_")

    def get_loaders(self):
        n_graphs = (
            self.tc["n_graphs_train"]
            + self.tc["n_graphs_test"]
            + self.tc["n_graphs_val"]
        )
        test_frac = self.tc["n_graphs_test"] / n_graphs
        val_frac = self.tc["n_graphs_val"] / n_graphs
        return get_loaders(
            get_graphs(
                n_graphs=n_graphs,
                test_frac=test_frac,
                val_frac=val_frac,
                sector=self.tc["sector"],
            ),
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
        trainer.training_without_noise = self.tc["training_without_noise"]
        trainer.training_pt_thld = self.tc["training_pt_thld"]
        return trainer

    def step(self):
        return self.trainer.step(max_batches=None if not self.tc["test"] else 1)

    def save_checkpoint(
        self,
        checkpoint_dir,
    ):
        return self.trainer.save_checkpoint(
            Path(checkpoint_dir) / "checkpoint.pt",
        )

    def load_checkpoint(self, checkpoint_path, **kwargs):
        self.trainer.load_checkpoint(checkpoint_path, **kwargs)
