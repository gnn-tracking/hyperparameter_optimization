from __future__ import annotations

import os
from abc import ABC
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import tabulate
from gnn_tracking.metrics.cluster_metrics import common_metrics
from gnn_tracking.metrics.losses import (
    BackgroundLoss,
    EdgeWeightFocalLoss,
    HaughtyFocalLoss,
    PotentialLoss,
)
from gnn_tracking.models.track_condensation_networks import GraphTCN
from gnn_tracking.postprocessing.clusterscanner import ClusterScanResult
from gnn_tracking.postprocessing.dbscanscanner import DBSCANHyperParamScanner
from gnn_tracking.training.tcn_trainer import TCNTrainer, TrainingTruthCutConfig
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped
from gnn_tracking.utils.seeds import fix_seeds
from ray import tune
from torch import nn
from torch.optim import SGD, Adam, lr_scheduler

from gnn_tracking_hpo.load import get_graphs_separate, get_graphs_split, get_loaders
from gnn_tracking_hpo.slurmcontrol import SlurmControl, get_slurm_job_id
from gnn_tracking_hpo.util.log import logger
from gnn_tracking_hpo.util.paths import find_checkpoint, get_config


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
    graphs: list[np.ndarray],
    truth: list[np.ndarray],
    sectors: list[np.ndarray],
    pts: list[np.ndarray],
    reconstructable: list[np.ndarray],
    *,
    guide="trk.double_majority_pt0.9",
    epoch=None,
    start_params: dict[str, Any] | None = None,
    node_mask: list[np.ndarray] | None = None,
) -> ClusterScanResult:
    """Convenience function for scanning DBSCAN hyperparameters with trial count
    that depends on the epoch (using many trials early on, then alternating between
    fixed and low samples in later epochs).
    """
    version_dependent_kwargs = {}
    if node_mask is not None:
        logger.warning("Running on a gnn_tracking version with post-EC node pruning.")
        version_dependent_kwargs["node_mask"] = node_mask
    dbss = DBSCANHyperParamScanner(
        data=graphs,
        truth=truth,
        sectors=sectors,
        pts=pts,
        reconstructable=reconstructable,
        guide=guide,
        metrics=common_metrics,
        min_samples_range=(1, 1),
        eps_range=(0.95, 1.0),
        **version_dependent_kwargs,
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
    ec="default",
    hc="default",
) -> None:
    """Set all config values, so that everything gets recorded in the database, even
    if we do not change anything.

    Args:
        config: Gets modified in place
        trial:
        ec: One of "default" (train), "perfect" (perfect ec), "fixed"
        hc: One of "default" (train), "none" (no hc)
    """
    if ec not in ["default", "perfect", "fixed"]:
        raise ValueError(f"Invalid ec: {ec}")
    if hc not in ["default", "none"]:
        raise ValueError(f"Invalid hc: {hc}")

    c = {**config, **(trial.params if trial is not None else {})}

    def d(k, v):
        if trial is not None and k in trial.params:
            return
        if k in config:
            return
        config[k] = v
        c[k] = v

    d("node_indim", 7)
    d("edge_indim", 4)

    if test_data_dir := os.environ.get("TEST_TRAIN_DATA_DIR"):
        d("train_data_dir", test_data_dir)
    else:
        d("train_data_dir", ["/tigress/jdezoort/object_condensation/graphs"])

    d("val_data_dir", None)

    if c["test"]:
        config["n_graphs_train"] = 1
        config["n_graphs_val"] = 1
    else:
        n_graphs_default = 5_000
        n_graphs_val = c.get("n_graphs_val", min(400, int(0.1 * n_graphs_default)))
        d("n_graphs_train", n_graphs_default - 1 - n_graphs_val)
        d("n_graphs_val", n_graphs_val)

    d("training_pt_thld", 0.0)
    d("training_without_noise", False)
    d("training_without_non_reconstructable", False)
    d("ec_pt_thld", 0.0)

    d("sector", None)
    d("batch_size", 1)
    d("_val_batch_size", 1)

    if hc != "none":
        d("repulsive_radius_threshold", 10.0)

    if ec == "perfect":
        d("m_ec_tpr", 1.0)
        d("m_ec_tnr", 1.0)
    elif ec == "fixed" and hc != "none":
        d("m_ec_threshold", 0.5)

    # Loss function parameters
    if hc != "none":
        d("q_min", 0.01)
        d("attr_pt_thld", 0.9)
        d("sb", 0.1)

    d("ec_loss", "focal")
    if ec in ["default"] and c["ec_loss"] in ["focal", "haughty_focal"]:
        d("focal_alpha", 0.25)
        d("focal_gamma", 2.0)

    # Optimizers
    d("lr", 5e-4)
    d("optimizer", "adam")
    if c["optimizer"] == "adam":
        d("adam_beta1", 0.9)
        d("adam_beta2", 0.999)
        d("adam_eps", 1e-8)
        d("adam_weight_decay", 0.0)
        d("adam_amsgrad", False)
    elif c["optimizer"] == "sgd":
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
    # d("m_h_dim", 5)
    # d("m_e_dim", 4)
    if hc != "none":
        d("m_h_outdim", 2)
    d("m_hidden_dim", 40)
    if ec in ["default"]:
        d("m_L_ec", 3)
        # d("m_alpha_ec", 0.5)
    if hc != "none":
        d("m_L_hc", 3)
        d("m_alpha_hc", 0.5)
    if ec in ["default"] and hc != "none":
        d("m_feed_edge_weights", False)


class HPOTrainable(tune.Trainable, ABC):
    """Add additional 'restore' capabilities to tune.Trainable."""

    @classmethod
    def reinstate(
        cls,
        project: str,
        hash: str,
        *,
        epoch=-1,
        n_graphs: int | None = None,
        config_override: dict[str, Any] | None = None,
    ):
        """Load config from wandb and restore on-disk checkpoint.

        This is different from `tune.Trainable.restore` which is called from
        an instance, i.e., already needs to be initialized with a config.

        Args:
            project: The wandb project name that should also correspond to the local
                folder with the checkpoints
            hash: The wandb run hash.
            epoch: The epoch to restore. If -1, restore the last epoch. If 0, do not
                restore any checkpoint.
            n_graphs: Total number of samples to load. ``None`` uses the values from
                training.
            config_override: Update the config with these values.
        """
        config = legacy_config_compatibility(get_config(project, hash))
        if n_graphs is not None:
            previous_n_graphs = config["n_graphs_train"] + config["n_graphs_val"]
            if previous_n_graphs == 0:
                raise ValueError(
                    "Cannot rescale n_graphs when previous_n_graphs == 0. Use "
                    "`config_override` to set graph numbers manually."
                )
            for key in ["n_graphs_train", "n_graphs_val"]:
                config[key] = int(config[key] * n_graphs / previous_n_graphs)
        if config_override is not None:
            config.update(config_override)
        trainable = cls(config)
        if epoch != 0:
            trainable.load_checkpoint(str(find_checkpoint(project, hash, epoch)))
        return trainable


def legacy_config_compatibility(config: dict[str, Any]) -> dict[str, Any]:
    """Preprocess config, for example to deal with legacy configs."""
    rename_keys = {
        "m_alpha_ec_node": "m_alpha",
        "m_use_intermediate_encodings": "m_use_intermediate_edge_embeddings",
        "m_feed_node_attributes": "m_use_node_embedding",
    }
    remove_keys = ["m_alpha_ec_edge"]
    for old, new in rename_keys.items():
        if old in config:
            logger.warning("Renaming key %s to %s", old, new)
            config[new] = config.pop(old)
    for key in remove_keys:
        if key in config:
            logger.warning("Removing key %s", key)
            del config[key]
    return config


class TCNTrainable(HPOTrainable):
    """A wrapper around `TCNTrainer` for use with Ray Tune."""

    # This is set explicitly by the Dispatcher class
    dispatcher_id: int = 0

    # Do not add blank self.tc or self.trainer to __init__, because it will be called
    # after setup when setting ``reuse_actor == True`` and overwriting your values
    # from set
    def setup(self, config: dict[str, Any]):
        config = legacy_config_compatibility(config)
        if sji := get_slurm_job_id():
            logger.info("I'm running on a node with job ID=%s", sji)
        else:
            logger.info("I appear to be running locally.")
        if self.dispatcher_id == 0:
            logger.warning(
                "Dispatcher ID was not set. This should be set by the dispatcher "
                "as a class attribute to the trainable."
            )
        logger.info("The ID of my dispatcher is %d", self.dispatcher_id)
        SlurmControl()(dispatcher_id=str(self.dispatcher_id))
        config_table = tabulate.tabulate(
            sorted([(k, str(v)[:40]) for k, v in config.items()]),
            tablefmt="simple_outline",
        )
        logger.debug("Got config\n%s", config_table)
        self.tc = config
        fix_seeds()
        self.trainer = self.get_trainer()

    def get_model(self) -> nn.Module:
        return GraphTCN(
            node_indim=self.tc.get("node_indim", 7),
            edge_indim=self.tc.get("edge_indim", 4),
            **subdict_with_prefix_stripped(self.tc, "m_"),
        )

    def get_edge_loss_function(self):
        if self.tc["ec_loss"] == "focal":
            return EdgeWeightFocalLoss(
                pt_thld=self.tc["ec_pt_thld"],
                alpha=self.tc["focal_alpha"],
                gamma=self.tc["focal_gamma"],
            )
        elif self.tc["ec_loss"] == "haughty_focal":
            return HaughtyFocalLoss(
                pt_thld=self.tc["ec_pt_thld"],
                alpha=self.tc["focal_alpha"],
                gamma=self.tc["focal_gamma"],
            )
        else:
            raise ValueError(f"Unknown edge loss: {self.tc['edge_loss']}")

    def get_potential_loss_function(self):
        return PotentialLoss(
            q_min=self.tc["q_min"],
            attr_pt_thld=self.tc["attr_pt_thld"],
            radius_threshold=self.tc["repulsive_radius_threshold"],
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
            return partial(
                Adam,
                betas=(self.tc["adam_beta1"], self.tc["adam_beta2"]),
                eps=self.tc["adam_eps"],
                weight_decay=self.tc["adam_weight_decay"],
                amsgrad=self.tc["adam_amsgrad"],
            )
        elif self.tc["optimizer"] == "sgd":
            return partial(SGD, **subdict_with_prefix_stripped(self.tc, "sgd_"))
        else:
            raise ValueError(f"Unknown optimizer {self.tc['optimizer']}")

    def get_loss_weights(self):
        return subdict_with_prefix_stripped(self.tc, "lw_")

    def get_loaders(self):
        logger.debug("Getting loaders")
        if self.tc.get("_no_data", False):
            logger.debug("Not adding loaders to trainer")
            return {}

        if self.tc["val_data_dir"]:
            graph_dict = get_graphs_separate(
                train_size=self.tc["n_graphs_train"],
                val_size=self.tc["n_graphs_val"],
                sector=self.tc["sector"],
                test=self.tc["test"],
                train_dirs=self.tc["train_data_dir"],
                val_dirs=self.tc["val_data_dir"],
            )
        else:
            graph_dict = get_graphs_split(
                train_size=self.tc["n_graphs_train"],
                val_size=self.tc["n_graphs_val"],
                sector=self.tc["sector"],
                test=self.tc["test"],
                input_dirs=self.tc["train_data_dir"],
            )
        return get_loaders(
            graph_dict,
            test=self.tc["test"],
            batch_size=self.tc["batch_size"],
            val_batch_size=self.tc["_val_batch_size"],
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
        trainer.training_truth_cuts = TrainingTruthCutConfig(
            without_noise=self.tc["training_without_noise"],
            pt_thld=self.tc["training_pt_thld"],
            without_non_reconstructable=self.tc["training_without_non_reconstructable"],
        )

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
        logger.debug("Loading checkpoint from %s", checkpoint_path)
        self.trainer.load_checkpoint(checkpoint_path, **kwargs)
