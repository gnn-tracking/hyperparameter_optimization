from __future__ import annotations

import logging
from abc import ABC
from functools import partial
from pathlib import Path
from typing import Any

import tabulate
from gnn_tracking.metrics.losses import (
    BackgroundLoss,
    EdgeWeightFocalLoss,
    HaughtyFocalLoss,
    PotentialLoss,
)
from gnn_tracking.models.edge_classifier import ECForGraphTCN
from gnn_tracking.models.track_condensation_networks import (
    GraphTCN,
    PreTrainedECGraphTCN,
)
from gnn_tracking.training.tcn_trainer import TCNTrainer
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped
from gnn_tracking.utils.seeds import fix_seeds
from ray import tune
from torch import nn
from torch.optim import SGD, Adam, lr_scheduler

from gnn_tracking_hpo.cluster_scans import reduced_dbscan_scan
from gnn_tracking_hpo.load import get_graphs_separate, get_graphs_split, get_loaders
from gnn_tracking_hpo.restore import restore_model
from gnn_tracking_hpo.slurmcontrol import SlurmControl, get_slurm_job_id
from gnn_tracking_hpo.util.log import logger
from gnn_tracking_hpo.util.paths import find_checkpoint, get_config


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
    remove_keys = ["m_alpha_ec_edge", "adam_epsilon"]
    for old, new in rename_keys.items():
        if old in config:
            logger.warning("Renaming key %s to %s", old, new)
            config[new] = config.pop(old)
    for key in remove_keys:
        if key in config:
            logger.warning("Removing key %s", key)
            del config[key]
    return config


class DefaultTrainable(HPOTrainable):
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

    def get_edge_loss_function(self) -> tuple[nn.Module, float]:
        if self.tc["ec_loss"] == "focal":
            lf = EdgeWeightFocalLoss(
                pt_thld=self.tc["ec_pt_thld"],
                alpha=self.tc["focal_alpha"],
                gamma=self.tc["focal_gamma"],
            )
        elif self.tc["ec_loss"] == "haughty_focal":
            lf = HaughtyFocalLoss(
                pt_thld=self.tc["ec_pt_thld"],
                alpha=self.tc["focal_alpha"],
                gamma=self.tc["focal_gamma"],
            )
        else:
            raise ValueError(f"Unknown edge loss: {self.tc['edge_loss']}")
        return lf, self.tc["lw_edge"]

    def get_potential_loss_function(self) -> tuple[nn.Module, dict[str, float]]:
        lf = PotentialLoss(
            q_min=self.tc["q_min"],
            attr_pt_thld=self.tc["attr_pt_thld"],
            radius_threshold=self.tc["repulsive_radius_threshold"],
        )
        lw = {
            "attractive": self.tc["lw_potential_attractive"],
            "repulsive": self.tc["lw_potential_repulsive"],
        }
        return lf, lw

    def get_background_loss_function(self) -> tuple[nn.Module, float]:
        return BackgroundLoss(sb=self.tc["sb"]), self.tc["lw_background"]

    def get_loss_functions(self) -> dict[str, tuple[nn.Module, Any]]:
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
        elif self.tc["scheduler"] == "cycliclr":
            return partial(
                lr_scheduler.CyclicLR,
                base_lr=self.tc["lr"],
                max_lr=self.tc["max_lr"],
                mode=self.tc["cycliclr_mode"],
                step_size_up=self.tc["cycliclr_step_size_up"],
                step_size_down=self.tc["cycliclr_step_size_down"],
            )
        elif self.tc["scheduler"] == "cosineannealinglr":
            return partial(
                lr_scheduler.CosineAnnealingLR,
                **subdict_with_prefix_stripped(self.tc, "cosineannealinglr_"),
            )
        elif self.tc["scheduler"] == "linearlr":
            return partial(
                lr_scheduler.LinearLR,
                **subdict_with_prefix_stripped(self.tc, "linearlr_"),
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
            lr=self.tc["lr"],
            lr_scheduler=self.get_lr_scheduler(),
            cluster_functions=self.get_cluster_functions(),  # type: ignore
            optimizer=self.get_optimizer(),
        )
        trainer.logger.setLevel(logging.DEBUG)
        trainer.max_batches_for_clustering = 100 if not test else 10
        if self.tc["scheduler"] == "cycliclr":
            logger.info("Setting lr_scheduler_step to batch")
            trainer.lr_scheduler_step = "batch"

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


class TCNTrainable(DefaultTrainable):
    def get_loss_functions(self) -> dict[str, tuple[nn.Module, Any]]:
        loss_functions = super().get_loss_functions()
        loss_functions.pop("edge")
        return loss_functions

    def get_trainer(self) -> TCNTrainer:
        trainer = super().get_trainer()
        trainer.ec_threshold = self.tc["m_ec_threshold"]
        return trainer

    @property
    def _is_continued_run(self) -> bool:
        """We're restoring a model from a previous run and continuing."""
        return "tc_project" in self.tc

    def _get_new_model(self) -> nn.Module:
        ec = restore_model(
            ECTrainable,
            self.tc["ec_project"],
            self.tc["ec_hash"],
            self.tc["ec_epoch"],
            freeze=self.tc["ec_freeze"],
        )
        return PreTrainedECGraphTCN(
            ec,
            node_indim=7,
            edge_indim=4,
            **subdict_with_prefix_stripped(self.tc, "m_"),
        )

    def _get_restored_model(self) -> nn.Module:
        """Load previously trained model to continue"""
        return restore_model(
            TCNTrainable,
            tune_dir=self.tc["tc_project"],
            run_hash=self.tc["tc_hash"],
            epoch=self.tc.get("tc_epoch", -1),
            freeze=False,
            config_update={
                "ec_freeze": self.tc["ec_freeze"],
            },
        )

    def get_model(self) -> nn.Module:
        if self._is_continued_run:
            return self._get_restored_model()
        return self._get_new_model()


class ECTrainable(DefaultTrainable):
    def get_loss_functions(self) -> dict[str, Any]:
        return {
            "edge": self.get_edge_loss_function(),
        }

    def get_cluster_functions(self) -> dict[str, Any]:
        return {}

    def get_trainer(self) -> TCNTrainer:
        trainer = super().get_trainer()
        trainer.ec_eval_pt_thlds = [0.0, 0.5, 0.9, 1.2, 1.5]
        return trainer

    @property
    def _is_continued_run(self) -> bool:
        """We're restoring a model from a previous run and continuing."""
        return "ec_project" in self.tc

    def _get_new_model(self) -> nn.Module:
        """New model to be trained (rather than continuing training a pretrained
        one).
        """
        return ECForGraphTCN(
            node_indim=7, edge_indim=4, **subdict_with_prefix_stripped(self.tc, "m_")
        )

    def _get_restored_model(self) -> nn.Module:
        """Load previously trained model to continue"""
        return restore_model(
            ECTrainable,
            tune_dir=self.tc["ec_project"],
            run_hash=self.tc["ec_hash"],
            epoch=self.tc.get("ec_epoch", -1),
            freeze=False,
        )

    def get_model(self) -> nn.Module:
        if self._is_continued_run:
            return self._get_restored_model()
        return self._get_new_model()
