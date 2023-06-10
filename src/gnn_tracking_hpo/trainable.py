from __future__ import annotations

import collections
import logging
import math
from abc import ABC
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import tabulate
import torch
from gnn_tracking.graph_construction.radius_scanner import RadiusScanner
from gnn_tracking.metrics.losses import (
    BackgroundLoss,
    EdgeWeightFocalLoss,
    GraphConstructionHingeEmbeddingLoss,
    HaughtyFocalLoss,
    PotentialLoss,
)
from gnn_tracking.models.edge_classifier import ECForGraphTCN
from gnn_tracking.models.graph_construction import GCWithEF, GraphConstructionFCNN
from gnn_tracking.models.mlp import MLP
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
from gnn_tracking_hpo.defaults import legacy_config_compatibility
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
        self.hook_before_trainer_setup()
        self.trainer = self.get_trainer()

    def hook_before_trainer_setup(self):
        pass

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
            max_sample_size=self.tc["max_sample_size"],
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


class PretrainedECTCNTrainable(DefaultTrainable):
    @property
    def _is_continued_run(self) -> bool:
        """We're restoring a model from a previous run and continuing."""
        return "tc_project" in self.tc

    @property
    def _need_edge_loss(self) -> bool:
        return not self.tc.get("ec_freeze", True)

    def hook_before_trainer_setup(self):
        """Ensure that we bring back the EC config before we initialize anything"""
        self._update_config_from_restored_tc()
        self._update_edge_loss_config_from_ec()

    def _update_config_from_restored_tc(self) -> None:
        if not self._is_continued_run:
            return
        tc_config = get_config(project=self.tc["tc_project"], part=self.tc["tc_hash"])
        if not tc_config["ec_freeze"]:
            _ = (
                "Currently cannot bring back pre-trained TCNS with ECs that were"
                " not frozen. "
            )
            raise NotImplementedError(_)
        for key in ["ec_project", "ec_hash", "ec_epoch"]:
            if key in tc_config:
                logger.debug(
                    "Setting %s to %s from restored TC configuration",
                    key,
                    tc_config[key],
                )
                self.tc[key] = tc_config[key]

    def _update_edge_loss_config_from_ec(self) -> None:
        """When we use an unfrozen pretrained EC, make sure the loss function
        configuration stays the same.
        """
        if not self._need_edge_loss:
            return
        ec_config = get_config(project=self.tc["ec_project"], part=self.tc["ec_hash"])
        keys = ["ec_loss", "ec_pt_thld", "focal_alpha", "focal_gamma"]
        for key in keys:
            if key in ec_config:
                logger.debug(
                    "Setting %s to %s from EC configuration", key, ec_config[key]
                )
                self.tc[key] = ec_config[key]

    def get_loss_functions(self) -> dict[str, tuple[nn.Module, Any]]:
        loss_functions = super().get_loss_functions()
        if not self._need_edge_loss:
            loss_functions.pop("edge")
        return loss_functions

    def get_trainer(self) -> TCNTrainer:
        trainer = super().get_trainer()
        trainer.ec_threshold = self.tc["m_ec_threshold"]
        logger.info("Final loss weights %s", trainer.loss_functions)
        return trainer

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
            PretrainedECTCNTrainable,
            tune_dir=self.tc["tc_project"],
            run_hash=self.tc["tc_hash"],
            epoch=self.tc.get("tc_epoch", -1),
            freeze=False,
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


class GCWithECTrainable(DefaultTrainable):
    def get_loss_functions(self) -> dict[str, Any]:
        return {
            "edge": self.get_edge_loss_function(),
        }

    def get_cluster_functions(self) -> dict[str, Any]:
        return {}

    def get_trainer(self) -> TCNTrainer:
        trainer = super().get_trainer()
        trainer.ec_eval_pt_thlds = [0.0, 0.9, 1.5]
        return trainer

    @property
    def _is_continued_run(self) -> bool:
        """We're restoring a model from a previous run and continuing."""
        return "ec_project" in self.tc

    def _get_gc(
        self,
        hash,
        tune_dir,
        epoch=-1,
    ):
        return restore_model(
            GCTrainable,
            tune_dir=tune_dir,
            run_hash=hash,
            epoch=epoch,
            freeze=True,
        )

    def _get_new_model(self) -> nn.Module:
        """New model to be trained (rather than continuing training a pretrained
        one).
        """
        gc = self._get_gc(
            hash=self.tc["gc_hash"],
            tune_dir=self.tc["gc_project"],
            epoch=self.tc["gc_epoch"],
        )
        ec = ECForGraphTCN(
            node_indim=14 + gc.out_dim,
            edge_indim=(14 + gc.out_dim) * 2,
            **subdict_with_prefix_stripped(self.tc, "m_"),
        )
        return GCWithEF(
            ml=gc,
            ef=ec,
            max_radius=self.tc["max_radius"],
            max_num_neighbors=self.tc["max_num_neighbors"],
            use_embedding_features=self.tc["ec_use_embedding_features"],
            ratio_of_false=None,
            build_edge_features=True,
        )

    def _get_restored_model(self) -> nn.Module:
        """Load previously trained model to continue"""
        raise NotImplementedError

    def get_model(self) -> nn.Module:
        if self._is_continued_run:
            return self._get_restored_model()
        return self._get_new_model()


class GCTrainer(TCNTrainer):
    def __init__(
        self,
        *args,
        rs_max_r=5,
        rs_max_edges=8_000_000,
        max_edges_per_node=128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._rs_max_edges = rs_max_edges
        self._max_edges_per_node = max_edges_per_node
        self._rs_max_r = rs_max_r

    @torch.no_grad()
    def test_step(self, val=True, max_batches: int | None = None) -> dict[str, float]:
        self.model.eval()
        loader = self.val_loader if val else self.test_loader
        assert loader is not None
        mos = []
        batch_metrics = collections.defaultdict(list)
        for _batch_idx, data in enumerate(loader):
            if max_batches and _batch_idx > max_batches:
                break
            data = data.to(self.device)  # noqa: PLW2901
            model_output = self.evaluate_model(
                data,
                mask_pids_reco=False,
            )
            batch_loss, these_batch_losses, loss_weights = self.get_batch_losses(
                model_output
            )

            batch_metrics["total"].append(batch_loss.item())
            for key, value in these_batch_losses.items():
                batch_metrics[key].append(value.item())
                batch_metrics[f"{key}_weighted"].append(
                    value.item() * loss_weights[key]
                )
            mos.append(model_output)
        start_radii = [0.97 * self._rs_max_r, 0.99 * self._rs_max_r]
        if self._best_cluster_params:
            radii: list[float] = [
                v
                for k, v in self._best_cluster_params.items()
                if k.endswith("_r") and not math.isnan(v)
            ]
            if radii:
                start_radii = [0.9 * min(radii), max(radii), 1.1 * max(radii)]
        rs = RadiusScanner(
            model_output=mos,
            radius_range=(1e-6, self._rs_max_r),
            max_num_neighbors=self._max_edges_per_node,
            n_trials=10,
            target_fracs=(0.8, 0.85, 0.88, 0.9, 0.93, 0.95),
            max_edges=self._rs_max_edges,
            start_radii=start_radii,
        )
        rs.logger.setLevel(logging.INFO)
        rsr = rs()
        self._rsr = rsr
        rsr_foms = rsr.get_foms()
        self._best_cluster_params = rsr_foms
        return (
            rsr_foms
            | {k: np.nanmean(v) for k, v in batch_metrics.items()}
            | {
                f"{k}_std": np.nanstd(v, ddof=1).item()
                for k, v in batch_metrics.items()
            }
        )


# todo: rather break down to pre-cluster-space (dim 12) and pre-beta (dim 12), apply
#   residuals there and then do another MLP to get to the output dim
#   use that to directly include r and eta in the beta calculation
class MetricLearningGraphConstruction(nn.Module):
    def __init__(
        self,
        *,
        node_indim: int,
        h_outdim: int = 12,
        L_gc: int,
        hidden_dim: int,
        n_from_eta: int = 0,
        midway_residual: bool = False,
        midway_layer_norm: bool = False,
    ):
        super().__init__()
        # Flags & settings
        self._midway_residual = midway_residual
        self._n_from_eta = n_from_eta
        self._midway_layer_norm = midway_layer_norm

        # Static modules
        self._sigmoid = nn.Sigmoid()
        self._relu = nn.ReLU()

        # Modules
        self._layer_norm = nn.LayerNorm(hidden_dim)
        self._encoder1 = MLP(
            node_indim,
            hidden_dim,
            hidden_dim,
            L=L_gc // 2,
        )
        self._encoder2 = MLP(
            hidden_dim,
            hidden_dim,
            hidden_dim,
            L=L_gc - L_gc // 2,
        )
        self._beta_nn = MLP(hidden_dim, 1, hidden_dim, L=1)
        self._latent = MLP(hidden_dim, h_outdim, hidden_dim, L=1)

    def forward(self, data) -> dict[str, torch.Tensor]:
        main_latent_halfway = self._relu(self._encoder1(data.x))
        main_latent = self._encoder2(main_latent_halfway)
        if self._midway_residual:
            main_latent += main_latent_halfway
        if self._midway_layer_norm:
            main_latent = self._layer_norm(main_latent)
        main_latent = self._relu(main_latent)
        cluster_space = self._latent(main_latent)
        eta = data.x[:, 3]
        cluster_space[:, : self._n_from_eta] += eta.reshape(-1, 1)
        beta = self._sigmoid(self._beta_nn(main_latent)).squeeze()
        return {"H": cluster_space, "B": beta}


class GCTrainable(DefaultTrainable):
    @property
    def _is_continued_run(self) -> bool:
        """We're restoring a model from a previous run and continuing."""
        return "gc_project" in self.tc

    def get_model(self) -> nn.Module:
        if self._is_continued_run:
            return self._get_restored_model()
        return self._get_new_model()

    def _get_restored_model(self):
        return restore_model(
            GCTrainable,
            tune_dir=self.tc["gc_project"],
            run_hash=self.tc["gc_hash"],
            epoch=self.tc.get("gc_epoch", -1),
            freeze=False,
        )

    def _get_new_model(self) -> nn.Module:
        return GraphConstructionFCNN(
            in_dim=14,
            **subdict_with_prefix_stripped(self.tc, "m_"),
        )

    def get_loss_functions(self) -> dict[str, Any]:
        return {
            "potential": (
                GraphConstructionHingeEmbeddingLoss(
                    r_emb=self.tc["r_emb"],
                    max_num_neighbors=self.tc["max_edges_per_node"],
                    attr_pt_thld=self.tc["attr_pt_thld"],
                    p_attr=self.tc.get("p_attr", 1),
                    p_rep=self.tc.get("p_rep", 1),
                ),
                {
                    "attractive": self.tc["lw_potential_attractive"],
                    "repulsive": self.tc["lw_potential_repulsive"],
                },
            )
        }

    def get_cluster_functions(self) -> dict[str, Any]:
        return {}

    def get_trainer(self) -> TCNTrainer:
        trainer = GCTrainer(
            model=self.get_model(),
            loaders=self.get_loaders(),
            loss_functions=self.get_loss_functions(),
            lr=self.tc["lr"],
            lr_scheduler=self.get_lr_scheduler(),
            optimizer=self.get_optimizer(),
            rs_max_edges=self.tc["rs_max_edges"],
            max_edges_per_node=self.tc["max_edges_per_node"],
            rs_max_r=self.tc["r_emb"],
        )
        trainer.logger.setLevel(logging.DEBUG)
        if self.tc["scheduler"] == "cycliclr":
            logger.info("Setting lr_scheduler_step to batch")
            trainer.lr_scheduler_step = "batch"
        return trainer
