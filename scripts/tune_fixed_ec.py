from __future__ import annotations

from argparse import ArgumentParser
from functools import partial
from typing import Any

import optuna
import torch
from gnn_tracking.models.track_condensation_networks import PreTrainedECGraphTCN
from gnn_tracking.training.tcn_trainer import TCNTrainer
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped
from rt_stoppers_contrib import ThresholdTrialStopper
from torch import nn

from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.restore import restore_model
from gnn_tracking_hpo.trainable import TCNTrainable, suggest_default_values
from gnn_tracking_hpo.tune import Dispatcher, add_common_options
from gnn_tracking_hpo.util.paths import add_scripts_path

add_scripts_path()
from tune_ec import ECTrainable  # noqa: E402


class ThresholdedEdgeLoss(nn.Module):
    def __init__(self, loss: nn.Module, threshold: float):
        super().__init__()
        self._threshold = threshold
        self._loss = loss

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return torch.nn.functional.relu(self._loss(*args, **kwargs) - self._threshold)


class PretrainedECTrainable(TCNTrainable):
    def __init__(self, config: dict[str, Any], **kwargs):
        self.ec = restore_model(
            ECTrainable,
            config["ec_project"],
            config["ec_hash"],
            config["ec_epoch"],
            freeze=config["ec_freeze"],
        )
        super().__init__(config=config, **kwargs)

    def get_edge_loss_function(self) -> tuple[nn.Module, float]:
        loss, weight = super().get_edge_loss_function()
        return ThresholdedEdgeLoss(loss, self.tc["edge_loss_threshold"]), weight

    def get_trainer(self) -> TCNTrainer:
        trainer = super().get_trainer()
        trainer.ec_threshold = self.tc["m_ec_threshold"]
        return trainer

    def get_model(self) -> nn.Module:
        return PreTrainedECGraphTCN(
            self.ec,
            node_indim=7,
            edge_indim=4,
            **subdict_with_prefix_stripped(self.tc, "m_"),
        )


def suggest_config(
    trial: optuna.Trial,
    *,
    sector: int | None = None,
    ec_project: str,
    ec_hash: str,
    ec_epoch: int = -1,
    test=False,
    fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = get_metadata(test=test)
    config.update(fixed or {})

    def d(key, *args, **kwargs):
        auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

    # Definitely Fixed hyperparameters
    # --------------------------------

    config["train_data_dir"] = [
        f"/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/graphs_v1/part_{i}"
        for i in range(1, 9)
    ]
    config[
        "val_data_dir"
    ] = "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/graphs_v1/part_9"
    d("n_graphs_train", 247776)
    d("n_graphs_val", 200)

    d("sector", sector)

    d("m_mask_orphan_nodes", True)
    d("m_use_ec_embeddings_for_hc", True)
    d("m_feed_edge_weights", True)

    d("ec_project", ec_project)
    d("ec_hash", ec_hash)
    d("ec_epoch", ec_epoch)
    d("edge_loss_threshold", 0.00019)
    d("ec_loss", "haughty_focal")
    d("ec_pt_thld", 0.81)
    d("focal_alpha", 0.45)
    d("focal_gamma", 3.5)
    d("lw_edge", 100)

    d("batch_size", 5)

    # Keep one fixed because of normalization invariance
    d("lw_potential_attractive", 1.0)

    d("m_hidden_dim", 120)
    d("m_h_dim", 120)
    d("m_e_dim", 120)

    # Most of the following parameters are fixed based on af5b5461

    d("attr_pt_thld", 0.6)
    d("q_min", 0.34)
    d("sb", 0.09)
    d("m_alpha_hc", 0.63)
    d("lw_background", 0.0041)
    d("repulsive_radius_threshold", 3.7)

    # Tuned hyperparameters
    # ---------------------

    d("ec_freeze", True)
    d("adam_beta1", 0.8, 0.99)
    d("adam_beta2", 0.990, 0.999)
    d("adam_epsilon", 1e-8, 1e-2, log=True)
    d("adam_weight_decay", 1e-7, 3e-5, log=True)
    d("lw_potential_repulsive", 0.1, 0.4)
    d("m_h_outdim", 7, 12)
    d("m_ec_threshold", 0.25, 0.5)
    d("lr", 0.0002, 0.0006)
    d("m_L_hc", 3, 5)

    suggest_default_values(config, trial, ec="continued")
    return config


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ec-hash", required=True, type=str, help="Hash of the edge classifier to load"
    )
    parser.add_argument(
        "--ec-project",
        required=True,
        type=str,
        help="Name of the folder that the edge classifier to load belongs to",
    )
    parser.add_argument(
        "--ec-epoch",
        type=int,
        default=-1,
        help="Epoch of the edge classifier to load. Defaults to -1 (last epoch).",
    )
    add_common_options(parser)
    kwargs = vars(parser.parse_args())
    this_suggest_config = partial(
        suggest_config,
        ec_hash=kwargs.pop("ec_hash"),
        ec_project=kwargs.pop("ec_project"),
        ec_epoch=kwargs.pop("ec_epoch"),
    )
    dispatcher = Dispatcher(
        grace_period=6,
        no_improvement_patience=6,
        metric="trk.double_majority_pt0.9",
        additional_stoppers=[
            ThresholdTrialStopper(
                "trk.double_majority_pt0.9", {2: 0.4, 5: 0.5, 10: 0.63, 15: 0.7}
            )
        ],
        **kwargs,
    )
    dispatcher(
        PretrainedECTrainable,
        this_suggest_config,
    )
