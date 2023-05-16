"""This script trains the edge classification separately from the rest of the
model.
"""

from __future__ import annotations

from argparse import ArgumentParser
from functools import partial
from typing import Any

import optuna
from gnn_tracking.models.edge_classifier import ECForGraphTCN
from gnn_tracking.training.tcn_trainer import TCNTrainer
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped
from rt_stoppers_contrib import NoImprovementTrialStopper
from torch import nn

from gnn_tracking_hpo.cli import add_ec_restore_options
from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.restore import restore_model
from gnn_tracking_hpo.trainable import TCNTrainable, suggest_default_values
from gnn_tracking_hpo.tune import Dispatcher, add_common_options
from gnn_tracking_hpo.util.dict import pop


class ECTrainable(TCNTrainable):
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


def suggest_config(
    trial: optuna.Trial,
    *,
    test=False,
    fixed: dict[str, Any] | None = None,
    ec_hash: str = "",
    ec_project: str = "",
    ec_epoch: int = -1,
) -> dict[str, Any]:
    config = get_metadata(test=test)
    config.update(fixed or {})

    def d(key, *args, **kwargs):
        return auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

    d("lw_edge", 1.0)

    d("n_graphs_train", 247776)
    config["train_data_dir"] = [
        f"/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/graphs_v1/part_{i}"
        for i in range(1, 9)
    ]
    d(
        "val_data_dir",
        "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/graphs_v1/part_9",
    )
    d("n_graphs_val", 320)
    d("batch_size", 5)
    d("_val_batch_size", 5)
    d("ec_loss", "haughty_focal")

    # Restoring
    # -----------------------

    if ec_hash:
        d("ec_project", ec_project)
        d("ec_hash", ec_hash)
        d("ec_epoch", ec_epoch)

    # fixed parameters
    # -----------------------

    d("m_L_ec", 6)
    d("m_residual_type", "skip1")
    d("m_use_node_embedding", True)
    d("m_use_intermediate_edge_embeddings", True)
    d("m_interaction_node_dim", 64)
    d("m_interaction_edge_dim", 64)
    d("m_hidden_dim", 64)
    d("m_alpha", 0.5)
    d("ec_pt_thld", 0.9)
    d("focal_alpha", 0.3966639332867394)
    d("focal_gamma", 3.9912747796867887)

    # Tuned parameters
    # ----------------

    d("lr", 1e-4, 1e-3, log=True)
    # d("adam_weight_decay", 0)
    # d("adam_beta1", 0.9, 0.99)
    # d("adam_beta2", 0.9, 0.9999)

    suggest_default_values(config, trial, hc="none")
    return config


class MyDispatcher(Dispatcher):
    def get_no_improvement_stopper(self) -> NoImprovementTrialStopper:
        return NoImprovementTrialStopper(
            metric=self.metric,
            patience=10,
            mode="max",
            grace_period=0,
            rel_change_thld=0.005,
        )

    # def get_optuna_sampler(self):
    #     return optuna.samplers.RandomSampler()


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_options(parser)
    add_ec_restore_options(parser)
    kwargs = vars(parser.parse_args())
    this_suggest_config = partial(
        suggest_config,
        **pop(kwargs, ["ec_hash", "ec_project", "ec_epoch"]),
    )
    dispatcher = MyDispatcher(
        **kwargs,
        metric="max_mcc_pt0.9",
    )
    dispatcher(ECTrainable, this_suggest_config)
