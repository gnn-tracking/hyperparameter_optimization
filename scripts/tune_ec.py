"""This script trains the edge classification separately from the rest of the
model.
"""

from __future__ import annotations

import subprocess
from argparse import ArgumentParser
from typing import Any

import optuna
from gnn_tracking.models.edge_classifier import ECForGraphTCN
from gnn_tracking.training.tcn_trainer import TCNTrainer
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped
from torch import nn

from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.trainable import TCNTrainable, suggest_default_values
from gnn_tracking_hpo.tune import Dispatcher, add_common_options
from gnn_tracking_hpo.util.log import logger


class SignatureAdaptedECForGraphTCN(ECForGraphTCN):
    """Adapt signature of ECForGraphTCN to match the signature of the
    main model.
    """

    def forward(self, *args, **kwargs):
        return {"W": ECForGraphTCN.forward(self, *args, **kwargs)}


class ECTrainable(TCNTrainable):
    def pre_setup_hook(self):
        result = subprocess.run(
            ["hostname", "--ip-address"], capture_output=True, text=True
        )
        logger.info(result)

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

    def get_model(self) -> nn.Module:
        return SignatureAdaptedECForGraphTCN(
            node_indim=6, edge_indim=4, **subdict_with_prefix_stripped(self.tc, "m_")
        )


def suggest_config(
    trial: optuna.Trial,
    *,
    test=False,
    fixed: dict[str, Any] | None = None,
    sector: int | None = None,
) -> dict[str, Any]:
    config = get_metadata(test=test)
    config.update(fixed or {})

    def d(key, *args, **kwargs):
        auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

    d("sector", sector)
    d("lw_edge", 1.0)

    d("n_graphs_train", 12180)
    d("n_graphs_val", 299)
    d("n_graphs_val", 1)
    d("batch_size", 5)

    if sector is not None:
        # Currently only have limited graphs available in that case
        d("n_graphs_train", 300)
        d("n_graphs_val", 69)
        d("n_graphs_test", 1)

    d("training_pt_thld", 0.0, 0.9)
    d("training_without_noise", [True, False])
    d("training_without_non_reconstructable", [True, False])

    # Tuned parameters
    # ----------------

    d("lr", 0.0001, 0.0006)
    d("m_hidden_dim", 64, 256)
    d("m_L_ec", 6, 8)
    d("focal_gamma", 0.0, 5.0)
    d("focal_alpha", 0.1, 1.0)
    d("m_alpha_ec", 0.3, 0.99)
    d("m_interaction_node_hidden_dim", 32, 128)
    d("m_interaction_edge_hidden_dim", 32, 128)
    d("ec_pt_thld", 0.0, 0.9)

    suggest_default_values(config, trial, hc="none")
    return config


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_options(parser)
    kwargs = vars(parser.parse_args())

    dispatcher = Dispatcher(
        **kwargs,
        metric="max_mcc_pt0.9",
        grace_period=2,
        no_improvement_patience=2,
    )
    dispatcher(
        ECTrainable,
        suggest_config,
    )
