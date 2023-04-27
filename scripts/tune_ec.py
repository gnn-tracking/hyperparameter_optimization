"""This script trains the edge classification separately from the rest of the
model.
"""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Any

import optuna
from gnn_tracking.models.edge_classifier import ECForGraphTCN
from gnn_tracking.training.tcn_trainer import TCNTrainer
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped
from rt_stoppers_contrib import ThresholdTrialStopper
from torch import nn

from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.trainable import TCNTrainable, suggest_default_values
from gnn_tracking_hpo.tune import Dispatcher, add_common_options


class ECTrainable(TCNTrainable):
    def get_loss_functions(self) -> dict[str, Any]:
        return {"edge": self.get_edge_loss_function()}

    def get_cluster_functions(self) -> dict[str, Any]:
        return {}

    def get_trainer(self) -> TCNTrainer:
        trainer = super().get_trainer()
        trainer.ec_eval_pt_thlds = [0.0, 0.5, 0.9, 1.2, 1.5]
        return trainer

    def get_model(self) -> nn.Module:
        return ECForGraphTCN(
            node_indim=7, edge_indim=4, **subdict_with_prefix_stripped(self.tc, "m_")
        )


def suggest_config(
    trial: optuna.Trial,
    *,
    test=False,
    fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = get_metadata(test=test)
    config.update(fixed or {})

    def d(key, *args, **kwargs):
        return auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

    d("lw_edge", 1.0)
    d("_val_batch_size", 5)
    d("ec_loss", "haughty_focal")

    # (Almost) fixed parameters
    # -----------------------

    d("m_residual_type", "skip1")
    d("m_use_node_embedding", [True])
    d("m_use_intermediate_edge_embeddings", [False])
    d("m_L_ec", 6)

    # Tuned parameters
    # ----------------

    d("ec_pt_thld", 0.75, 0.85)
    d("focal_gamma", 3, 4)  # 2 4
    d("focal_alpha", 0.4, 0.5)  # 0.35, 0.45
    d("lr", 2e-4, 8e-4)
    d("m_alpha", 0.3, 0.9)
    d("m_interaction_node_dim", 32, 64)  # 32 64
    d("m_interaction_edge_dim", 32, 64)  # 32 64
    d("adam_beta1", 0.9, 0.99)
    # d("adam_beta2", 0.990, 0.999)
    # d("adam_eps", 1e-9, 1e-7, log=True)
    # rt = d("m_residual_type", ["skip1", "skip2", "skip_top"])
    # if rt == "skip2":
    #     # This is a hack because of this: https://github.com/optuna/optuna/issues/372
    #     # choice = random.choice([4, 6])
    #     d("m_L_ec", 6)
    # else:
    #     d("m_L_ec", 5, 6)

    suggest_default_values(config, trial, hc="none")
    return config


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_options(parser)
    kwargs = vars(parser.parse_args())
    # todo: remove me
    kwargs.pop("no_scheduler")
    dispatcher = Dispatcher(
        **kwargs,
        metric="max_mcc_pt0.9",
        grace_period=4,
        no_improvement_patience=15,
        # todo: remove me
        no_scheduler=True,
        additional_stoppers=[
            ThresholdTrialStopper(
                "max_mcc_pt0.9",
                {
                    11: 0.8,
                    21: 0.85,
                },
            )
        ],
    )
    dispatcher(
        ECTrainable,
        suggest_config,
    )
