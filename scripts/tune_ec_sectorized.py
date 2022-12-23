"""This script trains the edge classification separately from the rest of the
model.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import click
import optuna
from gnn_tracking.models.edge_classifier import ECForGraphTCN
from gnn_tracking.training.tcn_trainer import TCNTrainer
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped
from torch import nn

from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.trainable import TCNTrainable, suggest_default_values
from gnn_tracking_hpo.tune import common_options, main


class SignatureAdaptedECForGraphTCN(ECForGraphTCN):
    def forward(self, *args, **kwargs):
        return {"W": ECForGraphTCN.forward(self, *args, **kwargs)}


class ECTrainable(TCNTrainable):
    def get_loss_functions(self) -> dict[str, Any]:
        return {
            "edge": self.get_edge_loss_function(),
        }

    def get_cluster_functions(self) -> dict[str, Any]:
        return {}

    def get_trainer(self) -> TCNTrainer:
        trainer = super().get_trainer()

        def test_every(*args, **kwargs):
            if trainer._epoch % 9 == 1:
                # note: first epoch we test is epoch 1
                trainer.last_test_result = TCNTrainer.test_step(
                    trainer, *args, **kwargs
                )
            return trainer.last_test_result

        trainer.test_step = test_every
        trainer.pt_thlds = [0.0, 0.9, 1.5]

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
    d("n_graphs_train", 300)
    d("n_graphs_val", 69)
    d("n_graphs_test", 1)
    d("lr", 0.0001, 0.0006)
    d("m_hidden_dim", 64, 256)
    d("m_L_ec", 1, 7)
    d("focal_gamma", 1, 20)  # 5 might be a good default
    d("focal_alpha", 0.1, 1)  # 0.95 might be a good default
    d("m_alpha_ec", 0.3, 0.99)
    d("m_interaction_node_hidden_dim", 32, 128)
    d("m_interaction_edge_hidden_dim", 32, 128)
    d("lw_edge", 1.0)

    suggest_default_values(config, trial, hc="none")
    return config


@click.command()
@click.option("--sector", type=int, required=True)
@common_options
def real_main(sector, **kwargs):
    main(
        ECTrainable,
        partial(suggest_config, sector=sector),
        **kwargs,
        metric="roc_auc_5FPR",
        grace_period=11,
        no_improvement_patience=19,
    )


if __name__ == "__main__":
    real_main()
