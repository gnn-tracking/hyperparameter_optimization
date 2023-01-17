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
from torch import nn

from gnn_tracking_hpo.cli import add_truth_cut_options
from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.trainable import TCNTrainable, suggest_default_values
from gnn_tracking_hpo.tune import add_common_options, main


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

        if self.tc["sector"] is not None:
            # Because we don't have many graphs, let's train
            # longer before validation
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
    ec_pt_thld: float = 0.0,
    training_pt_thld=0.0,
    training_without_noise=False,
    training_without_non_reconstructable=False,
) -> dict[str, Any]:
    config = get_metadata(test=test)
    config.update(fixed or {})

    def d(key, *args, **kwargs):
        auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

    d("ec_pt_thld", ec_pt_thld)
    d("sector", sector)
    d("lw_edge", 1.0)

    if sector is not None:
        # Currently only have limited graphs available in that case
        d("n_graphs_train", 300)
        d("n_graphs_val", 69)
        d("n_graphs_test", 1)

    d("training_pt_thld", training_pt_thld)
    d("training_without_noise", training_without_noise)
    d("training_without_non_reconstructable", training_without_non_reconstructable)

    # Tuned parameters
    # ----------------
    d("lr", 0.0001, 0.0006)
    d("m_hidden_dim", 64, 256)
    d("m_L_ec", 1, 5)
    d("focal_gamma", 1, 20)  # 5 might be a good default
    d("focal_alpha", 0.1, 1)  # 0.95 might be a good default
    d("m_alpha_ec", 0.3, 0.99)
    d("m_interaction_node_hidden_dim", 32, 128)
    d("m_interaction_edge_hidden_dim", 32, 128)

    suggest_default_values(config, trial, hc="none")
    return config


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_options(parser)
    parser.add_argument("--sector", type=int, required=False, default=None)
    parser.add_argument(
        "--ec-pt-thld",
        type=float,
        default="0.",
        required=False,
        help="Falsify all edges below this pt value",
    )
    add_truth_cut_options(parser)
    kwargs = vars(parser.parse_args())

    sector = kwargs.pop("sector")
    main(
        ECTrainable,
        partial(
            suggest_config,
            sector=sector,
            ec_pt_thld=kwargs.pop("ec_pt_thld"),
            training_pt_thld=kwargs.pop("training_pt_thld"),
            training_without_noise=kwargs.pop("training_without_noise"),
            training_without_non_reconstructable=kwargs.pop(
                "training_without_non_reconstructable"
            ),
        ),
        **kwargs,
        metric="tpr_eq_tnr_pt0.9",
        grace_period=11 if sector is not None else 4,
        no_improvement_patience=19 if sector is not None else 6,
    )
