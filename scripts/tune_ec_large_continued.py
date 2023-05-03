"""This script trains the edge classification separately from the rest of the
model.
"""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Any

import optuna
from rt_stoppers_contrib import NoImprovementTrialStopper, ThresholdTrialStopper
from torch import nn

from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.restore import restore_model
from gnn_tracking_hpo.trainable import suggest_default_values
from gnn_tracking_hpo.tune import Dispatcher, add_common_options
from gnn_tracking_hpo.util.paths import add_scripts_path

add_scripts_path()
from tune_ec import ECTrainable  # noqa: E402


class ContinuedECTrainable(ECTrainable):
    def get_model(self) -> nn.Module:
        return restore_model(
            ECTrainable,
            self.tc["ec_project"],
            self.tc["ec_hash"],
            self.tc["ec_epoch"],
            freeze=False,
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

    # fixed parameters
    # -----------------------

    d("m_L_ec", 6)
    d("m_residual_type", "skip1")
    d("m_use_node_embedding", [True])
    d("m_use_intermediate_edge_embeddings", True)
    d("m_interaction_node_dim", 120)
    d("m_interaction_edge_dim", 120)
    d("m_hidden_dim", 120)
    d("m_alpha", 0.5)

    d("ec_project", "ec_230502")
    d("ec_hash", "759be1c8")
    d("ec_epoch", -1)
    d("ec_pt_thld", 0.8292467946621858)
    d("focal_alpha", 0.430815829482855)
    d("focal_gamma", 3.781611232231564)

    # Tuned parameters
    # ----------------

    d("lr", 1e-9, 1e-4, log=True)
    # d("adam_weight_decay", 0)
    # d("adam_beta1", 0.9, 0.99)
    # d("adam_epsilon", 1e-8, 1e-5, log=True)
    # d("adam_beta2", 0.9, 0.9999)

    suggest_default_values(config, trial, hc="none")
    return config


class MyDispatcher(Dispatcher):
    def get_no_improvement_stopper(self) -> NoImprovementTrialStopper:
        return NoImprovementTrialStopper(
            metric=self.metric,
            patience=6,
            mode="max",
            grace_period=0,
            rel_change_thld=0.003,
        )

    def get_optuna_sampler(self):
        return optuna.samplers.RandomSampler()


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_options(parser)
    kwargs = vars(parser.parse_args())
    dispatcher = MyDispatcher(
        **kwargs,
        metric="max_mcc_pt0.9",
        no_scheduler=True,
        additional_stoppers=[
            ThresholdTrialStopper(
                "max_mcc_pt0.9",
                {
                    0: 0.91,
                    2: 0.95,
                },
            )
        ],
    )
    dispatcher(
        ContinuedECTrainable,
        suggest_config,
    )
