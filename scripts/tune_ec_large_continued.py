"""This script trains the edge classification separately from the rest of the
model.
"""

from __future__ import annotations

from argparse import ArgumentParser
from functools import partial
from typing import Any

import optuna
from rt_stoppers_contrib import NoImprovementTrialStopper, ThresholdTrialStopper
from torch import nn

from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.restore import restore_model
from gnn_tracking_hpo.trainable import suggest_default_values
from gnn_tracking_hpo.tune import Dispatcher, add_common_options
from gnn_tracking_hpo.util.paths import add_scripts_path, get_config

add_scripts_path()
from tune_ec import ECTrainable  # noqa: E402


class ContinuedECTrainable(ECTrainable):
    def get_model(self) -> nn.Module:
        return restore_model(
            ECTrainable,
            self.tc["ec_project"],
            self.tc["ec_hash"],
            self.tc.get("ec_epoch", -1),
            freeze=False,
        )


def suggest_config(
    trial: optuna.Trial,
    *,
    ec_hash: str,
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

    ec_project = d("ec_project", "ec")
    ec_hash = d("ec_hash", ec_hash)
    d("ec_epoch", -1)
    original_config = get_config(ec_project, ec_hash)

    # Tuned parameters
    # ----------------

    d("lr", 5e-5, 1e-4)
    eps = 0.1
    ea = 1 - eps
    eb = 1 + eps
    d(
        "ec_pt_thld",
        ea * original_config["ec_pt_thld"],
        eb * original_config["ec_pt_thld"],
    )
    d(
        "focal_alpha",
        ea * original_config["focal_alpha"],
        eb * original_config["focal_alpha"],
    )
    d(
        "focal_gamma",
        ea * original_config["focal_gamma"],
        eb * original_config["focal_gamma"],
    )
    d("scheduler", "exponentiallr")
    d("exponentiallr_gamma", 0.8, 0.9)
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
    parser.add_argument("--ec-hash", type=str, required=True)
    kwargs = vars(parser.parse_args())
    kwargs.pop("no_scheduler")
    ec_hash = kwargs.pop("ec_hash")
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
                    6: 0.96,
                },
            )
        ],
    )
    dispatcher(
        ContinuedECTrainable,
        partial(suggest_config, ec_hash=ec_hash),
    )