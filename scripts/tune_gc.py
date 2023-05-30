from __future__ import annotations

from argparse import ArgumentParser
from typing import Any

import optuna
from rt_stoppers_contrib import NoImprovementTrialStopper

from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.defaults import suggest_default_values
from gnn_tracking_hpo.trainable import GCTrainable
from gnn_tracking_hpo.tune import Dispatcher, add_common_options


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

    d("n_graphs_train", 7743)
    config["train_data_dir"] = [
        f"/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/point_clouds_v2/part_{i}"
        for i in range(1, 9)
    ]
    d(
        "val_data_dir",
        "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/point_clouds_v2/part_9",
    )
    d("n_graphs_val", 5)
    d("batch_size", 1)
    d("_val_batch_size", 1)

    # fixed parameters
    # -----------------------

    d("m_hidden_dim", 256)
    d("lw_potential_attractive", 1.0)
    d("attr_pt_thld", 0.9)
    d("sb", 0.09)
    d("q_min", 0.34)

    # Tuned parameters
    # ----------------
    d("m_L_gc", [5, 6, 8])
    d("lr", [3e-3, 1e-3])
    d("m_h_outdim", [8, 12, 16])
    d("repulsive_radius_threshold", 1, 10)
    d("lw_background", 1e-4, 3e-3)
    d("lw_potential_repulsive", 2e-4, 1e-2)

    # d("adam_weight_decay", 0)
    # d("adam_beta1", 0.9, 0.99)
    # d("adam_beta2", 0.9, 0.9999)

    suggest_default_values(config, trial, hc="none", ec="none")
    return config


class MyDispatcher(Dispatcher):
    def get_no_improvement_stopper(self) -> NoImprovementTrialStopper:
        return NoImprovementTrialStopper(
            metric="total",
            patience=3,
            mode="min",
            grace_period=3,
            rel_change_thld=0.01,
        )

    # def get_optuna_sampler(self):
    #     return optuna.samplers.RandomSampler()


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_options(parser)
    kwargs = vars(parser.parse_args())
    assert kwargs["wandb_project"] == "gnn_tracking_gc"
    dispatcher = MyDispatcher(
        **kwargs, metric="n_edges_frac_segment50_90", grace_period=4, comparison="min"
    )
    dispatcher(GCTrainable, suggest_config)
