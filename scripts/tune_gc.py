from __future__ import annotations

import collections
from argparse import ArgumentParser
from math import isnan
from typing import Any, DefaultDict

import optuna
from gnn_tracking.training.lw_setter import LinearLWSH
from gnn_tracking.training.tcn_trainer import TCNTrainer
from ray import tune
from rt_stoppers_contrib import NoImprovementTrialStopper

from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.defaults import suggest_default_values
from gnn_tracking_hpo.trainable import GCTrainable
from gnn_tracking_hpo.tune import Dispatcher, add_common_options


class LWSGCTrainable(GCTrainable):
    def get_trainer(self) -> TCNTrainer:
        trainer = super().get_trainer()
        trainer.add_hook(self.get_lws_hook(), "batch")
        return trainer

    def get_lws_hook(self):
        return LinearLWSH(
            loss_name=("potential", "repulsive"),
            start_value=self.tc["lws_repulsive_llw_start_value"],
            end_value=self.tc["lw_potential_repulsive"],
            end=self.tc["lws_repulsive_llw_end"],
            n_batches=800,
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

    d("m_hidden_dim", 512)
    d("lw_potential_attractive", 1.0)
    d("attr_pt_thld", 0.9)
    d("sb", 0.09)
    d("q_min", 0.34)

    d("max_edges_per_node", 256)
    d("m_L_gc", 6)
    d("rs_max_edges", 10_000_000)
    d("max_sample_size", 800)
    d("lr", 1e-3)
    d("repulsive_radius_threshold", 5)
    d("lw_background", 5e-4)
    d("m_midway_residual", True)
    d("m_midway_layer_norm", False)
    d("m_h_outdim", 12)
    d("m_n_from_eta", 0)

    # Tuned parameters
    # ----------------

    final_potential = d("lw_potential_repulsive", 1e-1, 1)

    # d("adam_weight_decay", 0)
    # d("adam_beta1", 0.9, 0.99)
    # d("adam_beta2", 0.9, 0.9999)

    d("lws_repulsive_llw_start_value", 5e-4)
    if final_potential > 0.5:
        d("lws_repulsive_llw_end", 9, 20)
    else:
        d("lws_repulsive_llw_end", 1, 9)

    suggest_default_values(config, trial, hc="none", ec="none")
    return config


class NoNaNStopper(tune.Stopper):
    def __init__(self, settings: dict[str, tuple[int, int]]):
        self._settings = settings
        self._epoch: DefaultDict[Any, int] = collections.defaultdict(int)
        self._n_consecutive_nans: Any = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )

    def __call__(self, trial_id: str, result: dict) -> bool:
        self._epoch[trial_id] += 1
        for metric, (min_epoch, patience) in self._settings.items():
            if self._epoch[trial_id] < min_epoch:
                continue
            if isnan(result[metric]):
                self._n_consecutive_nans[trial_id][metric] += 1
                if self._n_consecutive_nans[trial_id][metric] >= patience:
                    return True
            else:
                self._n_consecutive_nans[trial_id][metric] = 0
        return False

    def stop_all(self) -> bool:
        return False


class MyDispatcher(Dispatcher):
    def get_no_improvement_stopper(self) -> NoImprovementTrialStopper:
        return NoImprovementTrialStopper(
            metric="total",
            patience=5,
            mode="min",
            grace_period=5,
            rel_change_thld=0.01,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_options(parser)
    kwargs = vars(parser.parse_args())
    assert kwargs["wandb_project"] == "gnn_tracking_gc"

    nn_stopper = NoNaNStopper(
        {
            "n_edges_frac_segment50_80": (5, 3),
            "n_edges_frac_segment50_90": (8, 3),
            "n_edges_frac_segment50_93": (12, 4),
        }
    )
    dispatcher = MyDispatcher(
        **kwargs,
        metric="n_edges_frac_segment50_80",
        grace_period=10,
        comparison="min",
        additional_stoppers=[nn_stopper],
        no_scheduler=True,
    )
    dispatcher(LWSGCTrainable, suggest_config)
