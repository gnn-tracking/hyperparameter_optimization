from __future__ import annotations

from argparse import ArgumentParser
from functools import partial
from typing import Any

import optuna
from rt_stoppers_contrib import NoImprovementTrialStopper

from gnn_tracking_hpo.cli import add_restore_options
from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.defaults import suggest_default_values
from gnn_tracking_hpo.trainable import GCWithECTrainable
from gnn_tracking_hpo.tune import Dispatcher, add_common_options
from gnn_tracking_hpo.util.dict import pop


def suggest_config(
    trial: optuna.Trial,
    *,
    test=False,
    fixed: dict[str, Any] | None = None,
    gc_hash: str = "",
    gc_project: str = "",
    gc_epoch: int = -1,
) -> dict[str, Any]:
    config = get_metadata(test=test)
    config.update(fixed or {})

    def d(key, *args, **kwargs):
        return auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

    d("lw_edge", 1.0)

    d("n_graphs_train", 7463)
    config["train_data_dir"] = [
        f"/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/point_clouds_v5/part_{i}"
        for i in range(1, 9)
    ]
    d(
        "val_data_dir",
        "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/point_clouds_v5/part_9",
    )
    d("n_graphs_val", 10)
    d("batch_size", 1)
    d("_val_batch_size", 1)
    d("ec_loss", "haughty_focal")

    # Getting graph construction
    # -----------------------

    d("gc_project", gc_project)
    d("gc_hash", gc_hash)
    d("gc_epoch", gc_epoch)

    # fixed parameters
    # -----------------------

    d("m_L_ec", 6)
    d("m_residual_type", "skip1")
    d("m_use_node_embedding", True)
    d("m_use_intermediate_edge_embeddings", False)
    d("m_interaction_node_dim", 64)
    d("m_interaction_edge_dim", 64)
    d("m_hidden_dim", 64)
    d("m_alpha", 0.5)
    d("ec_pt_thld", 0.9)
    d("focal_alpha", 0.4)
    d("focal_gamma", 4)
    d("max_radius", 0.7)
    d("max_num_neighbors", 64)
    d("ec_use_embedding_features", True)

    # Tuned parameters
    # ----------------

    d("lr", 1e-3, log=True)
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
    add_restore_options(parser, required=True, prefix="gc", name="graph_construction")
    kwargs = vars(parser.parse_args())
    this_suggest_config = partial(
        suggest_config,
        **pop(kwargs, ["gc_hash", "gc_project", "gc_epoch"]),
    )
    dispatcher = MyDispatcher(
        **kwargs,
        metric="max_mcc_pt0.9",
    )
    dispatcher(GCWithECTrainable, this_suggest_config)
