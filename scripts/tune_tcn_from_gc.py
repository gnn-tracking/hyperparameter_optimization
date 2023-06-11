from __future__ import annotations

from argparse import ArgumentParser
from functools import partial
from typing import Any

import optuna

from gnn_tracking_hpo.cli import add_restore_options
from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.defaults import suggest_default_values
from gnn_tracking_hpo.trainable import TCNFromGCTrainable
from gnn_tracking_hpo.tune import Dispatcher, add_common_options
from gnn_tracking_hpo.util.dict import pop


def suggest_config(
    trial: optuna.Trial,
    *,
    sector: int | None = None,
    ec_project: str,
    ec_hash: str,
    ec_epoch: int = -1,
    gc_project: str = "",
    gc_hash: str = "",
    gc_epoch: int = -1,
    test=False,
    fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = get_metadata(test=test)
    config.update(fixed or {})

    def d(key, *args, **kwargs):
        return auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

    # Definitely Fixed hyperparameters
    # --------------------------------

    d("n_graphs_train", 7463)
    config["train_data_dir"] = [
        f"/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/point_clouds_v5/part_{i}"
        for i in range(1, 9)
    ]
    d(
        "val_data_dir",
        "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/point_clouds_v5/part_9",
    )
    d("max_sample_size", 300)
    d("n_graphs_val", 2)
    d("batch_size", 1)
    d("_val_batch_size", 1)

    d("sector", sector)

    d("m_mask_orphan_nodes", False)
    d("m_use_ec_embeddings_for_hc", False)
    d("m_feed_edge_weights", True)

    # Graph construction
    d("max_radius", 0.8)
    d("max_num_neighbors", 64)
    d("ec_use_embedding_features", True)
    d("gc_project", gc_project)
    d("gc_hash", gc_hash)
    d("gc_epoch", gc_epoch)

    d("ec_project", ec_project)
    d("ec_hash", ec_hash)
    d("ec_epoch", ec_epoch)
    d("ec_model", "ec")

    d("batch_size", 1)

    # Keep one fixed because of normalization invariance
    d("lw_potential_attractive", 1.0)

    d("m_hidden_dim", 128)
    d("m_h_dim", 128)
    d("m_e_dim", 128)
    d("node_indim", 14 + 8)
    d("edge_indim", (14 + 8) * 2)

    # Most of the following parameters are fixed based on af5b5461

    d("attr_pt_thld", 0.9)  # Changed
    d("q_min", 0.34)
    d("sb", 0.09)
    d("m_alpha_hc", 0.63)
    d("lw_background", 0.0041)
    d("m_h_outdim", 12)
    ec_freeze = d("ec_freeze", True)

    if not ec_freeze:
        d("lw_edge", 2_000)

    d("repulsive_radius_threshold", 1)
    d("lr", [7e-4])
    d("m_ec_threshold", 0.20)
    d("m_L_hc", 3)

    # Tuned hyperparameters
    # ---------------------

    d("lw_potential_repulsive", [0.5])
    # d("lws_repulsive_sine_amplitude", [-0.10, -0.15, -0.20])
    # d("lws_repulsive_sine_period", [2, 4])
    # d("lws_repulsive_sine_amplitude_halflife", [4, 6])

    ec_suggestions = "continued" if ec_freeze else "fixed"
    suggest_default_values(config, trial, ec=ec_suggestions)
    return config


class MyDispatcher(Dispatcher):
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_options(parser)
    add_restore_options(parser)
    add_restore_options(parser, prefix="gc", name="graph construction")
    kwargs = vars(parser.parse_args())
    this_suggest_config = partial(
        suggest_config,
        **pop(
            kwargs,
            ["ec_hash", "ec_project", "ec_epoch", "gc_hash", "gc_project", "gc_epoch"],
        ),
    )
    dispatcher = Dispatcher(
        grace_period=10,
        no_improvement_patience=3,
        metric="trk.double_majority_pt0.9",
        # additional_stoppers=[
        #     MaximumIterationStopper(10),
        # ],
        **kwargs,
    )
    dispatcher(
        TCNFromGCTrainable,
        this_suggest_config,
    )
