from __future__ import annotations

from argparse import ArgumentParser
from functools import partial
from typing import Any

import optuna
from ray.tune.stopper import MaximumIterationStopper

from gnn_tracking_hpo.cli import add_ec_restore_options, add_tc_restore_options
from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.defaults import suggest_default_values
from gnn_tracking_hpo.trainable import PretrainedECTCNTrainable
from gnn_tracking_hpo.tune import Dispatcher, add_common_options
from gnn_tracking_hpo.util.dict import pop


def suggest_config(
    trial: optuna.Trial,
    *,
    sector: int | None = None,
    ec_project: str,
    ec_hash: str,
    ec_epoch: int = -1,
    test=False,
    fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = get_metadata(test=test)
    config.update(fixed or {})

    def d(key, *args, **kwargs):
        auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

    # Definitely Fixed hyperparameters
    # --------------------------------

    config["train_data_dir"] = [
        f"/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/graphs_v1/part_{i}"
        for i in range(1, 9)
    ]
    config[
        "val_data_dir"
    ] = "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/graphs_v1/part_9"
    d("n_graphs_train", 247776)
    d("n_graphs_val", 200)

    d("sector", sector)

    d("m_mask_orphan_nodes", True)
    d("m_use_ec_embeddings_for_hc", True)
    d("m_feed_edge_weights", True)

    d("ec_project", ec_project)
    d("ec_hash", ec_hash)
    d("ec_epoch", ec_epoch)
    # d("edge_loss_threshold", 0.00019)
    # d("ec_loss", "haughty_focal")
    # d("ec_pt_thld", 0.81)
    # d("focal_alpha", 0.45)
    # d("focal_gamma", 3.5)
    # d("lw_edge", 100)

    d("batch_size", 5)

    # Keep one fixed because of normalization invariance
    d("lw_potential_attractive", 1.0)

    d("m_hidden_dim", 64)
    d("m_h_dim", 64)
    d("m_e_dim", 64)

    # Most of the following parameters are fixed based on af5b5461

    d("attr_pt_thld", 0.9)  # Changed
    d("q_min", 0.34)
    d("sb", 0.09)
    d("m_alpha_hc", 0.63)
    d("lw_background", 0.0041)
    d("repulsive_radius_threshold", 3.7)
    d("m_h_outdim", 12)
    d("m_L_hc", 4)
    d("ec_freeze", True)
    d("lw_potential_repulsive", 0.16)

    # Tuned hyperparameters
    # ---------------------

    d("m_ec_threshold", 0.3, 0.4)
    d("lr", [1e-3, 5e-4])

    suggest_default_values(config, trial, ec="continued")
    return config


class MyDispatcher(Dispatcher):
    pass
    # def get_optuna_sampler(self):
    #     return optuna.samplers.RandomSampler()


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_options(parser)
    add_ec_restore_options(parser)
    add_tc_restore_options(parser)
    kwargs = vars(parser.parse_args())
    if ("ec_hash" in kwargs) ^ ("tc_hash" in kwargs):
        raise ValueError("Must specify ec_hash XOR tc_hash at the moment")
    this_suggest_config = partial(
        suggest_config,
        **pop(
            kwargs,
            ["ec_hash", "ec_project", "ec_epoch", "tc_hash", "tc_project", "tc_epoch"],
        ),
    )
    dispatcher = Dispatcher(
        grace_period=6,
        no_improvement_patience=6,
        metric="trk.double_majority_pt0.9",
        additional_stoppers=[
            MaximumIterationStopper(10),
        ],
        **kwargs,
    )
    dispatcher(
        PretrainedECTCNTrainable,
        this_suggest_config,
    )
