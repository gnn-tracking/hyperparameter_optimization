from __future__ import annotations

from argparse import ArgumentParser
from functools import partial
from typing import Any

import optuna
from gnn_tracking.training.lw_setter import SineLWSH
from gnn_tracking.training.tc import TCNTrainer

from gnn_tracking_hpo.cli import add_restore_options
from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.defaults import suggest_default_values
from gnn_tracking_hpo.trainable import PretrainedECTCNTrainable
from gnn_tracking_hpo.tune import Dispatcher, add_common_options
from gnn_tracking_hpo.util.dict import pop


class LWSPretrainedECTCNTrainable(PretrainedECTCNTrainable):
    def get_trainer(self) -> TCNTrainer:
        trainer = super().get_trainer()
        trainer.add_hook(self.get_lws_hook(), "batch")
        return trainer

    def get_lws_hook(self):
        return SineLWSH(
            loss_name=("potential", "repulsive"),
            mean=self.tc["lw_potential_repulsive"],
            amplitude=self.tc["lws_repulsive_sine_amplitude"],
            period=self.tc["lws_repulsive_sine_period"],
            amplitude_halflife=self.tc["lws_repulsive_sine_amplitude_halflife"],
            n_batches=12000,
        )


def suggest_config(
    trial: optuna.Trial,
    *,
    sector: int | None = None,
    ec_project: str,
    ec_hash: str,
    ec_epoch: int = -1,
    tc_project: str = "",
    tc_hash: str = "",
    tc_epoch: int = -1,
    test=False,
    fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = get_metadata(test=test)
    config.update(fixed or {})

    def d(key, *args, **kwargs):
        return auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

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

    if tc_hash:
        d("tc_project", tc_project)
        d("tc_hash", tc_hash)
        d("tc_epoch", tc_epoch)
    if ec_hash:
        d("ec_project", ec_project)
        d("ec_hash", ec_hash)
        d("ec_epoch", ec_epoch)

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
    d("m_h_outdim", 12)
    ec_freeze = d("ec_freeze", True)

    if not ec_freeze:
        d("lw_edge", 2_000)

    d("repulsive_radius_threshold", 3.7)
    d("lr", [7e-4])
    d("m_ec_threshold", 0.27)
    d("m_L_hc", 3)

    # Tuned hyperparameters
    # ---------------------

    d("lw_potential_repulsive", [0.2, 0.28, 0.36])
    d("lws_repulsive_sine_amplitude", [-0.10, -0.15, -0.20])
    d("lws_repulsive_sine_period", [2, 4])
    d("lws_repulsive_sine_amplitude_halflife", [4, 6])

    ec_suggestions = "continued" if ec_freeze else "fixed"
    suggest_default_values(config, trial, ec=ec_suggestions)
    return config


class MyDispatcher(Dispatcher):
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_options(parser)
    add_restore_options(parser)
    add_restore_options(parser, prefix="tc", name="TCN")
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
        grace_period=10,
        no_improvement_patience=3,
        metric="trk.double_majority_pt0.9",
        # additional_stoppers=[
        #     MaximumIterationStopper(10),
        # ],
        **kwargs,
    )
    dispatcher(
        LWSPretrainedECTCNTrainable,
        this_suggest_config,
    )
