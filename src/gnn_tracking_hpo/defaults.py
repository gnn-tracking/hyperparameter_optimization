from __future__ import annotations

import os
from typing import Any

import optuna

from gnn_tracking_hpo.util.log import logger


def suggest_default_values(
    config: dict[str, Any],
    trial: None | optuna.Trial = None,
    ec="default",
    hc="default",
) -> None:
    """Set all config values, so that everything gets recorded in the database, even
    if we do not change anything.

    Args:
        config: Gets modified in place
        trial:
        ec: One of "default" (train), "perfect" (perfect ec), "fixed", "continued"
            (fixed architecture, continued training)
        hc: One of "default" (train), "none" (no hc)
    """
    if "adam_epsilon" in config:
        raise ValueError("It's adam_eps, not adam_epsilon")
    if ec not in ["default", "perfect", "fixed", "continued"]:
        raise ValueError(f"Invalid ec: {ec}")
    if hc not in ["default", "none"]:
        raise ValueError(f"Invalid hc: {hc}")

    c = {**config, **(trial.params if trial is not None else {})}

    def d(k, v):
        if trial is not None and k in trial.params:
            return
        if k in config:
            return
        config[k] = v
        c[k] = v

    d("node_indim", 7)
    d("edge_indim", 4)

    if test_data_dir := os.environ.get("TEST_TRAIN_DATA_DIR"):
        d("train_data_dir", test_data_dir)
        d("val_data_dir", test_data_dir)
    else:
        d(
            "train_data_dir",
            [
                f"/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/graphs_v3/part_{i}"
                for i in range(1, 9)
            ],
        )
        d(
            "val_data_dir",
            "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/graphs_v3/part_9",
        )

    if c["test"]:
        config["n_graphs_train"] = 1
        config["n_graphs_val"] = 1
    else:
        # Don't include val graphs
        d("n_graphs_train", 7743)
        d("n_graphs_val", 10)

    d("ec_pt_thld", 0.0)

    d("sector", None)
    d("batch_size", 1)
    d("_val_batch_size", 1)

    if hc != "none":
        d("repulsive_radius_threshold", 10.0)
        d("lw_potential_attractive", 1.0)
        d("lw_potential_repulsive", 1.0)
        d("lw_background", 1.0)

    if ec == "perfect":
        d("m_ec_tpr", 1.0)
        d("m_ec_tnr", 1.0)
    elif ec in ["fixed", "continued", "default"]:
        d("lw_edge", 1.0)
        if hc != "none":
            d("m_ec_threshold", 0.5)

    # Loss function parameters
    if hc != "none":
        d("q_min", 0.01)
        d("attr_pt_thld", 0.9)
        d("sb", 0.1)

    d("ec_loss", "focal")
    if ec in ["default", "continued"] and c["ec_loss"] in ["focal", "haughty_focal"]:
        d("focal_alpha", 0.25)
        d("focal_gamma", 2.0)

    # Optimizers
    d("lr", 5e-4)
    d("optimizer", "adam")
    if c["optimizer"] == "adam":
        d("adam_beta1", 0.9)
        d("adam_beta2", 0.999)
        d("adam_eps", 1e-8)
        d("adam_weight_decay", 0.0)
        d("adam_amsgrad", False)
    elif c["optimizer"] == "sgd":
        d("sgd_momentum", 0.0)
        d("sgd_weight_decay", 0.0)
        d("sgd_nesterov", False)
        d("sgd_dampening", 0.0)

    d("scheduler", None)

    # Schedulers
    if c["scheduler"] is None:
        pass
    elif c["scheduler"] == "steplr":
        d("steplr_step_size", 10)
        d("steplr_gamma", 0.1)
    elif c["scheduler"] == "exponentiallr":
        d("exponentiallr_gamma", 0.9)
    elif c["scheduler"] == "cycliclr":
        d("cycliclr_mode", "triangular")
        d("cycliclr_gamma", 1)
    elif c["scheduler"] == "cosineannealinglr":
        d("cosineannealinglr_T_max", 10)
        d("cosineannealinglr_eta_min", 0)
    elif c["scheduler"] == "linearlr":
        d("linearlr_total_iters", 10)
        d("linearlr_start_factor", 1)
        d("linearlr_end_factor", 1)
    else:
        raise ValueError(f"Unknown scheduler: {c['scheduler']}")

    # Model parameters
    # d("m_h_dim", 5)
    # d("m_e_dim", 4)
    if hc != "none":
        d("m_h_outdim", 2)
    d("m_hidden_dim", None)
    if ec in ["default"]:
        d("m_L_ec", 3)
        # d("m_alpha_ec", 0.5)
    if hc != "none":
        d("m_L_hc", 3)
        d("m_alpha_hc", 0.5)
    if ec in ["default", "continued"] and hc != "none":
        d("m_feed_edge_weights", False)


def legacy_config_compatibility(config: dict[str, Any]) -> dict[str, Any]:
    """Preprocess config, for example to deal with legacy configs."""
    rename_keys = {
        "m_alpha_ec_node": "m_alpha",
        "m_use_intermediate_encodings": "m_use_intermediate_edge_embeddings",
        "m_feed_node_attributes": "m_use_node_embedding",
    }
    remove_keys = ["m_alpha_ec_edge", "adam_epsilon"]
    for old, new in rename_keys.items():
        if old in config:
            logger.warning("Renaming key %s to %s", old, new)
            config[new] = config.pop(old)
    for key in remove_keys:
        if key in config:
            logger.warning("Removing key %s", key)
            del config[key]
    return config
