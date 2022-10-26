from __future__ import annotations

from typing import Any

import click
import numpy as np
import optuna
from gnn_tracking.postprocessing.cluster_metrics import common_metrics
from gnn_tracking.postprocessing.clusterscanner import ClusterScanResult
from gnn_tracking.postprocessing.dbscanscanner import DBSCANHyperParamScanner
from gnn_tracking.training.dynamiclossweights import NormalizeAt
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped
from gnn_tracking.utils.log import logger

from gte.config import auto_suggest_if_not_fixed, get_metadata
from gte.trainable import TCNTrainable, suggest_default_values
from gte.tune import common_options, main


def dbscan_scan(
    graphs: np.ndarray,
    truth: np.ndarray,
    sectors: np.ndarray,
    *,
    n_jobs=1,
    n_trials=30,
    guide="v_measure",
    epoch=None,
    start_params: dict[str, Any] | None = None,
) -> ClusterScanResult:
    if n_jobs == 1:
        logger.warning("Only using 1 thread for DBSCAN scan")
    dbss = DBSCANHyperParamScanner(
        graphs=graphs,
        truth=truth,
        sectors=sectors,
        guide=guide,
        metrics=common_metrics,
        min_samples_range=(1, 1),
    )
    return dbss.scan(
        n_jobs=n_jobs,
        n_trials=n_trials,
        start_params=start_params,
    )


class DynamicTCNTrainable(TCNTrainable):
    def get_loss_weights(self):
        relative_weights = [
            {
                "edge": 10,
            },
            subdict_with_prefix_stripped(self.tc, "rlw_"),
        ]
        return NormalizeAt(
            at=[0, 2],
            relative_weights=relative_weights,
        )


def suggest_config(
    trial: optuna.Trial, *, test=False, fixed: dict[str, Any] | None = None
) -> dict[str, Any]:
    # Everything with prefix "m_" is passed to the model
    # Everything with prefix "lw_" is treated as loss weight kwarg
    config = get_metadata(test=test)
    config.update(fixed or {})

    def d(key, *args, **kwargs):
        auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

    d("batch_size", 1)
    # sinf_choice("attr_pt_thld", [0.0, 0.9])
    d("attr_pt_thld", 0.0)
    # sinf_choice("m_feed_edge_weights", [True, False])
    d("m_feed_edge_weights", True)
    d("m_h_outdim", [2, 3, 4])
    d("q_min", 0.3, 0.5)
    # dinf("q_min", 0.4220881041839594)
    d("sb", 0.12, 0.135)
    # dinf("sb", 0.14219587966015457)
    d("lr", 0.0003, 0.0004)
    # dinf("lr", 0.0003640386078772556)
    # sinf_int("m_hidden_dim", 64, 256)
    d("m_hidden_dim", 116)
    # sinf_int("m_L_ec", 1, 7)
    d("m_L_ec", 3)
    # sinf_int("m_L_hc", 1, 7)
    d("m_L_hc", 3)
    # sinf_float("focal_gamma", 0, 20)  # 5 might be a good default
    # sinf_float("focal_alpha", 0, 1)  # 0.95 might be a good default
    d("rlw_edge", 1, 10)
    d("rlw_potential_attractive", 1, 10)
    d("rlw_potential_repulsive", 2, 3)

    suggest_default_values(config, trial)
    return config


@click.command()
@common_options
def real_main(**kwargs):
    main(DynamicTCNTrainable, suggest_config, grace_period=4, **kwargs)


if __name__ == "__main__":
    real_main()
