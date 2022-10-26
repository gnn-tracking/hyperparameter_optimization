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
            at=[0, 1],
            relative_weights=relative_weights,
        )


def suggest_config(
    trial: optuna.Trial, *, test=False, fixed: dict[str, Any] | None = None
) -> dict[str, Any]:
    config = get_metadata(test=test)
    config.update(fixed or {})

    def d(key, *args, **kwargs):
        auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

    d("batch_size", 1)
    d("attr_pt_thld", [0.0, 0.4, 0.9])
    d("m_feed_edge_weights", True)
    d("m_h_outdim", [4])
    d("q_min", 0.3, 1)
    d("sb", 0.12, 0.135)
    d("lr", 0.0001, 0.0005)
    d("m_hidden_dim", 116)
    d("m_L_ec", 3)
    d("m_L_hc", 3)
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
