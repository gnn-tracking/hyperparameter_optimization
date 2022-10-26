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


def reduced_dbscan_scan(
    graphs: np.ndarray,
    truth: np.ndarray,
    sectors: np.ndarray,
    *,
    guide="v_measure",
    epoch=None,
    start_params: dict[str, Any] | None = None,
) -> ClusterScanResult:
    dbss = DBSCANHyperParamScanner(
        graphs=graphs,
        truth=truth,
        sectors=sectors,
        guide=guide,
        metrics=common_metrics,
        min_samples_range=(1, 1),
    )
    n_trials = 20
    if epoch > 3 and epoch % 2 != 0 or epoch > 7 and epoch % 4 != 0:
        logger.debug("Skipping scanning over DBSCAN parameters")
        n_trials = 1
    return dbss.scan(
        n_jobs=12,  # todo: make flexible
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
        return auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

    d("optimizer", "sgd")
    d("sgd_momentum", 0.8, 0.99)
    scheduler = d("scheduler", ["steplr", "exponentiallr"])
    if scheduler == "steplr":
        d("steplr_step_size", 2, 5)
        d("steplr_gamma", 0.02, 0.5)
    elif scheduler == "exponentiallr":
        d("exponentiallr_gamma", 0.8, 0.999)
    else:
        raise ValueError("Invalid scheduler")
    d("batch_size", 1)
    d("attr_pt_thld", 0.0)
    d("m_feed_edge_weights", True)
    d("m_h_outdim", 2)
    d("q_min", 0.402200635027302)
    d("sb", 0.1237745028815143)
    d("lr", 1e-4, 9e-4)
    d("m_hidden_dim", 116)
    d("m_L_ec", 3)
    d("m_L_hc", 3)
    d("rlw_edge", 9.724314205420344)
    d("rlw_potential_attractive", 9.889861321497472)
    d("rlw_potential_repulsive", 2.1784381633400933)

    suggest_default_values(config, trial)
    return config


@click.command()
@common_options
def real_main(**kwargs):
    main(DynamicTCNTrainable, suggest_config, grace_period=4, **kwargs)


if __name__ == "__main__":
    real_main()
