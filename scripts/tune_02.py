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

from gte.tune import common_options, main
from gte.util import TCNTrainable, get_fixed_config, suggest_if_not_fixed


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
    fixed_config = get_fixed_config(test=test)
    if fixed is not None:
        fixed_config.update(fixed)

    def sinf_float(key, *args, **kwargs):
        suggest_if_not_fixed(trial.suggest_float, key, fixed_config, *args, **kwargs)

    def sinf_choice(key, *args, **kwargs):
        suggest_if_not_fixed(
            trial.suggest_categorical, key, fixed_config, *args, **kwargs
        )

    def dinf(key, value):
        if key not in fixed_config:
            fixed_config[key] = value

    # def sinf_int(key, *args, **kwargs):
    #     suggest_if_not_fixed(trial.suggest_int, key, fixed_config, *args, **kwargs)

    dinf("batch_size", 1)
    # sinf_choice("attr_pt_thld", [0.0, 0.9])
    dinf("attr_pt_thld", 0.0)
    # sinf_choice("m_feed_edge_weights", [True, False])
    dinf("m_feed_edge_weights", True)
    sinf_choice("m_h_outdim", [2, 3, 4])
    sinf_float("q_min", 0.3, 0.5)
    # dinf("q_min", 0.4220881041839594)
    sinf_float("sb", 0.12, 0.135)
    # dinf("sb", 0.14219587966015457)
    sinf_float("lr", 0.0003, 0.0004)
    # dinf("lr", 0.0003640386078772556)
    # sinf_int("m_hidden_dim", 64, 256)
    dinf("m_hidden_dim", 116)
    # sinf_int("m_L_ec", 1, 7)
    dinf("m_L_ec", 3)
    # sinf_int("m_L_hc", 1, 7)
    dinf("m_L_hc", 3)
    # sinf_float("focal_gamma", 0, 20)  # 5 might be a good default
    # sinf_float("focal_alpha", 0, 1)  # 0.95 might be a good default
    sinf_float("rlw_edge", 1, 10)
    sinf_float("rlw_potential_attractive", 1, 10)
    sinf_float("rlw_potential_repulsive", 2, 3)
    return fixed_config


@click.command()
@common_options
def real_main(**kwargs):
    main(DynamicTCNTrainable, suggest_config, grace_period=4, **kwargs)


if __name__ == "__main__":
    real_main()
