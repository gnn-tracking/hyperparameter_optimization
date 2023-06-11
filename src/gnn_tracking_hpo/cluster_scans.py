from __future__ import annotations

from typing import Any

import numpy as np
from gnn_tracking.metrics.cluster_metrics import common_metrics
from gnn_tracking.postprocessing.clusterscanner import ClusterScanResult
from gnn_tracking.postprocessing.dbscanscanner import DBSCANHyperParamScanner

from gnn_tracking_hpo.util.log import logger


def fixed_dbscan_scan(
    graphs: np.ndarray,
    truth: np.ndarray,
    sectors: np.ndarray,
    pts: np.ndarray,
    reconstructable: np.ndarray,
    *,
    guide="trk.double_majority_pt0.9",
    epoch=None,
    start_params: dict[str, Any] | None = None,
) -> ClusterScanResult:
    """Convenience function for not scanning for DBSCAN hyperparameters at all."""
    if start_params is None:
        start_params = {
            "eps": 0.95,
            "min_samples": 1,
        }
    dbss = DBSCANHyperParamScanner(
        graphs=graphs,
        truth=truth,
        sectors=sectors,
        pts=pts,
        reconstructable=reconstructable,
        guide=guide,
        metrics=common_metrics,
    )
    return dbss.scan(
        n_jobs=1,
        n_trials=1,
        start_params=start_params,
    )


def reduced_dbscan_scan(
    graphs: list[np.ndarray],
    truth: list[np.ndarray],
    sectors: list[np.ndarray],
    pts: list[np.ndarray],
    reconstructable: list[np.ndarray],
    *,
    guide="trk.double_majority_pt0.9",
    epoch=None,
    start_params: dict[str, Any] | None = None,
    node_mask: list[np.ndarray] | None = None,
) -> ClusterScanResult:
    """Convenience function for scanning DBSCAN hyperparameters with trial count
    that depends on the epoch (using many trials early on, then alternating between
    fixed and low samples in later epochs).
    """
    version_dependent_kwargs = {}
    if node_mask is not None:
        logger.warning("Running on a gnn_tracking version with post-EC node pruning.")
        version_dependent_kwargs["node_mask"] = node_mask
    dbss = DBSCANHyperParamScanner(
        data=graphs,
        truth=truth,
        sectors=sectors,
        pts=pts,
        reconstructable=reconstructable,
        guide=guide,
        metrics=common_metrics,
        min_samples_range=(1, 1),
        eps_range=(0.2, 1.0),
        **version_dependent_kwargs,
    )
    if epoch < 5:
        n_trials = 6
    elif epoch % 4 == 0:
        n_trials = 6
    else:
        n_trials = 1
    return dbss.scan(
        n_jobs=min(12, n_trials),  # todo: make flexible
        n_trials=n_trials,
        start_params=start_params,
    )
