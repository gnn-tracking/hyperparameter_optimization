#!/usr/bin/env python3

"""Population based training"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gnn_tracking
import optuna
from gnn_tracking.utils.versioning import get_commit_hash
from util import suggest_if_not_fixed


def suggest_config(
    trial: optuna.Trial, *, test=False, fixed: dict[str, Any] | None = None
) -> dict[str, Any]:
    # Everything with prefix "m_" is passed to the model
    # Everything with prefix "lw_" is treated as loss weight kwarg
    fixed_config = {
        "test": test,
        "gnn_tracking_hash": get_commit_hash(gnn_tracking),
        "gnn_tracking_experiments_hash": get_commit_hash(Path(__file__).parent),
    }
    if fixed is not None:
        fixed_config.update(fixed)

    def sinf_float(key, *args, **kwargs):
        suggest_if_not_fixed(trial.suggest_float, key, fixed_config, *args, **kwargs)

    def sinf_int(key, *args, **kwargs):
        suggest_if_not_fixed(trial.suggest_int, key, fixed_config, *args, **kwargs)

    sinf_float("q_min", 1e-3, 1, log=True)
    sinf_float("sb", 0, 1)
    sinf_float("lr", 2e-6, 1e-3, log=True)
    sinf_float("focal_gamma", 0, 20)  # 5 might be a good default
    sinf_float("focal_alpha", 0, 1)  # 0.95 might be a good default
    sinf_float("lw_edge", 0.001, 500)
    sinf_float("lw_potential_attractive", 1, 500)
    sinf_float("lw_potential_repulsive", 1e-2, 1e2)
    return fixed_config
