from __future__ import annotations

from argparse import ArgumentParser
from typing import Any

import optuna

from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.trainable import TCNTrainable, suggest_default_values
from gnn_tracking_hpo.tune import add_common_options, main


def suggest_config(
    trial: optuna.Trial, *, test=False, fixed: dict[str, Any] | None = None
) -> dict[str, Any]:
    config = get_metadata(test=test)
    config.update(fixed or {})

    def d(key, *args, **kwargs):
        auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

    d("q_min", 1e-3, 1, log=True)
    d("sb", 0, 1)
    d("lr", 2e-6, 1e-3, log=True)
    d("m_hidden_dim", 64, 256)
    d("m_L_ec", 1, 7)
    d("m_L_hc", 1, 7)
    d("focal_gamma", 0, 20)  # 5 might be a good default
    d("focal_alpha", 0, 1)  # 0.95 might be a good default
    d("lw_edge", 0.001, 500)
    d("lw_potential_attractive", 1, 500)
    d("lw_potential_repulsive", 1e-2, 1e2)

    suggest_default_values(config, trial)
    return config


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_options(parser)
    args = parser.parse_args()
    main(TCNTrainable, suggest_config, **vars(args))
