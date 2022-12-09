from __future__ import annotations

from typing import Any

import click
import optuna

from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.trainable import TCNTrainable, suggest_default_values
from gnn_tracking_hpo.tune import common_options, main


class DynamicTCNTrainable(TCNTrainable):
    pass


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
    d("m_h_outdim", 3, 5)
    d("q_min", 0.3, 0.5)
    d("sb", 0.12, 0.135)
    d("lr", 0.0002, 0.0006)
    d("m_hidden_dim", 116)
    d("m_L_ec", 3)
    d("m_L_hc", 3)
    d("m_h_dim", 5, 8)
    d("m_e_dim", 4, 6)
    d("rlw_edge", 1, 10)
    d("m_alpha_ec", 0.3, 0.7)
    d("m_alpha_hc", 0.3, 0.7)
    d("rlw_potential_attractive", 1.0, 30.0)
    d("rlw_potential_repulsive", 2.0, 3.0)

    suggest_default_values(config, trial)
    return config


@click.command()
@common_options
def real_main(**kwargs):
    main(DynamicTCNTrainable, suggest_config, grace_period=4, **kwargs)


if __name__ == "__main__":
    real_main()
