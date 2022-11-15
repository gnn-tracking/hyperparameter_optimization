from __future__ import annotations

from typing import Any

import click
import optuna
from gnn_tracking.training.dynamiclossweights import NormalizeAt
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped

from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.trainable import TCNTrainable, suggest_default_values
from gnn_tracking_hpo.tune import common_options, main


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
