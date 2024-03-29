from __future__ import annotations

from argparse import ArgumentParser
from functools import partial
from typing import Any

import optuna
from gnn_tracking.models.track_condensation_networks import PerfectECGraphTCN
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped
from torch import nn

from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.defaults import suggest_default_values
from gnn_tracking_hpo.trainable import DefaultTrainable
from gnn_tracking_hpo.tune import add_common_options, main


class PerfectECTCNTrainable(DefaultTrainable):
    def get_loss_functions(self) -> dict[str, Any]:
        return {
            "potential": self.get_potential_loss_function(),
            "background": self.get_background_loss_function(),
        }

    def get_model(self) -> nn.Module:
        return PerfectECGraphTCN(
            node_indim=6, edge_indim=4, **subdict_with_prefix_stripped(self.tc, "m_")
        )


def suggest_config(
    trial: optuna.Trial,
    *,
    test=False,
    fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = get_metadata(test=test)
    config.update(fixed or {})

    def d(key, *args, **kwargs):
        auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

    d("batch_size", 1)
    d("attr_pt_thld", 0.0, 0.9)
    d("m_h_outdim", 3, 5)
    d("q_min", 0.3, 0.5)
    d("sb", 0.12, 0.135)
    d("lr", 0.0002, 0.0006)
    d("m_hidden_dim", 116)
    d("m_L_hc", 3)
    d("m_h_dim", 5, 8)
    d("m_e_dim", 4, 6)
    d("m_alpha_hc", 0.3, 0.7)
    # fixme: Don't use RLW anymore
    d("rlw_background", 1.0)
    d("rlw_potential_attractive", 1.0)
    d("rlw_potential_repulsive", 2.0, 3.0)

    suggest_default_values(config, trial, ec="perfect")
    return config


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_options(parser)
    args = parser.parse_args()
    kwargs = vars(args)
    main(
        PerfectECTCNTrainable,
        partial(suggest_config, sector=kwargs.pop("sector")),
        grace_period=4,
        **kwargs,
    )
