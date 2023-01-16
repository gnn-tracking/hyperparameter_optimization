from __future__ import annotations

from argparse import ArgumentParser
from functools import partial
from typing import Any

import optuna
from gnn_tracking.training.tcn_trainer import TCNTrainer

from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.trainable import TCNTrainable, suggest_default_values
from gnn_tracking_hpo.tune import add_common_options, main


class ThisTrainable(TCNTrainable):
    def get_trainer(self) -> TCNTrainer:
        trainer = super().get_trainer()

        def test_every(*args, **kwargs):
            if trainer._epoch % 9 == 1:
                # note: first epoch we test is epoch 1
                trainer.last_test_result = TCNTrainer.test_step(
                    trainer, *args, **kwargs
                )
            return trainer.last_test_result

        trainer.test_step = test_every

        return trainer


def suggest_config(
    trial: optuna.Trial,
    *,
    test=False,
    fixed: dict[str, Any] | None = None,
    sector: int | None = None,
    truth_cut=False,
) -> dict[str, Any]:
    config = get_metadata(test=test)
    config.update(fixed or {})

    def d(key, *args, **kwargs):
        auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

    assert sector is not None
    d("sector", sector)
    d("n_graphs_train", 300)
    d("n_graphs_val", 69)
    d("n_graphs_test", 1)
    d("batch_size", 1)
    d("focal_gamma", 0, 10)  # 5 might be a good default
    d("focal_alpha", 0.8, 1)  # 0.95 might be a good default
    d("attr_pt_thld", 0.0, 0.9)
    d("m_h_outdim", 3)
    d("q_min", 0.3, 0.5)
    d("sb", 0.05, 0.12)
    d("lr", 0.0001, 0.0006)
    d("repulsive_radius_threshold", 1.5, 10)
    d("m_hidden_dim", 116)
    d("m_L_hc", 3)
    d("m_h_dim", 5, 8)
    d("m_e_dim", 4, 6)
    d("m_alpha_hc", 0.3, 0.99)
    # Keep one fixed because of normalization invariance
    d("lw_potential_attractive", 1.0)
    d("lw_background", 1e-6, 1e-1, log=True)
    d("lw_potential_repulsive", 1e-1, 1e1, log=True)
    d("lw_edge", 1e-3, 1e3, log=True)
    d("m_interaction_node_hidden_dim", 128)
    d("m_interaction_edge_hidden_dim", 128)

    suggest_default_values(config, trial)
    return config


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_options(parser)
    parser.add_argument("--sector", type=int, required=True)
    kwargs = vars(parser.parse_args())
    main(
        ThisTrainable,
        partial(
            suggest_config,
            sector=kwargs.pop("sector"),
        ),
        grace_period=11,
        no_improvement_patience=19,
        metric="trk.double_majority_pt1.5",
        # thresholds={
        #     40: 0.6,
        #     60: 0.8,
        #     70: 0.9,
        # },
        **kwargs,
    )
