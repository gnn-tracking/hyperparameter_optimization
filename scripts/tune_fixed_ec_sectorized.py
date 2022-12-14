from __future__ import annotations

from functools import partial
from typing import Any

import click
import optuna
from gnn_tracking.models.track_condensation_networks import PreTrainedECGraphTCN
from gnn_tracking.training.tcn_trainer import TCNTrainer
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped
from torch import nn

from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.trainable import TCNTrainable, suggest_default_values
from gnn_tracking_hpo.tune import common_options, main
from gnn_tracking_hpo.util.paths import add_scripts_path, find_checkpoints, get_config

add_scripts_path()
from tune_ec_sectorized import ECTrainable  # noqa: E402


class UnPackDictionaryForward(nn.Module):
    def __init__(self, ec):
        super().__init__()
        self.ec = ec

    def forward(self, *args, **kwargs):
        return self.ec(*args, **kwargs)["W"]


def load_ec(project: str, hash: str, *, config_update: dict | None = None) -> nn.Module:
    checkpoint_path = find_checkpoints(project, hash)[-1]
    config = get_config(project, hash)
    # In case any new values were added, we need to suggest this again
    suggest_default_values(config, None, ec="default", hc="none")
    config.update({"n_graphs_train": 1, "n_graphs_val": 1, "n_graphs_test": 1})
    if config_update is not None:
        config.update(config_update)
    trainable = ECTrainable(config)
    trainable.load_checkpoint(checkpoint_path, device="cpu")
    ec = trainable.trainer.model
    for param in ec.parameters():
        param.requires_grad = False
    return UnPackDictionaryForward(ec)


class PretrainedECTrainable(TCNTrainable):
    def __init__(self, config: dict[str, Any], **kwargs):
        self.ec = load_ec(config["ec_project"], config["ec_hash"])
        super().__init__(config=config, **kwargs)

    def get_loss_functions(self) -> dict[str, Any]:
        return {
            "potential": self.get_potential_loss_function(),
            "background": self.get_background_loss_function(),
        }

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
        trainer.ec_threshold = self.tc["m_ec_threshold"]

        return trainer

    def get_model(self) -> nn.Module:
        return PreTrainedECGraphTCN(
            self.ec,
            node_indim=6,
            edge_indim=4,
            **subdict_with_prefix_stripped(self.tc, "m_"),
        )


def suggest_config(
    trial: optuna.Trial,
    *,
    sector: int,
    ec_project: str,
    ec_hash: str,
    test=False,
    fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = get_metadata(test=test)
    config.update(fixed or {})

    def d(key, *args, **kwargs):
        auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

    d("sector", sector)
    d("n_graphs_train", 300)
    d("n_graphs_val", 69)
    d("n_graphs_test", 1)

    d("m_mask_nodes_with_leq_connections", 2)

    d("ec_project", ec_project)
    d("ec_hash", ec_hash)
    d("m_ec_threshold", 0.3746)

    d("batch_size", 1)
    d("attr_pt_thld", 0.0, 0.9)
    d("m_h_outdim", 2, 5)
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
    d("m_interaction_node_hidden_dim", 32, 128)
    d("m_interaction_edge_hidden_dim", 32, 128)

    suggest_default_values(config, trial, ec="fixed")
    return config


@click.command()
@click.option(
    "--ec-hash", required=True, type=str, help="Hash of the edge classifier to load"
)
@click.option(
    "--ec-project",
    required=True,
    type=str,
    help="Name of the jfolder that the edge classifier to load belongs to",
)
@common_options
def real_main(ec_hash: str, ec_project: str, **kwargs):
    main(
        PretrainedECTrainable,
        partial(
            suggest_config,
            sector=9,
            ec_hash=ec_hash,
            ec_project=ec_project,
        ),
        grace_period=11,
        no_improvement_patience=19,
        metric="trk.double_majority_pt0.9",
        **kwargs,
    )


if __name__ == "__main__":
    real_main()
