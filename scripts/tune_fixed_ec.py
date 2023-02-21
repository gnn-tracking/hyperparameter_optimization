from __future__ import annotations

from argparse import ArgumentParser
from functools import partial
from typing import Any

import optuna
from gnn_tracking.models.track_condensation_networks import PreTrainedECGraphTCN
from gnn_tracking.training.tcn_trainer import TCNTrainer
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped
from ray.tune.schedulers import ASHAScheduler
from rt_stoppers_contrib.threshold_by_epoch import ThresholdTrialStopper
from torch import nn

from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.trainable import TCNTrainable, suggest_default_values
from gnn_tracking_hpo.tune import Dispatcher, add_common_options
from gnn_tracking_hpo.util.log import logger
from gnn_tracking_hpo.util.paths import add_scripts_path, find_checkpoint, get_config

add_scripts_path()
from tune_ec import ECTrainable  # noqa: E402


class UnPackDictionaryForward(nn.Module):
    def __init__(self, ec: nn.Module):
        """Small wrapper class that basically undos what the
        `SignatureAdaptedECForGraphTCN` class that was used to adapt the signature of
        the edge classifier for training with the normal TCN trainer does.
        """
        super().__init__()
        self.ec = ec

    def forward(self, *args, **kwargs):
        return self.ec(*args, **kwargs)["W"]


def load_ec(
    tune_dir: str,
    run_hash: str,
    epoch: int = -1,
    *,
    config_update: dict | None = None,
    device=None,
) -> nn.Module:
    """Load pre-trained edge classifier

    Args:
        tune_dir (str): Name of ray tune outptu directory
        run_hash (str): Hash of the run
        epoch (int, optional): Epoch to load. Defaults to -1 (last epoch).
        config_update (dict, optional): Update the config with this dict.
    """
    logger.info("Initializing pre-trained EC")
    checkpoint_path = find_checkpoint(tune_dir, run_hash, epoch)
    config = get_config(tune_dir, run_hash)
    # In case any new values were added, we need to suggest this again
    suggest_default_values(config, None, ec="default", hc="none")
    config.update({"n_graphs_train": 1, "n_graphs_val": 1, "n_graphs_test": 1})
    if config_update is not None:
        config.update(config_update)
    trainable = ECTrainable(config)
    trainable.load_checkpoint(checkpoint_path)
    ec = trainable.trainer.model
    for param in ec.parameters():
        param.requires_grad = False
    logger.info("Pre-trained EC initialized")
    return UnPackDictionaryForward(ec)


class PretrainedECTrainable(TCNTrainable):
    def __init__(self, config: dict[str, Any], **kwargs):
        self.ec = load_ec(
            config["ec_project"],
            config["ec_hash"],
            config["ec_epoch"],
        )
        super().__init__(config=config, **kwargs)

    def get_loss_functions(self) -> dict[str, Any]:
        return {
            "potential": self.get_potential_loss_function(),
            "background": self.get_background_loss_function(),
        }

    def get_trainer(self) -> TCNTrainer:
        trainer = super().get_trainer()
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
    sector: int | None = None,
    ec_project: str,
    ec_hash: str,
    ec_epoch: int = -1,
    test=False,
    fixed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = get_metadata(test=test)
    config.update(fixed or {})

    def d(key, *args, **kwargs):
        auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

    # Definitely Fixed hyperparameters
    # --------------------------------

    d("sector", sector)
    d("n_graphs_train", 12379)
    d("n_graphs_val", 100)
    d("n_graphs_test", 1)

    d("m_mask_orphan_nodes", True)

    d("ec_project", ec_project)
    d("ec_hash", ec_hash)
    d("ec_epoch", ec_epoch)

    # >>>>>>>>>>>>> IMPORTANT, adjust that based on the EC <<<<<<<
    d("m_ec_threshold", 0.1, 0.5)

    d("batch_size", 5)

    # Keep one fixed because of normalization invariance
    d("lw_potential_attractive", 1.0)

    # Tuned hyperparameters
    # ---------------------

    d("attr_pt_thld", 0.0, 0.9)
    d("m_h_outdim", 2, 8)
    d("q_min", 0.3, 0.5)
    d("sb", 0.05, 0.12)
    d("lr", 0.0001, 0.0006)
    d("repulsive_radius_threshold", 1.5, 10)
    d("m_hidden_dim", 128, 256)
    d("m_L_hc", 3, 7)
    d("m_h_dim", 5, 8)
    d("m_e_dim", 4, 6)
    d("m_alpha_hc", 0.3, 0.99)
    d("lw_background", 1e-6, 1e-1, log=True)
    d("lw_potential_repulsive", 1e-1, 1e1, log=True)
    d("m_interaction_node_hidden_dim", 32, 128)
    d("m_interaction_edge_hidden_dim", 32, 128)

    suggest_default_values(config, trial, ec="fixed")
    return config


class ThisDispatcher(Dispatcher):
    def get_scheduler(self) -> None | ASHAScheduler:
        if self.no_scheduler:
            # FIFO scheduler
            return None
        return ASHAScheduler(
            metric=self.metric,
            mode="max",
            grace_period=self.grace_period,
            reduction_factor=2,
            stop_last_trials=False,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ec-hash", required=True, type=str, help="Hash of the edge classifier to load"
    )
    parser.add_argument(
        "--ec-project",
        required=True,
        type=str,
        help="Name of the folder that the edge classifier to load belongs to",
    )
    parser.add_argument(
        "--ec-epoch",
        type=int,
        default=-1,
        help="Epoch of the edge classifier to load. Defaults to -1 (last epoch).",
    )
    add_common_options(parser)
    kwargs = vars(parser.parse_args())
    if kwargs["cpu"]:
        PretrainedECTrainable._device = "cpu"
    this_suggest_config = partial(
        suggest_config,
        ec_hash=kwargs.pop("ec_hash"),
        ec_project=kwargs.pop("ec_project"),
        ec_epoch=kwargs.pop("ec_epoch"),
    )
    dispatcher = ThisDispatcher(
        grace_period=2,
        no_improvement_patience=6,
        metric="trk.double_majority_pt0.9",
        additional_stoppers=[
            ThresholdTrialStopper(
                "trk.double_majority_pt0.9", {5: 0.5, 10: 0.63, 15: 0.65}
            )
        ],
        **kwargs,
    )
    dispatcher(
        PretrainedECTrainable,
        this_suggest_config,
    )
