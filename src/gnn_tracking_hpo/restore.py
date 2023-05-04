from __future__ import annotations

from torch import nn

from gnn_tracking_hpo.trainable import (
    legacy_config_compatibility,
    suggest_default_values,
)
from gnn_tracking_hpo.util.log import logger
from gnn_tracking_hpo.util.paths import find_checkpoint, get_config


def restore_model(
    trainable_cls,
    tune_dir: str,
    run_hash: str,
    epoch: int = -1,
    *,
    config_update: dict | None = None,
    freeze: bool = True,
) -> nn.Module:
    """Load pre-trained edge classifier

    Args:
        tune_dir (str): Name of ray tune outptu directory
        run_hash (str): Hash of the run
        epoch (int, optional): Epoch to load. Defaults to -1 (last epoch).
        config_update (dict, optional): Update the config with this dict.
    """
    logger.info("Initializing pre-trained model")
    checkpoint_path = find_checkpoint(tune_dir, f"_{run_hash}_", epoch)
    config = legacy_config_compatibility(get_config(tune_dir, f"_{run_hash}_"))
    # In case any new values were added, we need to suggest this again
    suggest_default_values(config, None, ec="default", hc="none")
    if config_update is not None:
        config.update(config_update)
    config["_no_data"] = True
    trainable = trainable_cls(config)
    trainable.load_checkpoint(checkpoint_path)
    model = trainable.trainer.model
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    logger.info("Pre-trained model initialized")
    return model
