from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna

from gnn_tracking_hpo.config import get_metadata
from gnn_tracking_hpo.trainable import TCNTrainable, suggest_default_values
from gnn_tracking_hpo.tune import main

DATA_DIR = Path(".").expanduser().parent / "test_data"


def suggest_config(trial: optuna.Trial, *args, **kwargs) -> dict[str, Any]:
    config = get_metadata(test=True)
    suggest_default_values(config, trial)
    return config


def test_tune():
    main(
        TCNTrainable,
        suggest_config,
        test=True,
    )
