from __future__ import annotations

import json
import pprint
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import gnn_tracking
import optuna
from gnn_tracking.utils.versioning import get_commit_hash

from gnn_tracking_hpo.util.log import logger


def auto_suggest_if_not_fixed(
    key: str, config: dict[str, Any], trial: optuna.Trial, *args, **kwargs
) -> Any:
    """Similar to ``suggest_if_not_fixed``, but automatically chooses the correct
    function.

    **Important**: It matters whether the argument types are ints or floats!
    """
    if key in config:
        pass
    if key in trial.params:
        pass
    if len(args) == 2:
        if all(isinstance(x, int) for x in args):
            return trial.suggest_int(key, *args, **kwargs)
        else:
            return trial.suggest_float(key, *args, **kwargs)
    elif len(args) == 1:
        if isinstance(args[0], list):
            if all(isinstance(x, bool) for x in args[0]):
                # Careful because bools are ints
                pass
            elif all(isinstance(x, int) for x in args[0]):
                ma = max(args[0])
                mi = min(args[0])
                if ma - mi == len(args[0]) - 1:
                    logger.warning(
                        "Substituting suggest_int from %s to %s instead of "
                        "categorical %s",
                        mi,
                        ma,
                        args[0],
                    )
                    return trial.suggest_int(key, mi, ma)
                return trial.suggest_categorical(key, *args, **kwargs)
            return trial.suggest_categorical(key, *args, **kwargs)
        else:
            config[key] = args[0]
            return args[0]
    else:
        raise ValueError("Do not understand specification")


def read_json(path: PathLike | str) -> dict[str, Any]:
    """Open and read a json file"""
    with Path(path).open() as f:
        config = json.load(f)
    return config


def get_metadata(*, test=False):
    return {
        "test": test,
        "gnn_tracking_hash": get_commit_hash(gnn_tracking),
        "gnn_tracking_experiments_hash": get_commit_hash(Path(__file__).parent),
    }


def get_points_to_evaluate(
    paths: None | list[str] | list[PathLike] = None,
) -> list[dict[str, Any]]:
    """Read json files or read from wandb online and return a list of dicts.
    Json files can either contain dictionary (single config) or a list thereof.
    """
    points_to_evaluate: list[dict[str, Any]] = []
    if paths is None:
        paths = list[str]()
    for path in paths:
        if isinstance(path, str) and "/" not in path and not Path(path).exists():
            # Assume it's a wandb hash
            points_to_evaluate.append(retrieve_config_from_wandb(str(path)))
            continue
        obj = read_json(path)
        if isinstance(obj, list):
            points_to_evaluate.extend(obj)
        elif isinstance(obj, dict):
            points_to_evaluate.append(obj)
        else:
            raise ValueError("Decoding of json file failed")
    if points_to_evaluate:
        logger.info("Enqueued trials:\n%s", pprint.pformat(points_to_evaluate))
    return points_to_evaluate


@dataclass
class ServerConfig:
    """Config values for server that we run on"""

    #: Total number of GPUs available per node
    gpus: int = 0
    #: Total number of cpus available per node
    cpus: int = 1
    #: Max batches that we can load into the GPU RAM
    max_batches: int = 1

    @property
    def cpus_per_gpu(self) -> int:
        return self.cpus // self.gpus


della = ServerConfig(gpus=4, cpus=48, max_batches=20)
server = della


def retrieve_config_from_wandb(hash: str) -> dict[str, Any]:
    """Retrieve configuration of run from wandb based on (part of) a hash"""
    import wandb

    logger.debug("Attempting to retrieve config for hash %s from wandb", hash)
    api = wandb.Api()
    run = api.run(f"gnn_tracking/{hash}")
    logger.debug("Obtained run with URL %s from wandb", run.url)
    ignored_keys = [
        "pid",
        "test",
        "node_ip",
        "trial_id",
        "experiment_id",
        "gnn_tracking_experiments_hash",
        "gnn_tracking_hash",
        "hostname",
        "date",
        "trial_log_path",
    ]
    return {
        k: v
        for k, v in run.config.items()
        if not k.startswith("_") and k not in ignored_keys
    }
