from __future__ import annotations

import json
import pprint
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Callable

import gnn_tracking
from gnn_tracking.utils.log import logger
from gnn_tracking.utils.versioning import get_commit_hash


def suggest_if_not_fixed(
    f: Callable, key: str, config: dict[str, Any], *args, **kwargs
):
    """Call function with arguments if ``key`` is not in ``config``"""
    if key not in config:
        f(key, *args, **kwargs)


def read_json(path: PathLike | str) -> dict[str, Any]:
    """Open and read a json file"""
    with Path(path).open() as f:
        config = json.load(f)
    return config


def get_fixed_config(*, test=False):
    return {
        "test": test,
        "max_batches": 1 if test else None,
        "gnn_tracking_hash": get_commit_hash(gnn_tracking),
        "gnn_tracking_experiments_hash": get_commit_hash(Path(__file__).parent),
    }


def get_points_to_evaluate(
    paths: None | list[str] | list[PathLike] = None,
) -> list[dict[str, Any]]:
    """Read json files and return a list of dicts.
    Json files can either contain dictionary (single config) or a list thereof.
    """
    points_to_evaluate: list[dict[str, Any]] = []
    if paths is None:
        paths = list[str]()
    for path in paths:
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
