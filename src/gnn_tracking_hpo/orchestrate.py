from __future__ import annotations

import os
from http import client as httplib

import ray
from gnn_tracking.utils.log import logger
from ray.util.joblib import register_ray


def have_internet() -> bool:
    """Return True if we have internet connection"""
    conn = httplib.HTTPSConnection("8.8.8.8", timeout=5)
    try:
        conn.request("HEAD", "/")
        return True
    except Exception:
        return False
    finally:
        conn.close()


def maybe_run_wandb_offline() -> None:
    """If we do not have internet connection, run wandb in offline mode"""
    if not have_internet():
        logger.warning("Setting wandb mode to offline because you do not have internet")
        os.environ["WANDB_MODE"] = "offline"
    else:
        logger.debug("You seem to have internet, so directly syncing with wandb")


def maybe_run_distributed() -> None:
    """If it looks like we're running across multiple nodes, enable distributed
    mode of ray
    """
    if "redis_password" in os.environ:
        # We're running distributed
        ray.init(address="auto", _redis_password=os.environ["redis_password"])
        register_ray()
