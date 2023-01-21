from __future__ import annotations

import os
from http import client as httplib
from pathlib import Path

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

    def try_get_from_file(path: Path) -> str:
        try:
            return path.read_text().strip()
        except FileNotFoundError:
            return ""

    redis_password = os.environ.get("redis_password", "").strip() or try_get_from_file(
        Path.home() / ".ray_head_redis_password"
    )
    head_ip = os.environ.get("head_ip", "").strip() or try_get_from_file(
        Path.home() / ".ray_head_ip_address"
    )
    if redis_password and head_ip:
        logger.info(
            f"Connecting to ray head at {head_ip} with password {redis_password}"
        )
        ray.init(
            address="auto",
            _redis_password=os.environ["redis_password"],
            _node_ip_address=os.environ["head_ip"].split(":")[0],
        )
        register_ray()
