from __future__ import annotations

import os
from http import client as httplib
from pathlib import Path

import ray
from ray.util.joblib import register_ray

from gnn_tracking_hpo.util.log import logger


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
        logger.debug("You seem to have internet, so setting wandb to online")


def maybe_run_distributed(local=False, **kwargs) -> None:
    """If it looks like we're running across multiple nodes, enable distributed
    mode of ray

    Args:
        local: Force not to connect distributed
        kwargs: Additional kwargs to pass to ray.init
    """
    # Disable deduplication of logs
    os.environ["RAY_DEDUP_LOGS"] = "0"
    if local:
        if "num_cpus" not in kwargs and "num_gpus" not in kwargs:
            logger.warning(
                "Neither num_cpus nor num_gpus specified, so ray will use all available"
                " resources. This might be a bad idea if you're running on a shared"
                " machine. "
            )
        logger.debug("Running in local mode, so not attempting to connect to ray head")
        ray.init(address="local", **kwargs)
        return
    else:
        # Mustn't specify num cpus/gpus when connecting to ray head
        kwargs.pop("num_cpus", None)
        kwargs.pop("num_gpus", None)

    def get_from_file_or_environ(name: str, path: Path, env_name: str) -> str | None:
        from_env = os.environ.get(env_name)
        if from_env is not None:
            logger.debug(
                "Got %s = %s from environment var '%s'", name, from_env, env_name
            )
            return from_env
        try:
            from_file = path.read_text().strip()
        except FileNotFoundError:
            logger.debug(
                "Could not get %s from file %s or environment var %s",
                name,
                path,
                env_name,
            )
            return None
        logger.debug("Got %s = %s from file '%s'", name, from_file, path)
        return from_file

    head_ip = get_from_file_or_environ(
        "Head IP", Path.home() / ".ray_head_ip_address", "head_ip"
    )

    if head_ip:
        logger.info(f"Connecting to ray head at {head_ip}")
        ray.init(
            address=head_ip,
            _node_ip_address=head_ip.split(":")[0],
            **kwargs,
        )
        register_ray()
    else:
        logger.info("Not connecting to ray head because head ip not found")
