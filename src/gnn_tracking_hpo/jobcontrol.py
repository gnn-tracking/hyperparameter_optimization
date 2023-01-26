from __future__ import annotations

import os
import subprocess
import time
from enum import Enum, auto
from pathlib import Path

import yaml

from gnn_tracking_hpo.util.log import logger


class JobControlAction(Enum):
    KILL_NODE = auto()
    WAIT = auto()


def get_slurm_job_id() -> str:
    return os.environ["SLURM_JOB_ID"]


def kill_slurm_job(
    job_id: str,
):
    logger.info("Killing slurm job %s", job_id)
    subprocess.run(f"scancel {job_id}")


class JobControl:
    def __init__(self):
        self.job_control_path = Path.home() / "ray_job_control.yaml"
        self.config = []

    def refresh(self):
        if not self.job_control_path.exists():
            logger.warning("Job control file %s does not exist", self.job_control_path)
            return
        logger.debug("Refreshing job control config from %s", self.job_control_path)
        with open(self.job_control_path) as f:
            try:
                self.config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.error("Could not load job control config: %s", e)
                return

    def _get_actions(self, ip: str, dispatcher_id: str) -> list[JobControlAction]:
        actions = []
        for c in self.config:
            if c.get("ip") is not None and str(c.get("ip")) != str(ip):
                continue
            if c.get("dispatcher_id") is not None and str(
                c.get("dispatcher_id")
            ) != str(dispatcher_id):
                continue
            actions.append(JobControlAction[c["action"].upper()])
        if actions:
            logger.debug("Got actions %s", actions)
        return actions

    @staticmethod
    def _handle_action(action: JobControlAction) -> bool:
        if action == JobControlAction.WAIT:
            logger.info("Sleeping for 30s")
            time.sleep(30)
            return False
        if action == JobControlAction.KILL_NODE:
            job_id = get_slurm_job_id()
            kill_slurm_job(job_id)
            return False
        raise ValueError(f"Unknown action {action}")

    def __call__(self, ip: str, dispatcher_id: str) -> None:
        repeat_requested = True
        while repeat_requested:
            self.refresh()
            repeat_requested = False
            try:
                actions = self._get_actions(ip, dispatcher_id)
            except KeyError as e:
                logger.error("Could not get actions, please check your config: %s", e)
                return
            for action in actions:
                repeat_requested |= self._handle_action(action)
