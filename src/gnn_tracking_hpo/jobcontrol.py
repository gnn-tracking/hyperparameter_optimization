from __future__ import annotations

import datetime
import os
import subprocess
import time
from enum import Enum, auto
from pathlib import Path

import yaml

from gnn_tracking_hpo.util.log import get_logger


class JobControlAction(Enum):
    """Actions for job control to take"""

    KILL_NODE = auto()
    WAIT = auto()


def get_slurm_job_id() -> str:
    """Get SLURM job ID from environment variables. An empty string is
    returned if the job is not running on SLURM."""
    return os.environ.get("SLURM_JOB_ID", "")


def get_slurm_remaining_minutes(job_id: str) -> int:
    """How many more minutes does the SLURM job have to run?"""
    remaining_time_str = subprocess.check_output(
        ["squeue", "-h", "-j", job_id, "-o", "%L"],
        text=True,
        timeout=60,
    )
    days_str, _, time_str = remaining_time_str.rpartition("-")
    if time_str.count(":") == 2:
        _remaining_time = datetime.datetime.strptime(time_str, "%H:%M:%S")
    elif time_str.count(":") == 1:
        _remaining_time = datetime.datetime.strptime(time_str, "%M:%S")
    else:
        raise ValueError(f"Could not parse time string {time_str}")
    remaining_time = datetime.timedelta(
        hours=_remaining_time.hour,
        minutes=_remaining_time.minute,
        seconds=_remaining_time.second,
    )
    remaining_time += datetime.timedelta(days=int(days_str))
    return int(remaining_time.total_seconds() / 60)


def kill_slurm_job(
    job_id: str,
):
    """Kill a SLURM job. No error is raised should the killing fail."""
    subprocess.run(["scancel", job_id], timeout=60)


class JobControl:
    def __init__(self):
        self.job_control_path = Path.home() / "ray_job_control.yaml"
        self._config = []
        self.logger = get_logger("JobControl")

    def _refresh(self):
        """Reload the config."""
        if not self.job_control_path.exists():
            self.logger.warning(
                "Job control file %s does not exist", self.job_control_path
            )
            return
        self.logger.debug(
            "Refreshing job control config from %s", self.job_control_path
        )
        with open(self.job_control_path) as f:
            try:
                self._config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                self.logger.error("Could not load job control config: %s", e)
                return

    def _get_actions(self, dispatcher_id: str) -> list[JobControlAction]:
        actions = []
        for c in self._config:
            self.logger.debug("Looking at JobControl option %s", c)
            if selected_job_id := str(c.get("job_id", "")):
                if selected_job_id != str(get_slurm_job_id()):
                    self.logger.debug(
                        "Job ID %s does not match %s. Skip.",
                        get_slurm_job_id(),
                        selected_job_id,
                    )
                    continue
                self.logger.debug("Job ID matches %s", selected_job_id)
            if selected_d_id := str(c.get("dispatcher_id", "")):
                if selected_d_id != dispatcher_id:
                    self.logger.debug(
                        "Dispatcher ID %s does not match %s. Skip.",
                        dispatcher_id,
                        selected_d_id,
                    )
                    continue
                self.logger.debug("Dispatcher ID matches %s", selected_d_id)
            if remaining_minutes_leq := int(c.get("remaining_minutes_leq", 0)):
                try:
                    remaining_minutes = get_slurm_remaining_minutes(get_slurm_job_id())
                except Exception as e:
                    self.logger.error(
                        "Could not get remaining minutes from slurm because of %s. "
                        "Not taking action.",
                        e,
                    )
                    continue
                if remaining_minutes > remaining_minutes_leq:
                    self.logger.debug(
                        "Remaining minutes %s > %s. Skip.",
                        remaining_minutes,
                        remaining_minutes_leq,
                    )
                    continue
                self.logger.debug(
                    "Remaining minutes %s <= %s",
                    remaining_minutes,
                    remaining_minutes_leq,
                )
            action = JobControlAction[c["action"].upper()]
            self.logger.info("Queued action %s", action)
            actions.append(action)
        if actions:
            self.logger.debug("Got actions %s", actions)
        return actions

    def _handle_action(self, action: JobControlAction) -> bool:
        """Handles action. Returns True if we should refresh the config and check
        if everything has been resolved afterwards.
        """
        if action == JobControlAction.WAIT:
            self.logger.info("Sleeping for 30s")
            time.sleep(30)
            return True
        if action == JobControlAction.KILL_NODE:
            job_id = get_slurm_job_id()
            self.logger.warning("Killing slurm job %s", job_id)
            kill_slurm_job(job_id)
            return True
        raise ValueError(f"Unknown action {action}")

    def __call__(self, *, dispatcher_id: str) -> None:
        repeat_requested = True
        while repeat_requested:
            self._refresh()
            repeat_requested = False
            try:
                actions = self._get_actions(dispatcher_id)
            except KeyError as e:
                self.logger.error(
                    "Could not get actions, please check your config: %s", e
                )
                return
            for action in actions:
                repeat_requested |= self._handle_action(action)
