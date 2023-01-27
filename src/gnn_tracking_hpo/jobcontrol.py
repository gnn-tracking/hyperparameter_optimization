from __future__ import annotations

import datetime
import os
import subprocess
import time
from enum import Enum, auto
from pathlib import Path

import yaml
from dateutil import parser

from gnn_tracking_hpo.util.log import logger


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
    # yes, there's json output, but it doesn't show the required information
    job_stats_text = subprocess.check_output(
        ["jobstats", job_id],
        text=True,
        timeout=60,
    )
    run_time = None
    time_limit = None
    for line in job_stats_text.splitlines():
        if "Run Time" in line:
            time_str = (
                line.replace("Run Time:", "").replace("(in progress)", "").strip()
            )
            _run_time = parser.parse(time_str)
            run_time = datetime.timedelta(
                hours=_run_time.hour, minutes=_run_time.minute, seconds=_run_time.second
            )
        if "Time Limit" in line:
            time_str = line.replace("Time Limit:", "").strip()
            days_str, _, time_str = time_str.rpartition("-")
            time_limit = datetime.timedelta(0)
            if days_str:
                time_limit += datetime.timedelta(days=int(days_str))
            _time_limit = parser.parse(time_str)
            time_limit += datetime.timedelta(
                hours=_time_limit.hour,
                minutes=_time_limit.minute,
                seconds=_time_limit.second,
            )
    if run_time is None:
        raise ValueError("Couldn't find run time in jobstats output")
    if time_limit is None:
        raise ValueError("Couldn't find time limit in jobstats output")
    return int((time_limit - run_time).total_seconds() / 60)


def kill_slurm_job(
    job_id: str,
):
    """Kill a SLURM job. No error is raised should the killing fail."""
    logger.info("Killing slurm job %s", job_id)
    subprocess.run(["scancel", job_id], timeout=60)


class JobControl:
    def __init__(self):
        self.job_control_path = Path.home() / "ray_job_control.yaml"
        self.config = []

    def refresh(self):
        """Reload the config."""
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

    def _get_actions(self, dispatcher_id: str) -> list[JobControlAction]:
        actions = []
        for c in self.config:
            logger.debug("Looking at JobControl option %s", c)
            if selected_job_id := str(c.get("job_id", "")):
                if selected_job_id != str(get_slurm_job_id()):
                    logger.debug(
                        "Job ID %s does not match %s. Skip.",
                        get_slurm_job_id(),
                        selected_job_id,
                    )
                    continue
            if selected_d_id := str(c.get("dispatcher_id", "")):
                if selected_d_id != dispatcher_id:
                    logger.debug(
                        "Dispatcher ID %s does not match %s. Skip.",
                        dispatcher_id,
                        selected_d_id,
                    )
                    continue
            if remaining_minutes_leq := int(c.get("remaining_minutes_leq", 0)):
                try:
                    remaining_minutes = get_slurm_remaining_minutes(get_slurm_job_id())
                except Exception as e:
                    logger.error(
                        "Could not get remaining minutes from slurm because of %s. "
                        "Not taking action.",
                        e,
                    )
                    continue
                if remaining_minutes > remaining_minutes_leq:
                    logger.debug(
                        "Remaining minutes %s > %s. Skip.",
                        remaining_minutes,
                        remaining_minutes_leq,
                    )
                    continue
            action = JobControlAction[c["action"].upper()]
            logger.info("Queued action %s", action)
            actions.append(action)
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

    def __call__(self, *, dispatcher_id: str) -> None:
        repeat_requested = True
        while repeat_requested:
            self.refresh()
            repeat_requested = False
            try:
                actions = self._get_actions(dispatcher_id)
            except KeyError as e:
                logger.error("Could not get actions, please check your config: %s", e)
                return
            for action in actions:
                repeat_requested |= self._handle_action(action)
