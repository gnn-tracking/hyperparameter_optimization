from __future__ import annotations

from os import PathLike
from pathlib import Path

from gnn_tracking.utils.log import logger
from ray.tune.experiment import Trial
from ray.tune.logger import LoggerCallback


class TriggerSyncHook(LoggerCallback):
    def __init__(self, communication_dir: PathLike):
        super().__init__()
        self.communication_dir = Path(communication_dir)
        self.communication_dir.mkdir(parents=True, exist_ok=True)

    def log_trial_result(self, iteration: int, trial: Trial, result: dict):
        trial_dir = trial.logdir
        command_file = self.communication_dir / f"{trial_dir.name}.command"
        if command_file.is_file():
            logger.warning(
                "Syncing not active or too slow: Command %s file still exists",
                command_file,
            )
        command_file.write_text(trial_dir.resolve().as_posix())
        logger.debug("Wrote command file %s", command_file)
