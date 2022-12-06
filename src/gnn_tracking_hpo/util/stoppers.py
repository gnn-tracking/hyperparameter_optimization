from __future__ import annotations

import collections
from typing import Any, DefaultDict

from ray import tune


class ResultThresholdStopper(tune.Stopper):
    def __init__(self, metric: str, thresholds: dict[int, float], mode: str = "max"):
        """Stopper that stops if results at a certain epoch fall above/below a certain
        threshold.
        """
        self.metric = metric
        self.threshold = thresholds
        self.mode = mode

    def get_threshold(self, epoch: int) -> float:
        last_specified_epoch = max([k for k in self.threshold.keys() if k <= epoch])
        return self.threshold[last_specified_epoch]

    def __call__(self, trial_id, result) -> bool:
        epoch = result["epoch"]
        threshold = self.get_threshold(epoch)
        if self.mode == "max":
            return result[self.metric] > threshold
        elif self.mode == "min":
            return result[self.metric] < threshold
        else:
            raise ValueError(f"Invalid mode {self.mode}")

    def stop_all(self) -> bool:
        return False


class NoImprovementStopper(tune.Stopper):
    def __init__(
        self,
        metric: str,
        rel_change_thld=0.01,
        mode: str = "max",
        patience=6,
        grace_period=4,
    ):
        """Stopper that stops if at no iteration within ``num_results`` a better
        result than the current best one is observed.

        Args:
            metric:
            rel_change_thld: Relative change threshold to be considered for improvement.
                Any change that is less than that is considered no improvement.
        """
        self.metric = metric
        self.rel_change_thld = rel_change_thld
        self.mode = mode
        self.patience = patience
        self.grace_period = grace_period
        self._best: DefaultDict[Any, None | int] = collections.defaultdict(lambda: None)
        self._stagnant: DefaultDict[Any, int] = collections.defaultdict(int)
        self._epoch: DefaultDict[Any, int] = collections.defaultdict(int)

    def __call__(self, trial_id, result) -> bool:
        self._epoch[trial_id] += 1
        if self._best[trial_id] is None:
            self._best[trial_id] = result[self.metric]
            return False
        try:
            ratio = result[self.metric] / self._best[trial_id]
        except ZeroDivisionError:
            ratio = None
        if (
            self.mode == "max"
            and ratio is not None
            and ratio > 1 + self.rel_change_thld
        ) or (
            self.mode == "min"
            and ratio is not None
            and ratio < 1 - self.rel_change_thld
        ):
            self._best[trial_id] = result[self.metric]
            self._stagnant[trial_id] = 0
            return False
        self._stagnant[trial_id] += 1
        if self._epoch[trial_id] < self.grace_period:
            return False
        if self._stagnant[trial_id] > self.patience:
            return True
        return False

    def stop_all(self):
        return False
