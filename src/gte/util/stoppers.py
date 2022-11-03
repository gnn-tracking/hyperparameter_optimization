from __future__ import annotations

from ray import tune


class ResultThresholdStopper(tune.Stopper):
    def __init__(self, metric: str, thresholds: dict[int, float], mode: str = "max"):
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
