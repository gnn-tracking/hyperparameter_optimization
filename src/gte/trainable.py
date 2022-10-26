from __future__ import annotations

import pprint
from functools import partial
from pathlib import Path
from typing import Any

from gnn_tracking.metrics.losses import (
    BackgroundLoss,
    EdgeWeightFocalLoss,
    PotentialLoss,
)
from gnn_tracking.models.track_condensation_networks import GraphTCN
from gnn_tracking.postprocessing.dbscanscanner import dbscan_scan
from gnn_tracking.training.tcn_trainer import TCNTrainer
from gnn_tracking.utils.dictionaries import subdict_with_prefix_stripped
from gnn_tracking.utils.log import logger
from gnn_tracking.utils.seeds import fix_seeds
from ray import tune
from torch.optim import Adam

from gte.config import server
from gte.load import get_graphs, get_loaders


def faster_dbscan_scan(*args, n_epoch=0, n_trials=100, **kwargs):
    """Skip scanning every second trial."""
    if n_epoch % 2 == 1 and n_epoch >= 4:
        logger.debug("Not reevaluating scanning of DBSCAN in epoch %d", n_epoch)
        n_trials = 1
    return dbscan_scan(*args, n_trials=n_trials, **kwargs)


class TCNTrainable(tune.Trainable):
    # Do not add blank self.tc or self.trainer to __init__, because it will be called
    # after setup when setting ``reuse_actor == True`` and overwriting your values
    # from set
    def setup(self, config: dict[str, Any]):
        logger.debug("Got config\n%s", pprint.pformat(config))
        self.tc = config
        fix_seeds()
        self.trainer = self.get_trainer()
        logger.debug(f"Trainer: {self.trainer}")
        self.post_setup_hook()
        self.trainer.pt_thlds = [1.5]

    def post_setup_hook(self):
        pass

    def get_model(self) -> GraphTCN:
        return GraphTCN(
            node_indim=6, edge_indim=4, **subdict_with_prefix_stripped(self.tc, "m_")
        )

    def get_edge_loss_function(self):
        return EdgeWeightFocalLoss(
            alpha=self.tc.get("focal_alpha", 0.25),
            gamma=self.tc.get("focal_gamma", 2),
        )

    def get_potential_loss_function(self):
        return PotentialLoss(
            q_min=self.tc.get("q_min", 0.01),
            attr_pt_thld=self.tc.get("attr_pt_thld", 0.9),
        )

    def get_background_loss_function(self):
        return BackgroundLoss(sb=self.tc.get("sb", 0.1))

    def get_loss_functions(self) -> dict[str, Any]:
        return {
            "edge": self.get_edge_loss_function(),
            "potential": self.get_potential_loss_function(),
            "background": self.get_background_loss_function(),
        }

    def get_cluster_functions(self) -> dict[str, Any]:
        return {
            "dbscan": partial(
                faster_dbscan_scan,
                n_trials=100 if not self.tc.get("test", False) else 1,
                n_jobs=server.cpus_per_gpu if not self.tc.get("test", False) else 1,
            )
        }

    def get_lr_scheduler(self):
        return None

    def get_optimizer(self):
        return Adam

    def get_loss_weights(self):
        return subdict_with_prefix_stripped(self.tc, "lw_")

    def get_trainer(self) -> TCNTrainer:
        test = self.tc.get("test", False)
        trainer = TCNTrainer(
            model=self.get_model(),
            loaders=get_loaders(get_graphs(test=test), test=test),
            loss_functions=self.get_loss_functions(),
            loss_weights=self.get_loss_weights(),
            lr=self.tc.get("lr", 5e-4),
            lr_scheduler=self.get_lr_scheduler(),
            cluster_functions=self.get_cluster_functions(),  # type: ignore
            optimizer=self.get_optimizer(),
        )
        trainer.max_batches_for_clustering = 100 if not test else 10
        return trainer

    def step(self):
        return self.trainer.step(max_batches=self.tc.get("max_batches", None))

    def save_checkpoint(
        self,
        checkpoint_dir,
    ):
        return self.trainer.save_checkpoint(
            Path(checkpoint_dir) / "checkpoint.pt",
        )

    def load_checkpoint(self, checkpoint_path, **kwargs):
        self.trainer.load_checkpoint(checkpoint_path, **kwargs)
