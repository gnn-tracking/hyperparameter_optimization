from __future__ import annotations

import collections
from argparse import ArgumentParser
from functools import partial
from math import isnan
from typing import Any, DefaultDict

import optuna
from gnn_tracking.training.lw_setter import LinearLWSH
from gnn_tracking.training.tc import TCNTrainer
from ray import tune
from ray.air import CheckpointConfig
from rt_stoppers_contrib import NoImprovementTrialStopper, ThresholdTrialStopper

from gnn_tracking_hpo.cli import add_restore_options
from gnn_tracking_hpo.config import auto_suggest_if_not_fixed, get_metadata
from gnn_tracking_hpo.defaults import suggest_default_values
from gnn_tracking_hpo.trainable import MLTrainable
from gnn_tracking_hpo.tune import Dispatcher, add_common_options
from gnn_tracking_hpo.util.dict import pop

# class KiliansComplicatedLWS:
#     def __init__(self, ratio_up=1.1, ratio_down=0.9, max_attr=1., max_rep=6.,
#     target_ratio=0.2):
#         self._ratio_up = ratio_up
#         self._ratio_down = ratio_down
#         self._max_attr = max_attr
#         self._max_rep = max_rep
#         self._target_ratio = target_ratio
#
#
#     def _get_last_losses(self, trainer):
#         if len(trainer.test_loss) == 0:
#             return {
#                 "attractive": float('nan'),
#                 "repulsive": float('nan'),
#             }
#         return {
#             "attractive": trainer.test_loss[-1]["potential_attractive"].item(),
#             "repulsive": trainer.test_loss[-1]["potential_attractive"].item(),
#         }
#
#     def get_lw(self, losses):
#         if losses["attractive"] > self._max_attr:
#
#
#     def __call__(
#             self,
#             trainer,
#             epoch: int,
#             batch_idx: int,
#             model_output: dict[str, Any],
#             data: Data,
#     ):
#         lw = self.get_lw(self._get_last_losses(trainer))
#         trainer.loss_functions["potential"]["repulsive"] = lw


class LWSGCTrainable(MLTrainable):
    def get_trainer(self) -> TCNTrainer:
        trainer = super().get_trainer()
        trainer.add_hook(self.get_lws_hook(), "batch")
        return trainer

    def get_lws_hook(self):
        return LinearLWSH(
            loss_name=("potential", "repulsive"),
            start_value=self.tc["lws_repulsive_llw_start_value"],
            end_value=self.tc["lw_potential_repulsive"],
            end=self.tc["lws_repulsive_llw_end"],
            n_batches=800,
        )


def suggest_config(
    trial: optuna.Trial,
    *,
    test=False,
    fixed: dict[str, Any] | None = None,
    gc_hash: str = "",
    gc_epoch: int = -1,
    gc_project: str = "",
) -> dict[str, Any]:
    config = get_metadata(test=test)
    config.update(fixed or {})

    def d(key, *args, **kwargs):
        return auto_suggest_if_not_fixed(key, config, trial, *args, **kwargs)

    d("n_graphs_train", 7463)
    config["train_data_dir"] = [
        f"/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/point_clouds_v5/part_{i}"
        for i in range(1, 9)
    ]
    d(
        "val_data_dir",
        "/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/point_clouds_v5/part_9",
    )
    d("n_graphs_val", 5)
    d("batch_size", 1)
    d("_val_batch_size", 1)

    # model restore
    if gc_hash:
        d("gc_project", gc_project)
        d("gc_hash", gc_hash)
        d("gc_epoch", gc_epoch)

    # fixed parameters
    # -----------------------

    d("m_hidden_dim", 512)
    d("lw_potential_attractive", 1.0)
    # d("attr_pt_thld", 0.9)

    d("max_edges_per_node", 256)
    d("m_depth", 6)
    d("rs_max_edges", 10_000_000)
    d("max_sample_size", 800)
    # d("repulsive_radius_threshold", 10)
    d("max_num_neighbors", 256)
    d("r_emb", 1.0)

    # Tuned parameters
    # ----------------

    d("m_out_dim", 8)
    d("lr", 1e-3)
    d("p_attr", 2)
    d("p_rep", 2)
    d("m_beta", 0.4)
    d("attr_pt_thld", 0.9)
    d("lw_potential_repulsive", 1e-4, 1, log=True)  # 5e-2

    suggest_default_values(config, trial, hc="none", ec="none")
    return config


class NoNaNStopper(tune.Stopper):
    def __init__(self, settings: dict[str, tuple[int, int]]):
        self._settings = settings
        self._epoch: DefaultDict[Any, int] = collections.defaultdict(int)
        self._n_consecutive_nans: Any = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )

    def __call__(self, trial_id: str, result: dict) -> bool:
        self._epoch[trial_id] += 1
        for metric, (min_epoch, patience) in self._settings.items():
            if self._epoch[trial_id] < min_epoch:
                continue
            if isnan(result[metric]):
                self._n_consecutive_nans[trial_id][metric] += 1
                if self._n_consecutive_nans[trial_id][metric] >= patience:
                    return True
            else:
                self._n_consecutive_nans[trial_id][metric] = 0
        return False

    def stop_all(self) -> bool:
        return False


class MyDispatcher(Dispatcher):
    def get_no_improvement_stopper(self) -> NoImprovementTrialStopper | None:
        return NoImprovementTrialStopper(
            metric=self.metric,
            patience=10,
            mode=self.comparison,
            grace_period=10,
            rel_change_thld=0.01,
        )

    def get_checkpoint_config(self) -> CheckpointConfig:
        return CheckpointConfig(
            checkpoint_score_attribute=self.metric,
            checkpoint_score_order=self.comparison,
            num_to_keep=10,
            checkpoint_frequency=1,
        )

    # def get_optuna_sampler(self):
    #     return optuna.samplers.RandomSampler()


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_options(parser)
    add_restore_options(parser, prefix="gc", name="Graph Construction Network")
    kwargs = vars(parser.parse_args())
    assert kwargs["wandb_project"] == "gnn_tracking_gc"
    kwargs.pop("no_scheduler")
    this_suggest_config = partial(
        suggest_config, **pop(kwargs, ["gc_hash", "gc_project", "gc_epoch"])
    )

    # nn_stopper = NoNaNStopper(
    #     {
    #         "n_edges_frac_segment50_80": (10, 3),
    #         "n_edges_frac_segment50_90": (10, 3),
    #         "n_edges_frac_segment50_93": (15, 4),
    #     }
    # )
    thld_stopper = ThresholdTrialStopper(
        metric="max_frac_segment50",
        mode="max",
        thresholds={
            # 3: 0.7,
            # 10: 0.9,
            10: 0.5,
            15: 0.6,
            20: 0.7,
            35: 0.8,
        },
    )
    dispatcher = MyDispatcher(
        **kwargs,
        metric="max_frac_segment50",
        grace_period=6,
        comparison="max",
        no_scheduler=True,
        additional_stoppers=[thld_stopper],
    )
    dispatcher(MLTrainable, this_suggest_config)
