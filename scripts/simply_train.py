#!/usr/bin/env python3

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from gnn_tracking.metrics.losses import PotentialLoss

from gnn_tracking_hpo.cli import add_test_option
from gnn_tracking_hpo.config import read_json
from gnn_tracking_hpo.trainable import TCNTrainable


class ThisTrainable(TCNTrainable):
    def post_setup_hook(self):
        self.trainer.ec_eval_pt_thlds = [1.5]

    def get_potential_loss_function(self):
        return PotentialLoss(q_min=self.tc.get("q_min", 0.01), attr_pt_thld=0.0)

    def get_lr_scheduler(self):
        return None


def main():
    parser = ArgumentParser()
    add_test_option(parser)
    parser.add_argument("--config", dest="config_file", help="Config file")
    args = parser.parse_args()
    config = {}
    if args.config_file:
        config = read_json(Path(args.config_file))
    if args.test:
        config["test"] = True

    trainable = ThisTrainable()
    trainable.setup(config)
    while True:
        trainable.step()


if __name__ == "__main__":
    main()
