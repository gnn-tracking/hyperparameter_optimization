#!/usr/bin/env python3

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from gnn_tracking_hpo.config import retrieve_config_from_wandb

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("hash")
    args = parser.parse_args()
    config = retrieve_config_from_wandb(args.hash)
    Path(f"{args.hash}.json").write_text(json.dumps(config, indent=4))
