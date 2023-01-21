#!/usr/bin/env python3

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from gnn_tracking_hpo.config import retrieve_config_from_wandb
from gnn_tracking_hpo.util.log import logger

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("hashes", nargs="+")
    args = parser.parse_args()
    for this_hash in args.hashes:
        config = retrieve_config_from_wandb(this_hash)
        output_path = f"{this_hash}.json"
        Path(output_path).write_text(json.dumps(config, indent=4))
        logger.info("Wrote to %s", output_path)
