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
    parser.add_argument(
        "--output", "-o", help="Output directory", default=".", type=Path
    )
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    for this_hash in args.hashes:
        try:
            config = retrieve_config_from_wandb(this_hash)
        except ValueError:
            logger.error(f"Could not retrieve config for hash {this_hash}")
            continue
        output_path = args.output / f"{this_hash}.json"
        Path(output_path).write_text(json.dumps(config, indent=4))
        logger.info("Wrote to %s", output_path)
