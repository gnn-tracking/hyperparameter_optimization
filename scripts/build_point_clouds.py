#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from gnn_tracking.preprocessing.point_cloud_builder import PointCloudBuilder

pc_builder = PointCloudBuilder(
    indir="/tigress/jdezoort/codalab/train_1",
    outdir=str(Path("~/data/gnn_tracking/point_clouds").expanduser()),
    n_sectors=32,
    pixel_only=True,
    redo=False,
    measurement_mode=False,
    sector_di=0,
    sector_ds=1.3,
    thld=0.9,
    log_level=0,
    collect_data=False,
)
pc_builder.process(stop=100)
