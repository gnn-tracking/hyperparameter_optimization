#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

from gnn_tracking.graph_construction.graph_builder import GraphBuilder

graph_builder = GraphBuilder(
    str(Path("~/data/gnn_tracking/point_clouds").expanduser()),
    str(Path("~/data/gnn_tracking/graphs").expanduser()),
    redo=False,
)
graph_builder.process(stop=None)
