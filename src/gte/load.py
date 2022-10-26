from __future__ import annotations

from pathlib import Path

import sklearn.model_selection
from gnn_tracking.graph_construction.graph_builder import GraphBuilder
from gnn_tracking.utils.log import logger
from torch_geometric.loader import DataLoader

from gte.config import server


def get_graphs(test=False) -> dict[str, list]:
    logger.info("Loading data to cpu memory")
    graph_builder = GraphBuilder(
        str(Path("~/data/gnn_tracking/point_clouds").expanduser()),
        str(Path("~/data/gnn_tracking/graphs").expanduser()),
        redo=False,
    )
    n_graphs = 100 if test else None
    logger.debug("Loading %s graphs", n_graphs)
    graph_builder.process(stop=n_graphs)

    # partition graphs into train, test, val splits
    graphs = graph_builder.data_list
    _train_graphs, test_graphs = sklearn.model_selection.train_test_split(
        graphs, test_size=0.2
    )
    train_graphs, val_graphs = sklearn.model_selection.train_test_split(
        _train_graphs, test_size=0.15
    )
    return {
        "train": train_graphs,
        "val": val_graphs,
        "test": test_graphs,
    }


def get_loaders(
    graph_dct: dict[str, list], batch_size=1, test=False
) -> dict[str, DataLoader]:
    """Get data loaders

    Args:
        graph_dct:
        batch_size:
        test:

    Returns:
        Dictionary of data loaders
    """
    # build graph loaders
    params = {
        "batch_size": batch_size,
        "num_workers": server.cpus_per_gpu if not test else 1,
    }
    logger.debug("Parameters for data loaders: %s", params)
    loaders = {
        "train": DataLoader(list(graph_dct["train"]), **params, shuffle=True),
        "test": DataLoader(list(graph_dct["test"]), **params),
        "val": DataLoader(list(graph_dct["val"]), **params),
    }
    return loaders
