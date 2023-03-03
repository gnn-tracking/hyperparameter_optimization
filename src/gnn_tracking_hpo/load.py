"""Everything related to data loaders and data sets."""


from __future__ import annotations

import os

import sklearn.model_selection
from gnn_tracking.graph_construction.graph_builder import load_graphs
from gnn_tracking.utils.loading import get_loaders as _get_loaders
from torch_geometric.loader import DataLoader

from gnn_tracking_hpo.util.log import logger


def get_graphs_split(
    *,
    train_size: int,
    val_size: int,
    input_dirs: list[os.PathLike] | list[str],
    sector: int | None = None,
    test=False,
) -> dict[str, list]:
    """Load graphs for training, testing, and validation from one directory.

    Args:
        train_size: Number of graphs to use for training
        val_size: Number of graphs to use for validation
        input_dirs: Directory containing the graphs
        sector: Only load specific sector
        test:

    Returns:
        Training and validation graphs as dictionary

    """
    assert train_size >= 1 or train_size == 0
    assert val_size >= 1 or val_size == 0

    logger.debug("Loading graphs from %s", input_dirs)

    if test:
        # Let's cheat and only load one graph that we use for
        # train/val/test
        logger.debug(
            "Loading test graphs (only one graph is loaded and used for train/test/val"
        )
        graph = load_graphs(input_dirs, stop=1, sector=sector)[0]
        return {
            "train": [graph],
            "val": [graph],
            "test": [],
        }

    logger.info("Loading data to cpu memory")
    graphs = load_graphs(
        input_dirs,
        stop=train_size + val_size,
        sector=sector,
        n_processes=12 if not test else 1,
    )
    train_graphs, val_graphs = sklearn.model_selection.train_test_split(
        graphs,
        train_size=train_size,
        test_size=val_size,
    )
    return {
        "train": train_graphs,
        "val": val_graphs,
        "test": [],
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
    return _get_loaders(
        graph_dct,
        batch_size=batch_size,
        cpus=12 if not test else 1,
    )
