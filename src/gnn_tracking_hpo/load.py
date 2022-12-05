"""Everything related to data loaders and data sets."""


from __future__ import annotations

import os
from pathlib import Path

import sklearn.model_selection
from gnn_tracking.graph_construction.graph_builder import load_graphs
from gnn_tracking.utils.log import logger
from torch_geometric.loader import DataLoader

from gnn_tracking_hpo.config import server


def get_graphs(
    *,
    n_graphs,
    test_frac=0.2,
    val_frac=0.12,
    input_dir: os.PathLike | str = "/tigress/jdezoort/object_condensation/graphs",
    sector: int | None = None,
) -> dict[str, list]:
    """Load graphs for training, testing, and validation.

    Args:
        n_graphs: Total number of graphs
        test_frac: Fraction of graphs used for testing
        val_frac: Fraction of graphs for validation
        input_dir: Directory containing the graphs
        sector: Only load specific sector

    Returns:

    """
    assert 0 <= test_frac <= 1
    assert 0 <= val_frac <= 1
    assert test_frac + val_frac <= 1

    if n_graphs is None:
        raise ValueError(
            "Please explicitly set n_graphs to track it as a hyperparameter"
        )
    logger.info("Loading data to cpu memory")
    graphs = load_graphs(
        str(Path(input_dir).expanduser()), stop=n_graphs, sector=sector
    )
    rest, test_graphs = sklearn.model_selection.train_test_split(
        graphs, test_size=test_frac
    )
    train_graphs, val_graphs = sklearn.model_selection.train_test_split(
        rest, test_size=val_frac / (1 - test_frac)
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
