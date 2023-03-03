"""Everything related to data loaders and data sets."""


from __future__ import annotations

import os

from gnn_tracking.graph_construction.graph_builder import load_graphs
from gnn_tracking.utils.loading import get_loaders as _get_loaders
from gnn_tracking.utils.loading import train_test_val_split
from torch_geometric.loader import DataLoader

from gnn_tracking_hpo.util.log import logger


def get_graphs_split(
    *,
    n_graphs,
    test_frac=0.2,
    val_frac=0.12,
    input_dirs: list[os.PathLike] | list[str],
    sector: int | None = None,
    test=False,
) -> dict[str, list]:
    """Load graphs for training, testing, and validation from one directory.

    Args:
        n_graphs: Total number of graphs
        test_frac: Fraction of graphs used for testing
        val_frac: Fraction of graphs for validation
        input_dirs: Directory containing the graphs
        sector: Only load specific sector
        test:

    Returns:

    """
    logger.debug("Loading graphs from %s", input_dirs)

    if test:
        # Let's cheat and only load one graph that we use for
        # train/val/test
        graph = load_graphs(input_dirs, stop=1, sector=sector)[0]
        return {
            "train": [graph],
            "test": [graph],
            "val": [graph],
        }

    if n_graphs is None:
        raise ValueError(
            "Please explicitly set n_graphs to track it as a hyperparameter"
        )

    logger.info("Loading data to cpu memory")
    graphs = load_graphs(
        input_dirs,
        stop=n_graphs,
        sector=sector,
        n_processes=12 if not test else 1,
    )
    return train_test_val_split(
        graphs,
        test_frac=test_frac,
        val_frac=val_frac,
    )


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
