"""Everything related to data loaders and data sets."""


from __future__ import annotations

import os
from pathlib import Path

from gnn_tracking.graph_construction.graph_builder import load_graphs
from gnn_tracking.utils.loading import get_loaders as _get_loaders
from gnn_tracking.utils.loading import train_test_val_split
from gnn_tracking.utils.log import logger
from torch_geometric.loader import DataLoader

from gnn_tracking_hpo.config import server


def get_graphs(
    *,
    n_graphs,
    test_frac=0.2,
    val_frac=0.12,
    input_dir: os.PathLike | str | None = None,
    sector: int | None = None,
    test=False,
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
    if input_dir is None:
        input_dir = os.environ.get(
            "DATA_DIR", "/tigress/jdezoort/object_condensation/graphs"
        )
    assert input_dir is not None
    logger.debug("Loading graphs from %s", input_dir)

    if test:
        # Let's cheat and only load one graph that we use for
        # train/val/test
        graph = load_graphs(Path(input_dir), stop=1, sector=sector)[0]
        return {
            "train": [graph],
            "test": [graph],
            "val": [graph],
        }

    assert 0 <= test_frac <= 1
    assert 0 <= val_frac <= 1
    assert test_frac + val_frac <= 1

    if n_graphs is None:
        raise ValueError(
            "Please explicitly set n_graphs to track it as a hyperparameter"
        )

    logger.info("Loading data to cpu memory")
    graphs = load_graphs(str(Path(input_dir)), stop=n_graphs, sector=sector)
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
        cpus=server.cpus if not test else 1,
    )
