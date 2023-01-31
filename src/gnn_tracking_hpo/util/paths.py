from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gnn_tracking_hpo
from gnn_tracking_hpo.util.log import logger

_default_base_path = Path("~/ray_results/").expanduser()


def find_result_dir(project: str, part: str, *, base_path=_default_base_path) -> Path:
    """Find result dir of a trial

    Args:
        project: Ray project name/folder
        part: Part of filename of trial (part of the hash)
        base_path: Path to ray folder

    Returns:
        Path to result dir
    """
    project_dir = base_path / project
    hits = [d for d in project_dir.iterdir() if d.is_dir and part in d.name]
    if len(hits) == 0:
        raise ValueError(
            f"No such directory found: subdir of {project_dir}"
            f" with {part} in its name"
        )
    if len(hits) >= 2:
        raise ValueError(f"Non-unique description of dir: {project_dir}")
    result_dir = hits[0]
    assert result_dir.is_dir()
    return result_dir


def find_checkpoint(
    project: str, part: str, epoch: int = -1, *, base_path=_default_base_path
) -> Path:
    """Find checkpoints

    Args:
        project: Ray project name/folder
        part: Part of filename of trial (part of the hash)
        epoch: Epoch to which the checkpoint should belong. If -1, the last
            epoch will be taken
        base_path: Path to ray folder

    Returns:
        Path to `checkpoint.pt` file
    """
    result_dir = find_result_dir(project, part, base_path=base_path)
    if epoch == -1:
        glob_str = "checkpoint_*"
        try:
            checkpoint_dir = sorted(result_dir.glob(glob_str))[-1]
        except IndexError as e:
            raise ValueError(
                f"No checkpoint found in {result_dir} matching {glob_str}"
            ) from e
    else:
        checkpoint_dir = result_dir / f"checkpoint_{epoch:06}"
        if not checkpoint_dir.exists():
            raise ValueError(f"No checkpoint found at {checkpoint_dir}")
    checkpoint = checkpoint_dir / "checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint}, even though directory exists"
        )
    return checkpoint


def get_config(
    project: str, part: str, *, base_path=_default_base_path
) -> dict[str, Any]:
    """Get config of a trial

    Args:
        project: Ray project name/folder
        part: Part of filename of trial (part of the hash)
        base_path: Path to ray folder

    Returns:
        Config of trial
    """
    result_dir = find_result_dir(project, part, base_path=base_path)
    config_file = result_dir / "params.json"
    logger.debug("Loading config from %s", config_file)
    if not config_file.exists():
        raise ValueError(f"No config file at {config_file}")
    with open(config_file) as f:
        config = json.load(f)
    return config


def add_scripts_path() -> None:
    """Add the path of the scripts directory of this repository to
    the python PATH in order to import from there.
    """
    scripts_path = Path(gnn_tracking_hpo.__path__[0]).parent.parent / "scripts"
    assert scripts_path.is_dir()
    sys.path.append(str(scripts_path))
