from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gnn_tracking_hpo

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
        raise ValueError("No such directory found")
    if len(hits) >= 2:
        raise ValueError("Non-unique description of dir")
    result_dir = hits[0]
    assert result_dir.is_dir()
    return result_dir


def find_checkpoints(
    project: str, part: str, *, base_path=_default_base_path
) -> list[Path]:
    """Find checkpoints

    Args:
        project: Ray project name/folder
        part: Part of filename of trial (part of the hash)
        base_path: Path to ray folder

    Returns:
        List of paths to ``checkpoint.pt`` files (most recent last)
    """
    result_dir = find_result_dir(project, part, base_path=base_path)
    checkpoints = sorted(result_dir.glob("checkpoint_*"))
    if not checkpoints:
        raise ValueError(f"No checkpoints at {result_dir}")
    checkpoint_files = [cp / "checkpoint.pt" for cp in checkpoints]
    assert all(cpf for cpf in checkpoint_files)
    return checkpoint_files


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
