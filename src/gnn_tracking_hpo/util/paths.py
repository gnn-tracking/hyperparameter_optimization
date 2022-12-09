from __future__ import annotations

from pathlib import Path

_default_base_path = Path("~/ray_results/").expanduser()


def find_checkpoints(
    project: str, part: str, base_path=_default_base_path
) -> list[Path]:
    """Find checkpoints

    Args:
        project: Ray project name/folder
        part: Part of filename of trial (part of the hash)
        base_path: Path to ray folder

    Returns:
        List of paths to ``checkpoint.pt`` files (most recent last)
    """
    project_dir = base_path / project
    hits = [d for d in project_dir.iterdir() if d.is_dir and part in d.name]
    if len(hits) == 0:
        raise ValueError("No such directory found")
    if len(hits) >= 2:
        raise ValueError("Non-unique description of dir")
    result_dir = hits[0]
    assert result_dir.is_dir()
    checkpoints = sorted(result_dir.glob("checkpoint_*"))
    if not checkpoints:
        raise ValueError(f"No checkpoints at {result_dir}")
    checkpoint_files = [cp / "checkpoint.pt" for cp in checkpoints]
    assert all(cpf for cpf in checkpoint_files)
    return checkpoint_files
