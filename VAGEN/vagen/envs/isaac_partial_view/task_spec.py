from __future__ import annotations

import json
from pathlib import Path

from ..isaac.task_spec import BrickPosition, TaskSpec, format_positions, load_task_spec, scan_ground_truth_entries


def load_snapshot_task_spec(json_path: str | Path) -> TaskSpec:
    """Load one snapshot ``*_data.json`` into a TaskSpec.

    Expected format:
    - ``original_data.dimensions.{length,width,height}``
    - ``original_data.blocks`` with ``x,y,z`` integer coordinates
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    original = payload.get("original_data", {}) if isinstance(payload, dict) else {}
    dims = original.get("dimensions", {}) if isinstance(original, dict) else {}
    blocks = original.get("blocks", []) if isinstance(original, dict) else []

    dimensions = (
        int(dims.get("length", 0)),
        int(dims.get("width", 0)),
        int(dims.get("height", 0)),
    )
    positions = frozenset(
        BrickPosition(
            x=int(block["x"]),
            y=int(block["y"]),
            z=int(block["z"]),
        )
        for block in blocks
        if isinstance(block, dict) and {"x", "y", "z"}.issubset(block.keys())
    )
    return TaskSpec(source_path=path, dimensions=dimensions, positions=positions)


__all__ = [
    "BrickPosition",
    "TaskSpec",
    "format_positions",
    "load_snapshot_task_spec",
    "load_task_spec",
    "scan_ground_truth_entries",
]
