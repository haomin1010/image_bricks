from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, order=True)
class BrickPosition:
    """Discrete brick position on the tabletop grid."""

    x: int
    y: int
    z: int

    @classmethod
    def from_mapping(cls, payload: dict) -> "BrickPosition":
        return cls(x=int(payload["x"]), y=int(payload["y"]), z=int(payload["z"]))

    def below(self) -> "BrickPosition | None":
        if self.z <= 0:
            return None
        return BrickPosition(self.x, self.y, self.z - 1)

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass(frozen=True)
class TaskSpec:
    """Ground-truth target structure loaded from a JSON file."""

    source_path: Path | None
    dimensions: tuple[int, int, int]
    positions: frozenset[BrickPosition]

    @property
    def total_blocks(self) -> int:
        return len(self.positions)

    @property
    def stem(self) -> str | None:
        return None if self.source_path is None else self.source_path.stem

    @classmethod
    def empty(cls) -> "TaskSpec":
        return cls(source_path=None, dimensions=(0, 0, 0), positions=frozenset())


def scan_ground_truth_entries(root: str) -> list[Path]:
    """Return all ground-truth JSON files under *root* in stable order."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(path for path in root_path.glob("*.json") if path.is_file())


def load_task_spec(json_path: str | Path) -> TaskSpec:
    """Load one convex-JSON target description into a TaskSpec."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    dims = payload.get("dimensions", {})
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
        for block in payload.get("blocks", [])
    )
    return TaskSpec(source_path=path, dimensions=dimensions, positions=positions)


def format_positions(positions: Iterable[BrickPosition], *, limit: int = 6) -> str:
    """Format a small coordinate preview for logs/debug output."""
    ordered_positions = sorted(positions)
    preview = ordered_positions[: max(0, int(limit))]
    if not preview:
        return "[]"
    formatted = ", ".join(f"({pos.x}, {pos.y}, {pos.z})" for pos in preview)
    if len(preview) < len(ordered_positions):
        return f"[{formatted}, ...]"
    return f"[{formatted}]"
