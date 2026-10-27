"""ARC task loader. Minimal, numpy-only, fail-fast on malformed JSON."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

Grid = np.ndarray


@dataclass(frozen=True)
class Pair:
    input: Grid
    output: Grid | None  # None for hidden test pairs


@dataclass(frozen=True)
class Task:
    task_id: str
    train: tuple[Pair, ...]
    test: tuple[Pair, ...]


def _to_grid(data: list[list[int]]) -> Grid:
    arr = np.array(data, dtype=np.int8)
    if arr.ndim != 2:
        raise ValueError(f"Grid must be 2D, got shape {arr.shape}")
    if not arr.size:
        raise ValueError("Grid must be non-empty")
    if arr.min() < 0 or arr.max() > 9:
        raise ValueError(f"Grid values must be 0-9, got range {arr.min()}-{arr.max()}")
    return arr


def load_task(path: Path | str) -> Task:
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    train = tuple(Pair(_to_grid(p["input"]), _to_grid(p["output"])) for p in data["train"])
    test = tuple(
        Pair(_to_grid(p["input"]), _to_grid(p["output"]) if "output" in p else None)
        for p in data["test"]
    )
    return Task(task_id=path.stem, train=train, test=test)


def load_split(split_dir: Path | str) -> Iterator[Task]:
    split_dir = Path(split_dir)
    for json_path in sorted(split_dir.glob("*.json")):
        yield load_task(json_path)


ARC_AGI_V2_ROOT = Path(__file__).resolve().parents[2] / "reference" / "arc-agi-v2" / "data"
TRAIN_DIR = ARC_AGI_V2_ROOT / "training"
EVAL_DIR = ARC_AGI_V2_ROOT / "evaluation"
