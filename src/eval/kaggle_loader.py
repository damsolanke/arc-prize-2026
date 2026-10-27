"""Load ARC-AGI-2 2026 Kaggle data (flat-dict JSON format).

Kaggle distributes tasks as a single JSON per split (training_challenges.json,
training_solutions.json, evaluation_challenges.json, evaluation_solutions.json,
test_challenges.json, sample_submission.json). This is different from the
github arcprize/arc-agi-2 format which uses one JSON per task.

This loader converts Kaggle format to our Task/Pair dataclasses.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .loader import Pair, Task

KAGGLE_ROOT = Path(__file__).resolve().parents[2] / "data" / "arc-prize-2026-arc-agi-2"


def _to_grid(data: list[list[int]]) -> np.ndarray:
    arr = np.array(data, dtype=np.int8)
    if arr.ndim != 2:
        raise ValueError(f"Grid must be 2D, got shape {arr.shape}")
    if arr.min() < 0 or arr.max() > 9:
        raise ValueError(f"Grid values must be 0-9, got {arr.min()}-{arr.max()}")
    return arr


def load_kaggle_split(
    challenges_path: Path | str,
    solutions_path: Path | str | None = None,
) -> list[Task]:
    """Load Kaggle-format JSON. If solutions_path is None (test split), test outputs are None."""
    challenges_path = Path(challenges_path)
    with open(challenges_path) as f:
        challenges = json.load(f)

    solutions = None
    if solutions_path is not None:
        with open(solutions_path) as f:
            solutions = json.load(f)

    tasks: list[Task] = []
    for task_id in sorted(challenges):
        raw = challenges[task_id]
        train = tuple(Pair(_to_grid(p["input"]), _to_grid(p["output"])) for p in raw["train"])

        # test inputs come from challenges; outputs from solutions (if provided)
        test_pairs: list[Pair] = []
        sols = solutions[task_id] if solutions and task_id in solutions else None
        for i, p in enumerate(raw["test"]):
            out = None
            if sols is not None:
                # solutions file: list of grids, one per test input
                if isinstance(sols, list) and i < len(sols):
                    out = _to_grid(sols[i])
            test_pairs.append(Pair(input=_to_grid(p["input"]), output=out))
        tasks.append(Task(task_id=task_id, train=train, test=tuple(test_pairs)))
    return tasks


def load_kaggle_training() -> list[Task]:
    return load_kaggle_split(
        KAGGLE_ROOT / "arc-agi_training_challenges.json",
        KAGGLE_ROOT / "arc-agi_training_solutions.json",
    )


def load_kaggle_evaluation() -> list[Task]:
    return load_kaggle_split(
        KAGGLE_ROOT / "arc-agi_evaluation_challenges.json",
        KAGGLE_ROOT / "arc-agi_evaluation_solutions.json",
    )


def load_kaggle_test() -> list[Task]:
    """The 240 hidden-answer test tasks. Submission is scored against these on the LB."""
    return load_kaggle_split(KAGGLE_ROOT / "arc-agi_test_challenges.json")


def load_sample_submission() -> dict:
    path = KAGGLE_ROOT / "sample_submission.json"
    with open(path) as f:
        return json.load(f)


__all__ = [
    "load_kaggle_split",
    "load_kaggle_training",
    "load_kaggle_evaluation",
    "load_kaggle_test",
    "load_sample_submission",
    "KAGGLE_ROOT",
]
