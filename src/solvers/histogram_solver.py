"""HistogramSolver: output is a bar chart / histogram representation of input colors.

For tasks like: 'output has row per input color, height = count'. Or 1D bar
chart variants. Orthogonal mechanism: output is a SUMMARY of input statistics,
not a geometric/pixel transform of input.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _bar_vertical(g: Grid) -> Grid:
    """Output: one column per non-zero color in input, height = count; fg color = color."""
    counts = {}
    for c in np.unique(g):
        if c != 0:
            counts[int(c)] = int((g == c).sum())
    if not counts:
        return np.zeros_like(g)
    max_count = max(counts.values())
    colors_sorted = sorted(counts, key=counts.get, reverse=True)
    w = len(colors_sorted)
    out = np.zeros((max_count, w), dtype=np.int8)
    for j, col in enumerate(colors_sorted):
        out[-counts[col]:, j] = col
    return out


def _bar_horizontal(g: Grid) -> Grid:
    counts = {}
    for c in np.unique(g):
        if c != 0:
            counts[int(c)] = int((g == c).sum())
    if not counts:
        return np.zeros_like(g)
    max_count = max(counts.values())
    colors_sorted = sorted(counts, key=counts.get, reverse=True)
    h = len(colors_sorted)
    out = np.zeros((h, max_count), dtype=np.int8)
    for i, col in enumerate(colors_sorted):
        out[i, :counts[col]] = col
    return out


HISTO_RULES = (
    ("bar_vertical", _bar_vertical),
    ("bar_horizontal", _bar_horizontal),
)


def _fits_train(fn, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = fn(p.input)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class HistogramSolver(BaseSolver):
    name = "histogram_solver"

    def solve(self, task: Task) -> Prediction:
        winners = [fn for _, fn in HISTO_RULES if _fits_train(fn, task)]
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            for fn in winners[:2]:
                try:
                    cands.append(fn(test_pair.input))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["HistogramSolver"]
