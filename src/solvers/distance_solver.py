"""DistanceSolver: color cells based on their distance to nearest non-zero seed.

Different from nearest (which copies color): here output cell encodes distance.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _distance_cap(g: Grid, fill_color: int, cap: int = 9) -> Grid:
    """Each cell: min(manhattan distance to nearest non-zero, cap), with fill color."""
    h, w = g.shape
    seeds = [(r, c) for r in range(h) for c in range(w) if g[r, c] != 0]
    if not seeds:
        return g.copy()
    out = np.zeros_like(g)
    for r in range(h):
        for c in range(w):
            d = min(abs(sr - r) + abs(sc - c) for sr, sc in seeds)
            if d == 0:
                out[r, c] = int(g[r, c])
            else:
                out[r, c] = fill_color if d <= cap else 0
    return out


def _fits(color: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _distance_cap(p.input, color)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class DistanceSolver(BaseSolver):
    name = "distance_solver"

    def solve(self, task: Task) -> Prediction:
        color = None
        for c in range(1, 10):
            if _fits(c, task):
                color = c
                break
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if color is not None:
                try:
                    cands.append(_distance_cap(test_pair.input, color))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["DistanceSolver"]
