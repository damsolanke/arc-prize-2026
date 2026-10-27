"""NearestColorSolver: each cell becomes the color of its nearest non-zero neighbor.

Uses Manhattan or Euclidean distance. Different from CA (local counts) and
induction (exact patterns).
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _nearest_color_fill(g: Grid) -> Grid:
    """Each zero cell gets color of its nearest non-zero cell (manhattan)."""
    out = g.copy()
    h, w = g.shape
    non_zero = [(r, c, int(g[r, c])) for r in range(h) for c in range(w) if g[r, c] != 0]
    if not non_zero:
        return out
    for r in range(h):
        for c in range(w):
            if g[r, c] != 0:
                continue
            nearest = min(non_zero, key=lambda nz: abs(nz[0] - r) + abs(nz[1] - c))
            out[r, c] = nearest[2]
    return out


def _fits(task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _nearest_color_fill(p.input)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class NearestColorSolver(BaseSolver):
    name = "nearest_color_solver"

    def solve(self, task: Task) -> Prediction:
        fits = _fits(task)
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if fits:
                try:
                    cands.append(_nearest_color_fill(test_pair.input))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["NearestColorSolver"]
