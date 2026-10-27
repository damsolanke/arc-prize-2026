"""MorphologySolver: dilate or erode non-zero cells.

Dilation: each non-zero cell spreads to its neighbors.
Erosion: only cells whose all neighbors are same-color survive.

Orthogonal mechanism: morphological operations are not transforms or
memorizations but geometric set operations on the grid mask.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _dilate(g: Grid, iterations: int = 1) -> Grid:
    h, w = g.shape
    out = g.copy()
    for _ in range(iterations):
        new = out.copy()
        for r in range(h):
            for c in range(w):
                if out[r, c] == 0:
                    # take first non-zero neighbor
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and out[nr, nc] != 0:
                            new[r, c] = out[nr, nc]
                            break
        out = new
    return out


def _erode(g: Grid, iterations: int = 1) -> Grid:
    h, w = g.shape
    out = g.copy()
    for _ in range(iterations):
        new = out.copy()
        for r in range(h):
            for c in range(w):
                if out[r, c] != 0:
                    # zero if any 4-neighbor is 0 or out-of-bounds
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < h and 0 <= nc < w) or out[nr, nc] == 0:
                            new[r, c] = 0
                            break
        out = new
    return out


RULES = (
    ("dilate_1", lambda g: _dilate(g, 1)),
    ("dilate_2", lambda g: _dilate(g, 2)),
    ("erode_1", lambda g: _erode(g, 1)),
    ("erode_2", lambda g: _erode(g, 2)),
)


def _fits(fn, task: Task) -> bool:
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


class MorphologySolver(BaseSolver):
    name = "morphology_solver"

    def solve(self, task: Task) -> Prediction:
        winners = [fn for _, fn in RULES if _fits(fn, task)]
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


__all__ = ["MorphologySolver"]
