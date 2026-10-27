"""AccumulatorSolver: running counts/cumsum per row/col.

For tasks where output cells encode a running transformation along an axis.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _cumulative_left_to_right(g: Grid) -> Grid:
    """Each cell: keeps value if non-zero, else copies nearest non-zero on its left."""
    out = g.copy()
    h, w = g.shape
    for r in range(h):
        last = 0
        for c in range(w):
            if out[r, c] != 0:
                last = int(out[r, c])
            else:
                out[r, c] = last
    return out


def _cumulative_top_to_bottom(g: Grid) -> Grid:
    out = g.copy()
    h, w = g.shape
    for c in range(w):
        last = 0
        for r in range(h):
            if out[r, c] != 0:
                last = int(out[r, c])
            else:
                out[r, c] = last
    return out


def _cumulative_right_to_left(g: Grid) -> Grid:
    out = g.copy()
    h, w = g.shape
    for r in range(h):
        last = 0
        for c in range(w - 1, -1, -1):
            if out[r, c] != 0:
                last = int(out[r, c])
            else:
                out[r, c] = last
    return out


def _cumulative_bottom_to_top(g: Grid) -> Grid:
    out = g.copy()
    h, w = g.shape
    for c in range(w):
        last = 0
        for r in range(h - 1, -1, -1):
            if out[r, c] != 0:
                last = int(out[r, c])
            else:
                out[r, c] = last
    return out


RULES = (
    ("cum_ltr", _cumulative_left_to_right),
    ("cum_rtl", _cumulative_right_to_left),
    ("cum_ttb", _cumulative_top_to_bottom),
    ("cum_btt", _cumulative_bottom_to_top),
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


class AccumulatorSolver(BaseSolver):
    name = "accumulator_solver"

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


__all__ = ["AccumulatorSolver"]
