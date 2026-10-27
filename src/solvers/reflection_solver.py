"""ReflectionCompletion: detect axis of mirror symmetry, extend missing half.

Many ARC tasks feature a partial pattern that should be mirror-symmetric.
The transformation: detect axis via partial-match, reflect the known half.

Axes tried: horizontal (top-bottom), vertical (left-right), and the two diagonals.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _reflect_horizontal(g: Grid) -> Grid:
    """Extend top half to symmetric bottom half (or vice versa)."""
    h, w = g.shape
    out = g.copy()
    for i in range(h // 2):
        for j in range(w):
            a = out[i, j]
            b = out[h - 1 - i, j]
            if a == 0 and b != 0:
                out[i, j] = b
            elif b == 0 and a != 0:
                out[h - 1 - i, j] = a
    return out


def _reflect_vertical(g: Grid) -> Grid:
    h, w = g.shape
    out = g.copy()
    for i in range(h):
        for j in range(w // 2):
            a = out[i, j]
            b = out[i, w - 1 - j]
            if a == 0 and b != 0:
                out[i, j] = b
            elif b == 0 and a != 0:
                out[i, w - 1 - j] = a
    return out


def _reflect_diag(g: Grid) -> Grid:
    """Reflect along main diagonal: out[i,j] and out[j,i] merged."""
    if g.shape[0] != g.shape[1]:
        return g.copy()
    h = g.shape[0]
    out = g.copy()
    for i in range(h):
        for j in range(i):
            a = out[i, j]
            b = out[j, i]
            if a == 0 and b != 0:
                out[i, j] = b
            elif b == 0 and a != 0:
                out[j, i] = a
    return out


def _reflect_antidiag(g: Grid) -> Grid:
    if g.shape[0] != g.shape[1]:
        return g.copy()
    h = g.shape[0]
    out = g.copy()
    for i in range(h):
        for j in range(h - 1 - i):
            a = out[i, j]
            b = out[h - 1 - j, h - 1 - i]
            if a == 0 and b != 0:
                out[i, j] = b
            elif b == 0 and a != 0:
                out[h - 1 - j, h - 1 - i] = a
    return out


REFLECT_RULES = (
    ("reflect_h", _reflect_horizontal),
    ("reflect_v", _reflect_vertical),
    ("reflect_diag", _reflect_diag),
    ("reflect_antidiag", _reflect_antidiag),
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


class ReflectionCompletionSolver(BaseSolver):
    name = "reflection_solver"

    def solve(self, task: Task) -> Prediction:
        winners = [fn for _, fn in REFLECT_RULES if _fits_train(fn, task)]
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


__all__ = ["ReflectionCompletionSolver"]
