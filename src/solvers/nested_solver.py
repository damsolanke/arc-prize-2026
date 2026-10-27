"""NestedSolver: fractal-like — each cell in input is replaced by a scaled copy.

For each cell (r,c) in input: place a block of input-itself at position
(r*H, c*W) in output (where H,W = input.shape). The block is filled with
the color if cell was non-zero, else zeros.

This is the 'Sierpinski' style ARC task where input acts as both shape
and content.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _nested(g: Grid) -> Grid:
    h, w = g.shape
    out = np.zeros((h * h, w * w), dtype=np.int8)
    for r in range(h):
        for c in range(w):
            if g[r, c] != 0:
                out[r * h:(r + 1) * h, c * w:(c + 1) * w] = g
    return out


def _nested_pure_color(g: Grid, color: int) -> Grid:
    """Same but output block is pure color (not input shape)."""
    h, w = g.shape
    out = np.zeros((h * h, w * w), dtype=np.int8)
    for r in range(h):
        for c in range(w):
            if g[r, c] != 0:
                out[r * h:(r + 1) * h, c * w:(c + 1) * w] = color
    return out


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


class NestedSolver(BaseSolver):
    name = "nested_solver"

    def solve(self, task: Task) -> Prediction:
        winners = []
        if _fits(_nested, task):
            winners.append(_nested)
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


__all__ = ["NestedSolver"]
