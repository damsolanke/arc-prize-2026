"""StepPatternSolver: detect staircase / step patterns and continue them.

For tasks where input shows a partial step pattern and output completes it.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _diagonal_stairs(g: Grid, color: int) -> Grid:
    """Draw main-diagonal cells with specified color."""
    h, w = g.shape
    out = g.copy()
    for i in range(min(h, w)):
        if out[i, i] == 0:
            out[i, i] = color
    return out


def _anti_diagonal_stairs(g: Grid, color: int) -> Grid:
    h, w = g.shape
    out = g.copy()
    for i in range(min(h, w)):
        if out[i, w - 1 - i] == 0:
            out[i, w - 1 - i] = color
    return out


def _fits(fn, color, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = fn(p.input, color)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class StepPatternSolver(BaseSolver):
    name = "step_pattern_solver"

    def solve(self, task: Task) -> Prediction:
        rule = None
        for fn in (_diagonal_stairs, _anti_diagonal_stairs):
            for color in range(1, 10):
                if _fits(fn, color, task):
                    rule = (fn, color)
                    break
            if rule:
                break
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                fn, color = rule
                try:
                    cands.append(fn(test_pair.input, color))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["StepPatternSolver"]
