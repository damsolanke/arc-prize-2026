"""SubtractSolver: output = input with specific-color cells cleared.

For tasks: 'remove all X-colored cells, replace with background'. Orthogonal
to mask_solver (which keeps a color) — this removes a color.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _subtract_color(g: Grid, remove: int, replace: int = 0) -> Grid:
    out = g.copy()
    out[g == remove] = replace
    return out


def _fits(remove: int, replace: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _subtract_color(p.input, remove, replace)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class SubtractSolver(BaseSolver):
    name = "subtract_solver"

    def solve(self, task: Task) -> Prediction:
        rule = None
        for rem in range(1, 10):
            for rep in range(10):
                if rem == rep:
                    continue
                if _fits(rem, rep, task):
                    rule = (rem, rep)
                    break
            if rule:
                break

        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                rem, rep = rule
                try:
                    cands.append(_subtract_color(test_pair.input, rem, rep))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["SubtractSolver"]
