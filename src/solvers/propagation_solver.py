"""PropagationSolver: cells of specific color emit trails in specific direction.

Pattern: input has sparse 'dot' cells; output extends each dot into a line
(e.g., vertical down, horizontal right, diagonal).
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _propagate_direction(g: Grid, dr: int, dc: int) -> Grid:
    h, w = g.shape
    out = g.copy()
    for r in range(h):
        for c in range(w):
            if g[r, c] == 0:
                continue
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w:
                if out[rr, cc] == 0:
                    out[rr, cc] = g[r, c]
                rr += dr
                cc += dc
    return out


DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]


def _fits_train(dr: int, dc: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _propagate_direction(p.input, dr, dc)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class PropagationSolver(BaseSolver):
    name = "propagation_solver"

    def solve(self, task: Task) -> Prediction:
        rule = None
        for dr, dc in DIRECTIONS:
            if _fits_train(dr, dc, task):
                rule = (dr, dc)
                break
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                dr, dc = rule
                try:
                    cands.append(_propagate_direction(test_pair.input, dr, dc))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["PropagationSolver"]
