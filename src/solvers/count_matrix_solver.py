"""CountMatrixSolver: output[r,c] = count of a specific color in input's row r or col c.

Different mechanism: each output cell encodes a statistic of its row/column in input.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _row_counts_matrix(g: Grid, target_color: int, fill_color: int) -> Grid:
    h, w = g.shape
    out = np.zeros_like(g)
    for r in range(h):
        count = int((g[r] == target_color).sum())
        for c in range(min(count, w)):
            out[r, c] = fill_color
    return out


def _col_counts_matrix(g: Grid, target_color: int, fill_color: int) -> Grid:
    h, w = g.shape
    out = np.zeros_like(g)
    for c in range(w):
        count = int((g[:, c] == target_color).sum())
        for r in range(min(count, h)):
            out[r, c] = fill_color
    return out


def _fits(fn, tc: int, fc: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = fn(p.input, tc, fc)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class CountMatrixSolver(BaseSolver):
    name = "count_matrix_solver"

    def solve(self, task: Task) -> Prediction:
        rule = None
        for fn in (_row_counts_matrix, _col_counts_matrix):
            for tc in range(1, 10):
                for fc in range(1, 10):
                    if _fits(fn, tc, fc, task):
                        rule = (fn, tc, fc)
                        break
                if rule:
                    break
            if rule:
                break
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                fn, tc, fc = rule
                try:
                    cands.append(fn(test_pair.input, tc, fc))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["CountMatrixSolver"]
