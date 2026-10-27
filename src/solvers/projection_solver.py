"""ProjectionSolver: project grid onto row or column axis.

Output = 1D array of summary statistics per row or per column:
  - count of nonzero
  - max value
  - sum (mod 10)
  - most-common nonzero
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _project_rows(g: Grid, agg: Callable[[np.ndarray], int]) -> Grid:
    h = g.shape[0]
    return np.array([[agg(g[r])] for r in range(h)], dtype=np.int8)


def _project_cols(g: Grid, agg: Callable[[np.ndarray], int]) -> Grid:
    w = g.shape[1]
    return np.array([[agg(g[:, c]) for c in range(w)]], dtype=np.int8)


def _count_nz(row: np.ndarray) -> int:
    return min(int(np.count_nonzero(row)), 9)


def _max_val(row: np.ndarray) -> int:
    return int(row.max()) if row.size else 0


def _most_common_nz(row: np.ndarray) -> int:
    nz = row[row != 0]
    if not nz.size:
        return 0
    vals, counts = np.unique(nz, return_counts=True)
    return int(vals[counts.argmax()])


RULES = (
    ("rows_count", lambda g: _project_rows(g, _count_nz)),
    ("rows_max", lambda g: _project_rows(g, _max_val)),
    ("rows_mc", lambda g: _project_rows(g, _most_common_nz)),
    ("cols_count", lambda g: _project_cols(g, _count_nz)),
    ("cols_max", lambda g: _project_cols(g, _max_val)),
    ("cols_mc", lambda g: _project_cols(g, _most_common_nz)),
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


class ProjectionSolver(BaseSolver):
    name = "projection_solver"

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


__all__ = ["ProjectionSolver"]
