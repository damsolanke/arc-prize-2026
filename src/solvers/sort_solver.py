"""SortSolver: sort rows or columns by a criterion.

Criteria:
- by count of non-zero cells
- by min value
- by max value

Orthogonal to rowcol_solver (which sorts cells within each row/col) — this
permutes rows/cols themselves.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _sort_rows_by(g: Grid, key_fn: Callable[[np.ndarray], int | float]) -> Grid:
    indices = sorted(range(g.shape[0]), key=lambda i: key_fn(g[i]))
    return g[indices].copy()


def _sort_cols_by(g: Grid, key_fn: Callable[[np.ndarray], int | float]) -> Grid:
    indices = sorted(range(g.shape[1]), key=lambda j: key_fn(g[:, j]))
    return g[:, indices].copy()


def _count_nonzero(row: np.ndarray) -> int:
    return int(np.count_nonzero(row))


def _max_val(row: np.ndarray) -> int:
    return int(row.max()) if row.size else 0


def _min_nonzero(row: np.ndarray) -> int:
    non_zero = row[row != 0]
    return int(non_zero.min()) if non_zero.size else 10


RULES = (
    ("rows_by_count", lambda g: _sort_rows_by(g, _count_nonzero)),
    ("cols_by_count", lambda g: _sort_cols_by(g, _count_nonzero)),
    ("rows_by_max", lambda g: _sort_rows_by(g, _max_val)),
    ("cols_by_max", lambda g: _sort_cols_by(g, _max_val)),
    ("rows_by_min_nz", lambda g: _sort_rows_by(g, _min_nonzero)),
    ("cols_by_min_nz", lambda g: _sort_cols_by(g, _min_nonzero)),
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


class SortSolver(BaseSolver):
    name = "sort_solver"

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


__all__ = ["SortSolver"]
