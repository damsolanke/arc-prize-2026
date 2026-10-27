"""RowSummarySolver: output row = summary of input rows.

Examples:
- Output col j = max of input col j
- Output col j = majority of input col j
- Output col j = count of non-zero in input col j
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _col_max_row(g: Grid) -> Grid:
    return g.max(axis=0, keepdims=True).astype(np.int8)


def _col_min_nz_row(g: Grid) -> Grid:
    h, w = g.shape
    out = np.zeros((1, w), dtype=np.int8)
    for c in range(w):
        nz = g[:, c][g[:, c] != 0]
        out[0, c] = int(nz.min()) if nz.size else 0
    return out


def _col_majority_row(g: Grid) -> Grid:
    h, w = g.shape
    out = np.zeros((1, w), dtype=np.int8)
    for c in range(w):
        vals, cnt = np.unique(g[:, c], return_counts=True)
        out[0, c] = int(vals[cnt.argmax()])
    return out


def _row_max_col(g: Grid) -> Grid:
    return g.max(axis=1, keepdims=True).astype(np.int8)


def _row_min_nz_col(g: Grid) -> Grid:
    h, w = g.shape
    out = np.zeros((h, 1), dtype=np.int8)
    for r in range(h):
        nz = g[r][g[r] != 0]
        out[r, 0] = int(nz.min()) if nz.size else 0
    return out


RULES = (
    ("col_max_row", _col_max_row),
    ("col_min_nz_row", _col_min_nz_row),
    ("col_majority_row", _col_majority_row),
    ("row_max_col", _row_max_col),
    ("row_min_nz_col", _row_min_nz_col),
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


class RowSummarySolver(BaseSolver):
    name = "row_summary_solver"

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


__all__ = ["RowSummarySolver"]
