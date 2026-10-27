"""RowColInductionSolver: learn per-row or per-column transformations.

Many ARC tasks apply the SAME transformation independently to each row (or
column) of the input to produce the output. Examples: 'sort each row
ascending', 'reverse each column', 'shift each row left by count of zeros'.

This solver tries: (a) is there a row-wise transform that maps every train
input row to the corresponding output row? (b) column-wise? If yes, apply
at test time.

Orthogonal to 2D induction (which requires neighborhood rules) and to
composition (which applies one global transform).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _reverse_row(row: np.ndarray) -> np.ndarray:
    return row[::-1].copy()


def _sort_ascending(row: np.ndarray) -> np.ndarray:
    return np.sort(row)


def _sort_descending(row: np.ndarray) -> np.ndarray:
    return np.sort(row)[::-1]


def _shift_left(row: np.ndarray) -> np.ndarray:
    out = np.zeros_like(row)
    nz = row[row != 0]
    out[:len(nz)] = nz
    return out


def _shift_right(row: np.ndarray) -> np.ndarray:
    out = np.zeros_like(row)
    nz = row[row != 0]
    out[-len(nz):] = nz if len(nz) else out
    return out


def _count_nonzero(row: np.ndarray) -> np.ndarray:
    """Return a row of same length with count of non-zero at position 0, rest zero."""
    n = int(np.count_nonzero(row))
    out = np.zeros_like(row)
    if len(out) > 0:
        out[0] = min(n, 9)
    return out


ROW_RULES: tuple[tuple[str, Callable[[np.ndarray], np.ndarray]], ...] = (
    ("reverse", _reverse_row),
    ("sort_asc", _sort_ascending),
    ("sort_desc", _sort_descending),
    ("shift_left", _shift_left),
    ("shift_right", _shift_right),
)


def _check_row_rule(
    rule: Callable[[np.ndarray], np.ndarray],
    pair_inputs: list[np.ndarray],
    pair_outputs: list[np.ndarray],
) -> bool:
    for inp, out in zip(pair_inputs, pair_outputs):
        if inp.shape != out.shape:
            return False
        for r in range(inp.shape[0]):
            try:
                pred = rule(inp[r])
            except Exception:
                return False
            if not np.array_equal(pred, out[r]):
                return False
    return True


def _check_col_rule(
    rule: Callable[[np.ndarray], np.ndarray],
    pair_inputs: list[np.ndarray],
    pair_outputs: list[np.ndarray],
) -> bool:
    for inp, out in zip(pair_inputs, pair_outputs):
        if inp.shape != out.shape:
            return False
        for c in range(inp.shape[1]):
            try:
                pred = rule(inp[:, c])
            except Exception:
                return False
            if not np.array_equal(pred, out[:, c]):
                return False
    return True


def _apply_row_rule(rule: Callable, g: Grid) -> Grid:
    out = np.zeros_like(g)
    for r in range(g.shape[0]):
        out[r] = rule(g[r])
    return out


def _apply_col_rule(rule: Callable, g: Grid) -> Grid:
    out = np.zeros_like(g)
    for c in range(g.shape[1]):
        out[:, c] = rule(g[:, c])
    return out


class RowColInductionSolver(BaseSolver):
    name = "rowcol_solver"

    def _find_rule(self, task: Task) -> tuple[str, str, Callable] | None:
        pair_inputs = [p.input for p in task.train if p.output is not None]
        pair_outputs = [p.output for p in task.train if p.output is not None]
        if not pair_inputs:
            return None
        for name, rule in ROW_RULES:
            if _check_row_rule(rule, pair_inputs, pair_outputs):
                return ("row", name, rule)
            if _check_col_rule(rule, pair_inputs, pair_outputs):
                return ("col", name, rule)
        return None

    def solve(self, task: Task) -> Prediction:
        found = self._find_rule(task)
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if found is not None:
                axis, _name, rule = found
                try:
                    if axis == "row":
                        cands.append(_apply_row_rule(rule, test_pair.input))
                    else:
                        cands.append(_apply_col_rule(rule, test_pair.input))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["RowColInductionSolver"]
