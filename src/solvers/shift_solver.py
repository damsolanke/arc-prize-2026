"""ShiftSolver: output = input shifted by fixed (dr, dc) with zero-padding.

For tasks where every train pair shows a grid translated by the same fixed
offset. Searches dr, dc in [-5, 5] × [-5, 5].
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _shift(g: Grid, dr: int, dc: int, pad: int = 0) -> Grid:
    h, w = g.shape
    out = np.full_like(g, pad)
    for r in range(h):
        for c in range(w):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = g[r, c]
    return out


def _find_shift(task: Task) -> tuple[int, int] | None:
    pair_inputs = [p.input for p in task.train if p.output is not None]
    pair_outputs = [p.output for p in task.train if p.output is not None]
    if not pair_inputs:
        return None
    for dr in range(-5, 6):
        for dc in range(-5, 6):
            if dr == 0 and dc == 0:
                continue
            ok = True
            for inp, out in zip(pair_inputs, pair_outputs):
                if inp.shape != out.shape:
                    ok = False
                    break
                pred = _shift(inp, dr, dc)
                if not np.array_equal(pred, out):
                    ok = False
                    break
            if ok:
                return (dr, dc)
    return None


class ShiftSolver(BaseSolver):
    name = "shift_solver"

    def solve(self, task: Task) -> Prediction:
        rule = _find_shift(task)
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                dr, dc = rule
                try:
                    cands.append(_shift(test_pair.input, dr, dc))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["ShiftSolver"]
