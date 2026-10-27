"""FillBetweenSolver: for tasks where pairs of same-colored dots get connected
by line segments in the output.

Different from LineSolver (which required exactly 2 endpoint cells of a color
horizontally or vertically aligned). Here we find pairs of same-color cells
and draw a line between them if they're aligned — generalizing to multi-pair
tasks and diagonals.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _find_cells(g: Grid, color: int) -> list[tuple[int, int]]:
    rs, cs = np.where(g == color)
    return list(zip(rs.tolist(), cs.tolist()))


def _draw_line(g: Grid, p1: tuple[int, int], p2: tuple[int, int], color: int) -> Grid:
    r1, c1 = p1
    r2, c2 = p2
    out = g.copy()
    if r1 == r2:
        for c in range(min(c1, c2), max(c1, c2) + 1):
            if out[r1, c] == 0:
                out[r1, c] = color
    elif c1 == c2:
        for r in range(min(r1, r2), max(r1, r2) + 1):
            if out[r, c1] == 0:
                out[r, c1] = color
    elif abs(r2 - r1) == abs(c2 - c1):
        # diagonal
        n = abs(r2 - r1) + 1
        dr = 1 if r2 > r1 else -1
        dc = 1 if c2 > c1 else -1
        for k in range(n):
            r, c = r1 + k * dr, c1 + k * dc
            if out[r, c] == 0:
                out[r, c] = color
    return out


def _try_rule(endpoint_color: int, line_color: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        pred = p.input.copy()
        cells = _find_cells(p.input, endpoint_color)
        for a, b in combinations(cells, 2):
            r1, c1 = a
            r2, c2 = b
            aligned = (r1 == r2) or (c1 == c2) or (abs(r1 - r2) == abs(c1 - c2))
            if aligned:
                pred = _draw_line(pred, a, b, line_color)
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class FillBetweenSolver(BaseSolver):
    name = "fill_between_solver"

    def _find_rule(self, task: Task) -> tuple[int, int] | None:
        for ep in range(1, 10):
            for ln in range(10):
                if ep == ln:
                    continue
                if _try_rule(ep, ln, task):
                    return (ep, ln)
        return None

    def solve(self, task: Task) -> Prediction:
        rule = self._find_rule(task)
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                ep, ln = rule
                pred = test_pair.input.copy()
                cells = _find_cells(test_pair.input, ep)
                for a, b in combinations(cells, 2):
                    r1, c1 = a
                    r2, c2 = b
                    aligned = (r1 == r2) or (c1 == c2) or (abs(r1 - r2) == abs(c1 - c2))
                    if aligned:
                        pred = _draw_line(pred, a, b, ln)
                cands.append(pred)
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["FillBetweenSolver"]
