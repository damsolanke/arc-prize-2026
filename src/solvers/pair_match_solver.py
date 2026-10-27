"""PairMatchSolver: for each color, find its spatial pairs and apply a rule.

Pattern: input has pairs of cells of the same color; output marks their
midpoint or fills between them. Different from line_solver (which only does
H/V lines) and fill_between (which requires any alignment).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _find_cells(g: Grid, color: int) -> list[tuple[int, int]]:
    rs, cs = np.where(g == color)
    return list(zip(rs.tolist(), cs.tolist()))


def _pair_midpoints(g: Grid, pair_color: int, mark_color: int) -> Grid:
    """Mark midpoint of each pair of pair_color cells with mark_color."""
    out = g.copy()
    cells = _find_cells(g, pair_color)
    for a, b in combinations(cells, 2):
        mr = (a[0] + b[0]) // 2
        mc = (a[1] + b[1]) // 2
        if 0 <= mr < g.shape[0] and 0 <= mc < g.shape[1]:
            if out[mr, mc] == 0:
                out[mr, mc] = mark_color
    return out


def _fits(pair_color: int, mark_color: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _pair_midpoints(p.input, pair_color, mark_color)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class PairMatchSolver(BaseSolver):
    name = "pair_match_solver"

    def solve(self, task: Task) -> Prediction:
        rule = None
        for pc in range(1, 10):
            for mc in range(10):
                if pc == mc:
                    continue
                if _fits(pc, mc, task):
                    rule = (pc, mc)
                    break
            if rule:
                break

        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                pc, mc = rule
                try:
                    cands.append(_pair_midpoints(test_pair.input, pc, mc))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["PairMatchSolver"]
