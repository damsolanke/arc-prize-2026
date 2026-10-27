"""BoundarySolver: output highlights boundary cells (where neighbors differ).

Boundary detection = output is 1 at cells whose 4-neighbors have different
values, 0 otherwise. Orthogonal mechanism: detects cells by neighborhood
*variability* rather than exact pattern match.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _boundary_mask(g: Grid, mark_color: int = 1, bg_color: int = 0) -> Grid:
    h, w = g.shape
    out = np.full_like(g, bg_color)
    for r in range(h):
        for c in range(w):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != g[r, c]:
                    out[r, c] = mark_color
                    break
    return out


def _fits_boundary(mark: int, bg: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _boundary_mask(p.input, mark_color=mark, bg_color=bg)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class BoundarySolver(BaseSolver):
    name = "boundary_solver"

    def solve(self, task: Task) -> Prediction:
        rule = None
        for mark in range(1, 10):
            for bg in range(10):
                if mark == bg:
                    continue
                if _fits_boundary(mark, bg, task):
                    rule = (mark, bg)
                    break
            if rule:
                break

        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                mark, bg = rule
                try:
                    cands.append(_boundary_mask(test_pair.input, mark_color=mark, bg_color=bg))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["BoundarySolver"]
