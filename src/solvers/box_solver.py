"""BoxSolver: output = input with bounding-box outlines drawn around objects.

For each connected component, draw a rectangular outline of a specified color.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _draw_box(g: Grid, box_color: int) -> Grid:
    try:
        from scipy.ndimage import label
    except ImportError:
        return g.copy()
    out = g.copy()
    labeled, n = label(g != 0)
    for i in range(1, n + 1):
        m = labeled == i
        rs, cs = np.where(m)
        r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
        # Draw border
        for c in range(c0, c1 + 1):
            if out[r0, c] == 0:
                out[r0, c] = box_color
            if out[r1, c] == 0:
                out[r1, c] = box_color
        for r in range(r0, r1 + 1):
            if out[r, c0] == 0:
                out[r, c0] = box_color
            if out[r, c1] == 0:
                out[r, c1] = box_color
    return out


def _fits(color: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _draw_box(p.input, color)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class BoxSolver(BaseSolver):
    name = "box_solver"

    def solve(self, task: Task) -> Prediction:
        color = None
        for c in range(1, 10):
            if _fits(c, task):
                color = c
                break
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if color is not None:
                try:
                    cands.append(_draw_box(test_pair.input, color))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["BoxSolver"]
