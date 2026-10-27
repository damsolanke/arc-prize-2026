"""CornerSolver: output is the bounding-box corners of the main object(s).

For tasks where the transformation extracts corner-only information from
a filled shape — different from both subgrid (region) and boundary (edges).
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _keep_corners_only(g: Grid) -> Grid:
    """Keep only the 4 corner cells of the nonzero bounding box."""
    mask = g != 0
    if not mask.any():
        return np.zeros_like(g)
    rs, cs = np.where(mask)
    r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
    out = np.zeros_like(g)
    for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
        out[r, c] = g[r, c]
    return out


def _fits(task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _keep_corners_only(p.input)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class CornerSolver(BaseSolver):
    name = "corner_solver"

    def solve(self, task: Task) -> Prediction:
        fits = _fits(task)
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if fits:
                try:
                    cands.append(_keep_corners_only(test_pair.input))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["CornerSolver"]
