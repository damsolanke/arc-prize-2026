"""FrameFillSolver: detect rectangular frame of color X, fill interior with Y."""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _find_frame_bbox(g: Grid, frame_color: int):
    """Return (r0, c0, r1, c1) of tightest bbox where perimeter is frame_color."""
    try:
        from scipy.ndimage import label
    except ImportError:
        return None
    mask = g == frame_color
    if not mask.any():
        return None
    labeled, n = label(mask)
    for i in range(1, n + 1):
        m = labeled == i
        rs, cs = np.where(m)
        r0, c0, r1, c1 = rs.min(), cs.min(), rs.max() + 1, cs.max() + 1
        # Check if perimeter is full of frame_color
        top = g[r0, c0:c1]
        bot = g[r1 - 1, c0:c1]
        left = g[r0:r1, c0]
        right = g[r0:r1, c1 - 1]
        if (top == frame_color).all() and (bot == frame_color).all() and (left == frame_color).all() and (right == frame_color).all():
            return (r0, c0, r1, c1)
    return None


def _fill_frame_interior(g: Grid, frame_color: int, fill_color: int) -> Grid | None:
    bbox = _find_frame_bbox(g, frame_color)
    if bbox is None:
        return None
    r0, c0, r1, c1 = bbox
    out = g.copy()
    if r1 - r0 > 2 and c1 - c0 > 2:
        out[r0 + 1:r1 - 1, c0 + 1:c1 - 1] = fill_color
    return out


def _fits(fc: int, fill: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _fill_frame_interior(p.input, fc, fill)
        except Exception:
            return False
        if pred is None or pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class FrameFillSolver(BaseSolver):
    name = "frame_fill_solver"

    def solve(self, task: Task) -> Prediction:
        rule = None
        for fc in range(1, 10):
            for fill in range(10):
                if fc == fill:
                    continue
                if _fits(fc, fill, task):
                    rule = (fc, fill)
                    break
            if rule:
                break
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                fc, fill = rule
                pred = _fill_frame_interior(test_pair.input, fc, fill)
                if pred is not None:
                    cands.append(pred)
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["FrameFillSolver"]
