"""LargestShapeSolver: output = largest connected component as-is (bbox cropped)."""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _largest_component_cropped(g: Grid) -> np.ndarray | None:
    try:
        from scipy.ndimage import label
    except ImportError:
        return None
    labeled, n = label(g != 0)
    if n == 0:
        return None
    best_size = 0
    best_crop = None
    for i in range(1, n + 1):
        m = labeled == i
        size = int(m.sum())
        if size > best_size:
            rs, cs = np.where(m)
            r0, c0, r1, c1 = rs.min(), cs.min(), rs.max() + 1, cs.max() + 1
            crop = g[r0:r1, c0:c1].copy()
            # zero out other components within bbox
            mask_in_bbox = m[r0:r1, c0:c1]
            crop[~mask_in_bbox] = 0
            best_size = size
            best_crop = crop
    return best_crop


def _fits(task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        pred = _largest_component_cropped(p.input)
        if pred is None or pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class LargestShapeSolver(BaseSolver):
    name = "largest_shape_solver"

    def solve(self, task: Task) -> Prediction:
        fits = _fits(task)
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if fits:
                pred = _largest_component_cropped(test_pair.input)
                if pred is not None:
                    cands.append(pred)
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["LargestShapeSolver"]
