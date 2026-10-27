"""CropByColorSolver: output = bounding-box crop around cells of a specific color."""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _crop_color(g: Grid, target_color: int) -> np.ndarray | None:
    mask = g == target_color
    if not mask.any():
        return None
    rs, cs = np.where(mask)
    r0, c0, r1, c1 = rs.min(), cs.min(), rs.max() + 1, cs.max() + 1
    return g[r0:r1, c0:c1].copy()


def _fits(color: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        crop = _crop_color(p.input, color)
        if crop is None or crop.shape != p.output.shape or not np.array_equal(crop, p.output):
            return False
    return True


class CropByColorSolver(BaseSolver):
    name = "crop_by_color_solver"

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
                crop = _crop_color(test_pair.input, color)
                if crop is not None:
                    cands.append(crop)
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["CropByColorSolver"]
