"""OutlineSolver: add a border of specific color around input.

For tasks like 'add a 1-pixel frame of color X around grid'. Orthogonal
mechanism from existing solvers: adds to output perimeter, doesn't transform
the interior.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _add_outline(g: Grid, border_color: int, thickness: int = 1) -> Grid:
    h, w = g.shape
    new_h, new_w = h + 2 * thickness, w + 2 * thickness
    out = np.full((new_h, new_w), border_color, dtype=np.int8)
    out[thickness:thickness + h, thickness:thickness + w] = g
    return out


def _remove_outline(g: Grid, thickness: int = 1) -> Grid:
    h, w = g.shape
    if h <= 2 * thickness or w <= 2 * thickness:
        return g.copy()
    return g[thickness:h - thickness, thickness:w - thickness].copy()


def _fits_add_outline(color: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _add_outline(p.input, color)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


def _fits_remove_outline(task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _remove_outline(p.input)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class OutlineSolver(BaseSolver):
    name = "outline_solver"

    def solve(self, task: Task) -> Prediction:
        candidates: list[list[Grid]] = []
        # Check for add-outline rule (per color)
        add_color: int | None = None
        for c in range(10):
            if _fits_add_outline(c, task):
                add_color = c
                break
        remove_fit = _fits_remove_outline(task)

        for test_pair in task.test:
            cands: list[Grid] = []
            if add_color is not None:
                cands.append(_add_outline(test_pair.input, add_color))
            if remove_fit:
                cands.append(_remove_outline(test_pair.input))
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["OutlineSolver"]
