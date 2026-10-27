"""HolesSolver: detect holes (enclosed zero-regions) in objects and fill/mark them.

Different from fill_holes (which fills inside object boundaries). This solver
marks ONLY the hole cells with a specific color.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _mark_holes(g: Grid, hole_color: int) -> Grid:
    try:
        from scipy.ndimage import binary_fill_holes  # noqa: F401
    except ImportError:
        return g.copy()
    out = np.zeros_like(g)
    # For each non-bg color, find hole cells
    for color in np.unique(g):
        if color == 0:
            continue
        mask = g == color
        filled = binary_fill_holes(mask)
        hole_mask = filled & ~mask
        out[hole_mask] = hole_color
    return out


def _mark_only_holes(g: Grid, hole_color: int) -> Grid:
    """Output = hole cells only; rest zero."""
    return _mark_holes(g, hole_color)


def _fits(color: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _mark_only_holes(p.input, color)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class HolesSolver(BaseSolver):
    name = "holes_solver"

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
                    cands.append(_mark_only_holes(test_pair.input, color))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["HolesSolver"]
