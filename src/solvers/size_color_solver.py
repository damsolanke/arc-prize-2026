"""SizeColorSolver: recolor objects based on their size.

Pattern: each connected component in input is recolored by size (largest→one
color, smallest→another, middle→third). Different from ObjectIsolationSolver
which just selects one component.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _components(g: Grid):
    try:
        from scipy.ndimage import label
    except ImportError:
        return []
    labeled, n = label(g != 0)
    comps = []
    for i in range(1, n + 1):
        m = labeled == i
        comps.append((m, int(m.sum())))
    return comps


def _recolor_by_size(g: Grid, large_color: int, small_color: int) -> Grid:
    comps = _components(g)
    if not comps:
        return g.copy()
    out = g.copy()
    sizes = sorted({s for _, s in comps})
    max_size = sizes[-1]
    min_size = sizes[0]
    for mask, size in comps:
        if size == max_size:
            out[mask] = large_color
        elif size == min_size:
            out[mask] = small_color
    return out


def _fits(task: Task, large: int, small: int) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _recolor_by_size(p.input, large, small)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class SizeColorSolver(BaseSolver):
    name = "size_color_solver"

    def solve(self, task: Task) -> Prediction:
        rule = None
        for large in range(1, 10):
            for small in range(10):
                if large == small:
                    continue
                if _fits(task, large, small):
                    rule = (large, small)
                    break
            if rule:
                break

        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                large, small = rule
                try:
                    cands.append(_recolor_by_size(test_pair.input, large, small))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["SizeColorSolver"]
