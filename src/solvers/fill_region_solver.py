"""FillRegionSolver: fill entire grid with the most common non-background color."""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _fill_with_rarest(g: Grid) -> Grid:
    vals, counts = np.unique(g[g != 0], return_counts=True) if (g != 0).any() else (np.array([0]), np.array([1]))
    rare = int(vals[counts.argmin()]) if len(vals) else 0
    return np.full_like(g, rare)


def _fill_with_mean(g: Grid) -> Grid:
    if not (g != 0).any():
        return np.zeros_like(g)
    mean_val = int(np.round(g[g != 0].mean()))
    return np.full_like(g, mean_val)


RULES = (("rarest", _fill_with_rarest), ("mean", _fill_with_mean))


def _fits(fn, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = fn(p.input)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class FillRegionSolver(BaseSolver):
    name = "fill_region_solver"

    def solve(self, task: Task) -> Prediction:
        winners = [fn for _, fn in RULES if _fits(fn, task)]
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            for fn in winners[:2]:
                try:
                    cands.append(fn(test_pair.input))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["FillRegionSolver"]
