"""ColorCycleSolver: shift each color by a fixed amount (e.g., all +1, +2).

For tasks where every non-zero color gets shifted (mod 10 or with clamping).
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _cycle_plus(g: Grid, shift: int) -> Grid:
    """Add shift to non-zero cells (mod 10, but skip 0 as background)."""
    out = g.copy()
    for color in range(1, 10):
        new_color = ((color - 1 + shift) % 9) + 1
        mask = g == color
        out[mask] = new_color
    return out


def _fits(shift: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _cycle_plus(p.input, shift)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class ColorCycleSolver(BaseSolver):
    name = "color_cycle_solver"

    def solve(self, task: Task) -> Prediction:
        shift = None
        for s in range(1, 9):
            if _fits(s, task):
                shift = s
                break
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if shift is not None:
                try:
                    cands.append(_cycle_plus(test_pair.input, shift))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["ColorCycleSolver"]
