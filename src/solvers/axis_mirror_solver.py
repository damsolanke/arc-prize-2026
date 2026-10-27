"""AxisMirrorSolver: output has input mirrored/duplicated along specific axis.

For tasks where output is input concatenated with a mirror image
(horizontally or vertically).
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _mirror_right(g: Grid) -> Grid:
    return np.concatenate([g, g[:, ::-1]], axis=1)


def _mirror_left(g: Grid) -> Grid:
    return np.concatenate([g[:, ::-1], g], axis=1)


def _mirror_down(g: Grid) -> Grid:
    return np.concatenate([g, g[::-1, :]], axis=0)


def _mirror_up(g: Grid) -> Grid:
    return np.concatenate([g[::-1, :], g], axis=0)


RULES = (
    ("mirror_right", _mirror_right),
    ("mirror_left", _mirror_left),
    ("mirror_down", _mirror_down),
    ("mirror_up", _mirror_up),
)


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


class AxisMirrorSolver(BaseSolver):
    name = "axis_mirror_solver"

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


__all__ = ["AxisMirrorSolver"]
