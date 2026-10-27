"""CountOutputSolver: output is a 1xN or Nx1 grid where N = some count."""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _count_components(g: Grid) -> int:
    try:
        from scipy.ndimage import label
    except ImportError:
        return 0
    _, n = label(g != 0)
    return n


def _horizontal_bar(g: Grid, color: int) -> np.ndarray:
    n = _count_components(g)
    return np.full((1, max(n, 1)), color, dtype=np.int8)


def _vertical_bar(g: Grid, color: int) -> np.ndarray:
    n = _count_components(g)
    return np.full((max(n, 1), 1), color, dtype=np.int8)


RULES = (
    ("h_bar", _horizontal_bar),
    ("v_bar", _vertical_bar),
)


def _fits(fn, color: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = fn(p.input, color)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class CountOutputSolver(BaseSolver):
    name = "count_output_solver"

    def solve(self, task: Task) -> Prediction:
        rule = None
        for _, fn in RULES:
            for color in range(1, 10):
                if _fits(fn, color, task):
                    rule = (fn, color)
                    break
            if rule:
                break
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                fn, color = rule
                try:
                    cands.append(fn(test_pair.input, color))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["CountOutputSolver"]
