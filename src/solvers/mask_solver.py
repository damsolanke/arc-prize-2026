"""MaskSolver: output is a color-masked / extracted / replaced version of input.

Common ARC pattern: output keeps only cells where input == specific_color,
OR output = input with specific_color swapped with another. These are
variants on the color-map solver but with more specific detection.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _keep_only(g: Grid, keep_color: int, fill: int = 0) -> Grid:
    out = np.full_like(g, fill)
    out[g == keep_color] = keep_color
    return out


def _remove_color(g: Grid, remove: int, replacement: int = 0) -> Grid:
    out = g.copy()
    out[g == remove] = replacement
    return out


def _binarize(g: Grid, foreground: int, background_color: int = 0, fg_out: int = 1) -> Grid:
    out = np.full_like(g, background_color)
    out[g == foreground] = fg_out
    return out


def _fits_train(fn: Callable[[Grid], Grid], task: Task) -> bool:
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


class MaskSolver(BaseSolver):
    name = "mask_solver"

    def _find_rule(self, task: Task) -> Callable[[Grid], Grid] | None:
        # Try: keep-only for each color
        for c in range(10):
            fn = lambda g, cc=c: _keep_only(g, cc)  # noqa: E731
            if _fits_train(fn, task):
                return fn
        # Try: remove color
        for c in range(10):
            for rep in range(10):
                if c == rep:
                    continue
                fn = lambda g, cc=c, rr=rep: _remove_color(g, cc, rr)  # noqa: E731
                if _fits_train(fn, task):
                    return fn
        return None

    def solve(self, task: Task) -> Prediction:
        rule = self._find_rule(task)
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                try:
                    cands.append(rule(test_pair.input))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["MaskSolver"]
