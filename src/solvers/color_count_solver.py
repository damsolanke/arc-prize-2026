"""Solvers whose output depends on color counts in the input.

Captures ARC tasks like:
- "output is 1x1 grid whose color = most-common input color"
- "output is a solid grid with most/least common color"
- "output colors the most-frequent color of input 2x everywhere"
- Bar-chart / histogram outputs

Orthogonal to transform/composition/induction families.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _color_counts(g: Grid) -> dict[int, int]:
    vals, counts = np.unique(g, return_counts=True)
    return dict(zip(vals.tolist(), counts.tolist()))


def _most_common_nonzero(g: Grid) -> int:
    cc = _color_counts(g)
    non_zero = {c: n for c, n in cc.items() if c != 0}
    if not non_zero:
        return 0
    return max(non_zero, key=lambda k: (non_zero[k], -k))


def _least_common_nonzero(g: Grid) -> int:
    cc = _color_counts(g)
    non_zero = {c: n for c, n in cc.items() if c != 0}
    if not non_zero:
        return 0
    return min(non_zero, key=lambda k: (non_zero[k], k))


def solid_1x1_most_common(g: Grid) -> Grid:
    return np.array([[_most_common_nonzero(g)]], dtype=np.int8)


def solid_1x1_least_common(g: Grid) -> Grid:
    return np.array([[_least_common_nonzero(g)]], dtype=np.int8)


def fill_with_most_common_nonzero(g: Grid) -> Grid:
    return np.full_like(g, _most_common_nonzero(g))


def fill_with_least_common_nonzero(g: Grid) -> Grid:
    return np.full_like(g, _least_common_nonzero(g))


def keep_only_most_common(g: Grid) -> Grid:
    """Zero out all cells except the most-common non-zero color."""
    mc = _most_common_nonzero(g)
    out = np.zeros_like(g)
    out[g == mc] = mc
    return out


def keep_only_least_common(g: Grid) -> Grid:
    lc = _least_common_nonzero(g)
    out = np.zeros_like(g)
    out[g == lc] = lc
    return out


COLOR_COUNT_TRANSFORMS: tuple[tuple[str, Callable[[Grid], Grid]], ...] = (
    ("solid_1x1_most_common", solid_1x1_most_common),
    ("solid_1x1_least_common", solid_1x1_least_common),
    ("fill_with_most_common_nonzero", fill_with_most_common_nonzero),
    ("fill_with_least_common_nonzero", fill_with_least_common_nonzero),
    ("keep_only_most_common", keep_only_most_common),
    ("keep_only_least_common", keep_only_least_common),
)


def _fits_train(fn, task: Task) -> bool:
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


class ColorCountSolver(BaseSolver):
    """Try each count-based transform; use whichever fits all train pairs."""

    name = "color_count_solver"

    def solve(self, task: Task) -> Prediction:
        winners = [fn for _, fn in COLOR_COUNT_TRANSFORMS if _fits_train(fn, task)]

        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            for fn in winners[:3]:
                try:
                    cands.append(fn(test_pair.input))
                except Exception:
                    cands.append(test_pair.input.copy())
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["ColorCountSolver", "COLOR_COUNT_TRANSFORMS"]
