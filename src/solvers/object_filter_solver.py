"""ObjectFilterSolver: filter connected components by spatial predicate.

Predicates inspired by Hodel's ARC-DSL:
- touches_border: objects with any cell on grid edge
- does_not_touch_border: strictly interior objects
- is_tallest: the tallest bounding box
- is_widest: the widest bounding box
- is_square: bounding box is square

Output = grid with only filtered objects kept (rest zeroed).
"""
from __future__ import annotations

from collections.abc import Callable

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
        mask = labeled == i
        rs, cs = np.where(mask)
        bbox = (int(rs.min()), int(cs.min()), int(rs.max()) + 1, int(cs.max()) + 1)
        comps.append((mask, bbox))
    return comps


def _touches_border(bbox: tuple[int, int, int, int], grid_shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h, w = grid_shape
    return r0 == 0 or c0 == 0 or r1 == h or c1 == w


def _filter_keep_border(g: Grid) -> Grid:
    comps = _components(g)
    if not comps:
        return np.zeros_like(g)
    out = np.zeros_like(g)
    for mask, bbox in comps:
        if _touches_border(bbox, g.shape):
            out[mask] = g[mask]
    return out


def _filter_keep_interior(g: Grid) -> Grid:
    comps = _components(g)
    if not comps:
        return np.zeros_like(g)
    out = np.zeros_like(g)
    for mask, bbox in comps:
        if not _touches_border(bbox, g.shape):
            out[mask] = g[mask]
    return out


def _filter_keep_tallest(g: Grid) -> Grid:
    comps = _components(g)
    if not comps:
        return np.zeros_like(g)
    tallest = max(comps, key=lambda c: c[1][2] - c[1][0])
    out = np.zeros_like(g)
    out[tallest[0]] = g[tallest[0]]
    return out


def _filter_keep_widest(g: Grid) -> Grid:
    comps = _components(g)
    if not comps:
        return np.zeros_like(g)
    widest = max(comps, key=lambda c: c[1][3] - c[1][1])
    out = np.zeros_like(g)
    out[widest[0]] = g[widest[0]]
    return out


FILTERS: tuple[tuple[str, Callable[[Grid], Grid]], ...] = (
    ("border", _filter_keep_border),
    ("interior", _filter_keep_interior),
    ("tallest", _filter_keep_tallest),
    ("widest", _filter_keep_widest),
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


class ObjectFilterSolver(BaseSolver):
    name = "object_filter_solver"

    def solve(self, task: Task) -> Prediction:
        winners = [fn for _, fn in FILTERS if _fits(fn, task)]
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


__all__ = ["ObjectFilterSolver"]
