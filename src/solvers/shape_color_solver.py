"""ShapeColorSolver: recolor each object based on its shape property.

Property options: is_square, is_line, is_rectangle, has_hole.
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
        rs, cs = np.where(m)
        bbox = (int(rs.min()), int(cs.min()), int(rs.max()) + 1, int(cs.max()) + 1)
        comps.append((m, bbox, int(m.sum())))
    return comps


def _is_square(bbox, size) -> bool:
    r0, c0, r1, c1 = bbox
    h = r1 - r0
    w = c1 - c0
    return h == w and size == h * w


def _is_line(bbox, size) -> bool:
    r0, c0, r1, c1 = bbox
    h = r1 - r0
    w = c1 - c0
    return (h == 1 or w == 1) and size == h * w


def _recolor_by_shape(g: Grid, square_color: int, line_color: int, default: int) -> Grid:
    comps = _components(g)
    if not comps:
        return g.copy()
    out = g.copy()
    for mask, bbox, size in comps:
        if _is_square(bbox, size):
            out[mask] = square_color
        elif _is_line(bbox, size):
            out[mask] = line_color
        else:
            out[mask] = default
    return out


def _fits(sc: int, lc: int, dc: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _recolor_by_shape(p.input, sc, lc, dc)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class ShapeColorSolver(BaseSolver):
    name = "shape_color_solver"

    def solve(self, task: Task) -> Prediction:
        rule = None
        for sc in range(1, 10):
            for lc in range(1, 10):
                for dc in range(10):
                    if sc == lc or sc == dc or lc == dc:
                        continue
                    if _fits(sc, lc, dc, task):
                        rule = (sc, lc, dc)
                        break
                if rule:
                    break
            if rule:
                break
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                sc, lc, dc = rule
                try:
                    cands.append(_recolor_by_shape(test_pair.input, sc, lc, dc))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["ShapeColorSolver"]
