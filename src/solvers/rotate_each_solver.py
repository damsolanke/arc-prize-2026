"""RotateEachObjectSolver: rotate each connected component by 90/180/270 in place.

Different from global rot90 (D8) which rotates the whole grid. Here each
object is independently rotated within its bounding box.
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
        comps.append((m, bbox))
    return comps


def _rotate_each(g: Grid, k: int) -> Grid:
    """Rotate each connected component by k * 90 degrees within its bbox."""
    comps = _components(g)
    if not comps:
        return g.copy()
    out = g.copy()
    # Clear all objects then draw rotated
    for mask, bbox in comps:
        out[mask] = 0
    for mask, bbox in comps:
        r0, c0, r1, c1 = bbox
        sub = g[r0:r1, c0:c1].copy()
        # Only keep this component's cells
        comp_mask = mask[r0:r1, c0:c1]
        sub_only = np.where(comp_mask, sub, 0)
        rotated = np.rot90(sub_only, k)
        # Place rotated at bbox — note shape may change for k=1,3
        rh, rw = rotated.shape
        if r0 + rh <= g.shape[0] and c0 + rw <= g.shape[1]:
            out[r0:r0 + rh, c0:c0 + rw] = np.where(rotated != 0, rotated, out[r0:r0 + rh, c0:c0 + rw])
    return out


def _flip_each(g: Grid, axis: int) -> Grid:
    """Flip each component within bbox, axis=0 (vertical) or 1 (horizontal)."""
    comps = _components(g)
    if not comps:
        return g.copy()
    out = g.copy()
    for mask, bbox in comps:
        out[mask] = 0
    for mask, bbox in comps:
        r0, c0, r1, c1 = bbox
        sub = g[r0:r1, c0:c1].copy()
        comp_mask = mask[r0:r1, c0:c1]
        sub_only = np.where(comp_mask, sub, 0)
        flipped = np.flip(sub_only, axis=axis)
        out[r0:r1, c0:c1] = np.where(flipped != 0, flipped, out[r0:r1, c0:c1])
    return out


RULES = (
    ("rot90_each", lambda g: _rotate_each(g, 1)),
    ("rot180_each", lambda g: _rotate_each(g, 2)),
    ("rot270_each", lambda g: _rotate_each(g, 3)),
    ("flip_each_h", lambda g: _flip_each(g, 1)),
    ("flip_each_v", lambda g: _flip_each(g, 0)),
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


class RotateEachObjectSolver(BaseSolver):
    name = "rotate_each_solver"

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


__all__ = ["RotateEachObjectSolver"]
