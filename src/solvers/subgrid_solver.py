"""SubgridSolver: output is one of the non-background rectangular regions of input.

ARC pattern: input contains multiple 'sub-grids' separated by a background
divider; output is the one with specific property (largest, smallest,
most-colored).
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _find_subgrids(g: Grid, divider: int = 0) -> list[tuple[int, int, int, int, np.ndarray]]:
    """Return list of (r0, c0, r1, c1, subgrid_array) for each non-divider rectangle."""
    try:
        from scipy.ndimage import label
    except ImportError:
        return []
    mask = g != divider
    labeled, n = label(mask)
    subs = []
    for i in range(1, n + 1):
        m = labeled == i
        rs, cs = np.where(m)
        r0, c0, r1, c1 = int(rs.min()), int(cs.min()), int(rs.max()) + 1, int(cs.max()) + 1
        # Check if the bounding box is entirely filled with non-divider cells
        box = g[r0:r1, c0:c1]
        if (box != divider).all():
            subs.append((r0, c0, r1, c1, box.copy()))
    return subs


def _pick_largest(g: Grid):
    subs = _find_subgrids(g)
    if not subs:
        return None
    return max(subs, key=lambda s: s[4].size)[4]


def _pick_smallest(g: Grid):
    subs = _find_subgrids(g)
    if not subs:
        return None
    return min(subs, key=lambda s: s[4].size)[4]


def _pick_most_distinct_colors(g: Grid):
    subs = _find_subgrids(g)
    if not subs:
        return None
    return max(subs, key=lambda s: len(set(s[4].flatten().tolist())))[4]


PICK_RULES = (
    ("largest", _pick_largest),
    ("smallest", _pick_smallest),
    ("most_colors", _pick_most_distinct_colors),
)


def _fits_train(fn, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = fn(p.input)
        except Exception:
            return False
        if pred is None or pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class SubgridSolver(BaseSolver):
    name = "subgrid_solver"

    def solve(self, task: Task) -> Prediction:
        winners = [fn for _, fn in PICK_RULES if _fits_train(fn, task)]
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            for fn in winners[:2]:
                try:
                    out = fn(test_pair.input)
                    if out is not None:
                        cands.append(out)
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["SubgridSolver"]
