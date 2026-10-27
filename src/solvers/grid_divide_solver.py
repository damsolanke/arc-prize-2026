"""GridDivideSolver: input has grid divided by lines; output extracts one cell.

Pattern: input has a grid of sub-cells separated by line-color cells.
Output = the most-common / unique / specific sub-cell.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _find_divider_and_cells(g: Grid, divider_color: int) -> list[np.ndarray]:
    """Identify sub-regions separated by divider lines. Return list of subgrid arrays."""
    h, w = g.shape
    # Find rows and columns that are all divider
    row_mask = np.all(g == divider_color, axis=1)
    col_mask = np.all(g == divider_color, axis=0)
    if not row_mask.any() or not col_mask.any():
        return []

    # Row segments
    row_boundaries = [-1] + [i for i in range(h) if row_mask[i]] + [h]
    row_segs = [(row_boundaries[i] + 1, row_boundaries[i + 1]) for i in range(len(row_boundaries) - 1)]
    row_segs = [(a, b) for a, b in row_segs if b > a]

    col_boundaries = [-1] + [j for j in range(w) if col_mask[j]] + [w]
    col_segs = [(col_boundaries[i] + 1, col_boundaries[i + 1]) for i in range(len(col_boundaries) - 1)]
    col_segs = [(a, b) for a, b in col_segs if b > a]

    subgrids = []
    for r0, r1 in row_segs:
        for c0, c1 in col_segs:
            sg = g[r0:r1, c0:c1]
            if sg.size > 0:
                subgrids.append(sg.copy())
    return subgrids


def _pick_unique(g: Grid, divider_color: int) -> np.ndarray | None:
    """Return the subgrid that appears uniquely (once); others may be duplicated."""
    subs = _find_divider_and_cells(g, divider_color)
    if not subs:
        return None
    from collections import Counter
    keys = [s.tobytes() for s in subs]
    counts = Counter(keys)
    unique_keys = [k for k, c in counts.items() if c == 1]
    if len(unique_keys) == 1:
        idx = keys.index(unique_keys[0])
        return subs[idx]
    return None


def _fits(color: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        sub = _pick_unique(p.input, color)
        if sub is None or sub.shape != p.output.shape or not np.array_equal(sub, p.output):
            return False
    return True


class GridDivideSolver(BaseSolver):
    name = "grid_divide_solver"

    def solve(self, task: Task) -> Prediction:
        color = None
        for c in range(10):
            if _fits(c, task):
                color = c
                break
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if color is not None:
                sub = _pick_unique(test_pair.input, color)
                if sub is not None:
                    cands.append(sub)
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["GridDivideSolver"]
