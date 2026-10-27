"""LegendSolver: detect a 'legend' region in input + apply its mapping.

Pattern: input has a small color-pair legend (e.g., 2-cell horizontal pairs)
at a specific location, and a content region. Output = content region with
legend mapping applied.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _detect_legend_top_row(g: Grid) -> dict[int, int] | None:
    """If top row looks like [a, b, a, c, a, d, ...] → build {b: a', c: a', ...}?
    Simpler: top row has pairs like [a, b, c, d]; treat (a,b) and (c,d) as legend."""
    h, w = g.shape
    if w < 4:
        return None
    row = g[0]
    # Look for even-length pairs
    if w % 2 == 0:
        mapping = {}
        for i in range(0, w, 2):
            src = int(row[i])
            dst = int(row[i + 1])
            if src == 0 or dst == 0:
                return None
            if src in mapping and mapping[src] != dst:
                return None
            mapping[src] = dst
        if len(mapping) > 0:
            return mapping
    return None


def _apply_legend(g: Grid) -> np.ndarray | None:
    """Extract legend from top row; apply mapping to rest."""
    mapping = _detect_legend_top_row(g)
    if not mapping:
        return None
    h, w = g.shape
    if h < 2:
        return None
    content = g[1:].copy()
    out = content.copy()
    for src, dst in mapping.items():
        out[content == src] = dst
    return out


def _fits(task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _apply_legend(p.input)
        except Exception:
            return False
        if pred is None or pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class LegendSolver(BaseSolver):
    name = "legend_solver"

    def solve(self, task: Task) -> Prediction:
        fits = _fits(task)
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if fits:
                pred = _apply_legend(test_pair.input)
                if pred is not None:
                    cands.append(pred)
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["LegendSolver"]
