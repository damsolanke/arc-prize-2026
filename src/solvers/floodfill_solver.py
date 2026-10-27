"""FloodFillSolver: detect seed cells and flood-fill connected regions.

Pattern: input has one or a few 'seed' cells of a specific color; output
floods outward from each seed until hitting a barrier. Different mechanism
than CA (which applies rules everywhere simultaneously) and induction
(which memorizes neighborhoods).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _flood_fill_barrier(g: Grid, seed_color: int, fill_color: int, barrier: int = 0) -> Grid:
    """From each seed_color cell, BFS-fill all neighbors == barrier with fill_color."""
    out = g.copy()
    h, w = g.shape
    seeds = []
    for r in range(h):
        for c in range(w):
            if g[r, c] == seed_color:
                seeds.append((r, c))
    visited = set()
    queue = deque(seeds)
    for s in seeds:
        visited.add(s)
    while queue:
        r, c = queue.popleft()
        if out[r, c] == barrier:
            out[r, c] = fill_color
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                if out[nr, nc] == barrier:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return out


def _fits_train_flood(seed_color: int, fill_color: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _flood_fill_barrier(p.input, seed_color, fill_color)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class FloodFillSolver(BaseSolver):
    name = "floodfill_solver"

    def _find_rule(self, task: Task) -> tuple[int, int] | None:
        for seed in range(10):
            for fill in range(10):
                if seed == 0 or seed == fill:
                    continue
                if _fits_train_flood(seed, fill, task):
                    return (seed, fill)
        return None

    def solve(self, task: Task) -> Prediction:
        rule = self._find_rule(task)
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                seed, fill = rule
                try:
                    cands.append(_flood_fill_barrier(test_pair.input, seed, fill))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["FloodFillSolver"]
