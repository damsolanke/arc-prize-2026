"""PeriodicityRepairSolver: detect 2D periodicity in grid, vote across tile copies
to repair noisy cells.

Many ARC tasks feature a grid that's *almost* a tiling of a small pattern with
a few cells perturbed. Repairing = vote each cell via its periodic partners.

This solver:
1. For grid of size (h, w), try divisors (rh, rw) where rh | h and rw | w
2. For each candidate tile shape (h/rh, w/rw), vote each cell with its (rh * rw)
   tile-copy counterparts
3. Pick the (rh, rw) that (a) fits train pairs exactly, or (b) maximizes
   train cell-match rate

Orthogonal to all existing solvers — targets 'almost-periodic grid' tasks.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _vote_periodic(g: Grid, rh: int, rw: int) -> Grid:
    """Given tile-replication (rh, rw), vote each cell among rh*rw periodic copies."""
    h, w = g.shape
    tile_h = h // rh
    tile_w = w // rw
    if tile_h <= 0 or tile_w <= 0:
        return g.copy()
    # tile_vote[r, c] collects the rh*rw values that should all be equal
    out = np.zeros_like(g)
    for r in range(tile_h):
        for c in range(tile_w):
            values = []
            for ir in range(rh):
                for ic in range(rw):
                    values.append(int(g[r + ir * tile_h, c + ic * tile_w]))
            voted = Counter(values).most_common(1)[0][0]
            for ir in range(rh):
                for ic in range(rw):
                    out[r + ir * tile_h, c + ic * tile_w] = voted
    return out


def _best_periodicity(pair_inputs: list[Grid], pair_outputs: list[Grid]) -> tuple[int, int] | None:
    """Find (rh, rw) such that _vote_periodic(input, rh, rw) == output for all pairs."""
    if not pair_inputs:
        return None
    first_shape = pair_inputs[0].shape
    for inp in pair_inputs + pair_outputs:
        if inp.shape != first_shape:
            return None

    h, w = first_shape
    best: tuple[int, int] | None = None
    best_match = 0
    for rh in _divisors(h):
        for rw in _divisors(w):
            if rh == 1 and rw == 1:
                continue  # trivial: identity
            all_match = True
            for inp, out in zip(pair_inputs, pair_outputs):
                voted = _vote_periodic(inp, rh, rw)
                if not np.array_equal(voted, out):
                    all_match = False
                    break
            if all_match:
                # prefer smaller rh*rw? (smaller means finer tile, more copies voting)
                # pick first match (rh, rw) sorted ascending
                return (rh, rw)
            # fallback: track best partial match
            match_count = 0
            total = 0
            for inp, out in zip(pair_inputs, pair_outputs):
                voted = _vote_periodic(inp, rh, rw)
                match_count += int((voted == out).sum())
                total += out.size
            if total > 0 and match_count > best_match:
                best_match = match_count
                best = (rh, rw)
    return best


class PeriodicityRepairSolver(BaseSolver):
    name = "periodicity_solver"

    def solve(self, task: Task) -> Prediction:
        pair_inputs = [p.input for p in task.train if p.output is not None]
        pair_outputs = [p.output for p in task.train if p.output is not None]
        rule = _best_periodicity(pair_inputs, pair_outputs)  # type: ignore[arg-type]

        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                rh, rw = rule
                voted = _vote_periodic(test_pair.input, rh, rw)
                cands.append(voted)
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["PeriodicityRepairSolver"]
