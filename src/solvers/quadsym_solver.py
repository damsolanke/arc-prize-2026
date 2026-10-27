"""QuadSymmetrySolver: make grid 4-fold symmetric via OR of all 4 rotations.

For tasks where output is the union of the grid and its 3 rotations,
creating 4-way rotational symmetry.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _rot_symmetrize(g: Grid) -> Grid:
    if g.shape[0] != g.shape[1]:
        return g.copy()
    out = g.copy()
    for k in range(1, 4):
        rotated = np.rot90(g, k)
        mask = rotated != 0
        out[(out == 0) & mask] = rotated[(out == 0) & mask]
    return out


def _flip_symmetrize_all(g: Grid) -> Grid:
    """Fill zeros using any of the D8 reflections/rotations."""
    out = g.copy()
    rotations = [
        np.rot90(g, 1),
        np.rot90(g, 2),
        np.rot90(g, 3),
        g[:, ::-1],
        g[::-1, :],
    ]
    for rot in rotations:
        if rot.shape != g.shape:
            continue
        mask = (out == 0) & (rot != 0)
        out[mask] = rot[mask]
    return out


RULES = (
    ("rot_symmetrize", _rot_symmetrize),
    ("flip_symmetrize", _flip_symmetrize_all),
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


class QuadSymmetrySolver(BaseSolver):
    name = "quad_symmetry_solver"

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


__all__ = ["QuadSymmetrySolver"]
