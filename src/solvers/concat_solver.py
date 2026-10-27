"""ConcatSolver: output is input concatenated with a TRANSFORMED copy of itself.

Extends AxisMirrorSolver's idea: not just mirror, but concat with various
transforms (rot90, rot180, etc.) along horizontal or vertical axis.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _concat_h(g: Grid, transform_fn) -> Grid:
    return np.concatenate([g, transform_fn(g)], axis=1)


def _concat_v(g: Grid, transform_fn) -> Grid:
    return np.concatenate([g, transform_fn(g)], axis=0)


TRANSFORMS_WITH_NAME = (
    ("rot90", lambda g: np.rot90(g, 1)),
    ("rot180", lambda g: np.rot90(g, 2)),
    ("rot270", lambda g: np.rot90(g, 3)),
    ("flip_h", lambda g: g[:, ::-1]),
    ("flip_v", lambda g: g[::-1, :]),
    ("transpose", lambda g: g.T),
)


def _build_rules():
    rules = []
    for tname, tfn in TRANSFORMS_WITH_NAME:
        rules.append((f"concat_h_{tname}", lambda g, f=tfn: _concat_h(g, f)))
        rules.append((f"concat_v_{tname}", lambda g, f=tfn: _concat_v(g, f)))
    return rules


RULES = tuple(_build_rules())


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


class ConcatSolver(BaseSolver):
    name = "concat_solver"

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


__all__ = ["ConcatSolver"]
