"""PositionalSolver: output depends on (r, c) position, independent of input content.

For tasks like: 'output is chessboard pattern', 'output is striped grid',
'output has specific color at each position regardless of input'.

Learns a positional template from train pairs if output is identical across
train examples (same shape + same content — pure template). Then applies.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _extract_template(task: Task) -> Grid | None:
    """If all train outputs are identical grids, return that grid; else None."""
    outputs = [p.output for p in task.train if p.output is not None]
    if not outputs:
        return None
    first = outputs[0]
    for out in outputs[1:]:
        if out.shape != first.shape or not np.array_equal(out, first):
            return None
    return first.copy()


class PositionalSolver(BaseSolver):
    """Output is the same grid across all train pairs → memorize and return it."""

    name = "positional_solver"

    def solve(self, task: Task) -> Prediction:
        template = _extract_template(task)
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if template is not None:
                cands.append(template.copy())
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["PositionalSolver"]
