"""CopyFromTrainSolver: if test input exactly matches any train input, return
the corresponding train output.

Trivial but catches degenerate 'memorization' tasks. Useful as sanity check.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


class CopyFromTrainSolver(BaseSolver):
    name = "copy_from_train"

    def solve(self, task: Task) -> Prediction:
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            for train_pair in task.train:
                if train_pair.output is None:
                    continue
                if train_pair.input.shape == test_pair.input.shape and np.array_equal(
                    train_pair.input, test_pair.input
                ):
                    cands.append(train_pair.output.copy())
                    break
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["CopyFromTrainSolver"]
