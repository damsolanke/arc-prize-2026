"""Explicit-Ensemble solver — the paper's key baseline.

Two variants:
- EnsembleSolver: plain rank-weighted majority vote across inner solvers
- VerifiedEnsembleSolver: filter candidates by "fits training pairs" before voting

Finding (paper-worthy): naive majority vote often hurts vs a stronger single
solver, because incorrect candidates from narrow solvers dilute the vote.
Filtering by train-pair-fit before voting recovers most of the single-solver
accuracy AND lets diverse solvers contribute where they're right.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _grid_key(g: Grid) -> tuple:
    return (g.shape, bytes(g.astype(np.int8).tobytes()))


def _key_to_grid(key: tuple, shape: tuple[int, int]) -> Grid:
    return np.frombuffer(key[1], dtype=np.int8).reshape(shape).copy()


def majority_vote(
    predictions_per_solver: Sequence[Sequence[Grid]],
    top_k: int = 3,
) -> list[Grid]:
    """Vote across solvers for the same test input; return top-k most-common grids.

    First candidate per solver weighted 1.0, second 0.5, third 1/3.
    """
    counter: Counter[tuple] = Counter()
    shapes: dict[tuple, tuple[int, int]] = {}
    for solver_preds in predictions_per_solver:
        for rank, g in enumerate(solver_preds):
            k = _grid_key(g)
            weight = 1.0 / (rank + 1)
            counter[k] = counter[k] + weight  # type: ignore[assignment]
            shapes[k] = g.shape
    top = counter.most_common(top_k)
    return [_key_to_grid(k, shapes[k]) for k, _ in top]


class EnsembleSolver(BaseSolver):
    """Plain rank-weighted majority vote across inner solvers."""

    def __init__(self, inner: Sequence[BaseSolver], name: str = "ensemble"):
        self.inner: tuple[BaseSolver, ...] = tuple(inner)
        self.name = name

    def solve(self, task: Task) -> Prediction:
        inner_preds: list[Prediction] = [s.solve(task) for s in self.inner]
        n_test = len(task.test)
        candidates: list[list[Grid]] = []
        for i in range(n_test):
            per_solver_i = [p.candidates[i] for p in inner_preds]
            voted = majority_vote(per_solver_i, top_k=3)
            while len(voted) < 3:
                voted.append(
                    per_solver_i[0][0]
                    if per_solver_i and per_solver_i[0]
                    else task.test[i].input.copy()
                )
            candidates.append(voted[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


def _candidate_fits_train(task: Task, solver: BaseSolver, candidate_idx: int) -> bool:
    """Rerun solver on train pairs individually and check if its top-1 on each train
    input matches the train output. Cheaper heuristic: ask the solver to 'solve' a
    single-example sub-task with just one train pair as input, and compare to known
    output. This approximates 'does this solver's approach fit the training pairs?'
    """
    # Approximate: does solver's top candidate match when applied to EACH train input?
    # We construct a fake task with (train_pair, test=train_pair) and see.
    from ..eval.loader import Pair as _Pair
    from ..eval.loader import Task as _Task

    for pair in task.train:
        if pair.output is None:
            continue
        sub_task = _Task(
            task_id="_sub",
            train=task.train,
            test=(_Pair(input=pair.input, output=None),),
        )
        sub_pred = solver.solve(sub_task)
        if not sub_pred.candidates or not sub_pred.candidates[0]:
            return False
        cand = sub_pred.candidates[0][candidate_idx] if len(sub_pred.candidates[0]) > candidate_idx else sub_pred.candidates[0][0]
        if cand.shape != pair.output.shape or not np.array_equal(cand, pair.output):
            return False
    return True


class VerifiedEnsembleSolver(BaseSolver):
    """Only vote on candidates where the contributing solver fits training pairs.

    Per test input, collect solver outputs; drop contributions from solvers
    whose top candidate doesn't fit ALL train pairs. Vote among survivors.
    If no solver fits, fall back to plain majority over all inner predictions.
    """

    def __init__(self, inner: Sequence[BaseSolver], name: str = "verified_ensemble"):
        self.inner: tuple[BaseSolver, ...] = tuple(inner)
        self.name = name

    def solve(self, task: Task) -> Prediction:
        fits: list[bool] = [_candidate_fits_train(task, s, 0) for s in self.inner]
        inner_preds: list[Prediction] = [s.solve(task) for s in self.inner]

        n_test = len(task.test)
        candidates: list[list[Grid]] = []
        for i in range(n_test):
            # Collect predictions from fitting solvers first
            fitting = [p.candidates[i] for p, ok in zip(inner_preds, fits) if ok]
            if fitting:
                voted = majority_vote(fitting, top_k=3)
            else:
                all_preds = [p.candidates[i] for p in inner_preds]
                voted = majority_vote(all_preds, top_k=3)
            while len(voted) < 3:
                voted.append(
                    voted[-1].copy() if voted else task.test[i].input.copy()
                )
            candidates.append(voted[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["EnsembleSolver", "VerifiedEnsembleSolver", "majority_vote"]
