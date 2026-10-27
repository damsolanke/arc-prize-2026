"""ARC scoring. Exact-match, 2 attempts per test input per 2026 Kaggle sample_submission.

The 2026 ARC-AGI-2 Kaggle submission format specifies exactly
`{task_id: [{attempt_1, attempt_2}, ...per test input]}`. Per the official
github.com/arcprize/arc-agi-2 README: "The test-taker is allowed 2 trials".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loader import Grid, Task

N_ATTEMPTS = 2  # 2026 rule: attempt_1 and attempt_2 only


def exact_match(pred: Grid, target: Grid) -> bool:
    if pred.shape != target.shape:
        return False
    return bool(np.array_equal(pred, target))


def two_of_three(preds: list[Grid], target: Grid) -> bool:
    """Task solved if ANY of the first N_ATTEMPTS predictions matches exactly.

    Named historically; actually checks the first 2 attempts per the 2026 rule.
    """
    return any(exact_match(p, target) for p in preds[:N_ATTEMPTS])


def any_attempt_matches(preds: list[Grid], target: Grid, n: int = N_ATTEMPTS) -> bool:
    """Explicit name for the 'any of N attempts matches' check."""
    return any(exact_match(p, target) for p in preds[:n])


@dataclass
class TaskResult:
    task_id: str
    solved: bool
    n_test: int
    solved_tests: int


@dataclass
class SubmissionScore:
    total: int
    solved: int
    per_task: list[TaskResult]

    @property
    def accuracy(self) -> float:
        return self.solved / self.total if self.total else 0.0


Prediction = dict[str, list[list[Grid]]]
"""Per task_id, per test index, list of up to 3 candidate grids."""


def score_submission(
    tasks: list[Task],
    predictions: Prediction,
) -> SubmissionScore:
    """A task is 'solved' only if ALL test inputs have a correct prediction among the top-3."""
    results: list[TaskResult] = []
    for t in tasks:
        solved_tests = 0
        for idx, test_pair in enumerate(t.test):
            if test_pair.output is None:
                continue  # hidden test, can't score
            cands = predictions.get(t.task_id, [[]] * len(t.test))[idx]
            if two_of_three(cands, test_pair.output):
                solved_tests += 1
        solved_all = solved_tests == len(t.test)
        results.append(
            TaskResult(
                task_id=t.task_id,
                solved=solved_all,
                n_test=len(t.test),
                solved_tests=solved_tests,
            )
        )
    return SubmissionScore(
        total=len(results),
        solved=sum(r.solved for r in results),
        per_task=results,
    )
