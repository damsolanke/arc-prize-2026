"""Common interface for ARC solvers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..eval.loader import Grid, Task


@dataclass(frozen=True)
class Prediction:
    task_id: str
    candidates: list[list[Grid]]
    """per test-input index, list of up to 3 candidate output grids"""


class BaseSolver(ABC):
    name: str = "base"

    @abstractmethod
    def solve(self, task: Task) -> Prediction:
        ...

    def solve_batch(self, tasks: list[Task]) -> list[Prediction]:
        return [self.solve(t) for t in tasks]
