"""MultiRadiusInduction: try induction at multiple neighborhood radii; take the
smallest consistent one (Occam's razor)."""
from __future__ import annotations

from ..eval.loader import Task
from .base import BaseSolver, Prediction
from .induction_solver import _apply_rule, _induce_rule


class MultiRadiusInductionSolver(BaseSolver):
    name = "multi_induction_solver"

    def __init__(self, radii: tuple[int, ...] = (0, 1, 2)):
        self.radii = radii

    def solve(self, task: Task) -> Prediction:
        pairs = [(p.input, p.output) for p in task.train if p.output is not None]

        rule_by_radius: dict[int, dict] = {}
        for r in self.radii:
            rule = _induce_rule(pairs, radius=r)
            if rule is not None:
                rule_by_radius[r] = rule

        candidates: list[list] = []
        for test_pair in task.test:
            cands = []
            # Occam: smallest consistent radius first
            for r in sorted(rule_by_radius):
                applied = _apply_rule(rule_by_radius[r], test_pair.input, radius=r)
                if applied is not None:
                    cands.append(applied)
                if len(cands) >= 2:
                    break
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["MultiRadiusInductionSolver"]
