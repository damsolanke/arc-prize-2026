"""Rule induction from train-pair diffs.

For same-shape same-palette tasks, we can observe input/output cell-by-cell and
induce a *local* rule: for each cell in input that changed, what caused the change?

Approach: for each pair of (input_cell, output_cell) where they differ, record
the 3x3 neighborhood of input as the rule key and the output cell as the rule value.
If the induced rule is consistent across all training pairs, apply to test.

This captures tasks like "replace every X surrounded by Y with Z" — a common
ARC pattern that is out of reach for D8 + composition.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction

PAD = 0  # treat out-of-bounds as color 0


def _neighborhood(g: Grid, r: int, c: int, radius: int = 1) -> tuple:
    """Return (2R+1)x(2R+1) neighborhood around (r, c), padded with PAD outside."""
    h, w = g.shape
    n = 2 * radius + 1
    patch = np.full((n, n), PAD, dtype=np.int8)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                patch[dr + radius, dc + radius] = g[rr, cc]
    return tuple(patch.flatten().tolist())


def _induce_rule(pairs: list[tuple[Grid, Grid]], radius: int = 1) -> dict | None:
    """Return neighborhood → output-cell mapping if consistent, else None.

    Rule key = (2R+1)^2 tuple of input neighborhood cells.
    Rule value = output cell color.
    """
    if not pairs:
        return None
    rule: dict[tuple, int] = {}
    for inp, out in pairs:
        if inp.shape != out.shape:
            return None
        h, w = inp.shape
        for r in range(h):
            for c in range(w):
                key = _neighborhood(inp, r, c, radius)
                val = int(out[r, c])
                if key in rule and rule[key] != val:
                    return None  # inconsistent
                rule[key] = val
    return rule


def _apply_rule(rule: dict, g: Grid, radius: int = 1) -> Grid | None:
    """Apply rule to g, cell by cell. Return None if any key is unseen."""
    out = np.zeros_like(g)
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            key = _neighborhood(g, r, c, radius)
            if key not in rule:
                return None
            out[r, c] = rule[key]
    return out


class InductionSolver(BaseSolver):
    """Induce a neighborhood→cell rule from train pairs; apply to test."""

    name = "induction_solver"

    def __init__(self, radius: int = 1):
        self.radius = radius

    def solve(self, task: Task) -> Prediction:
        pairs = [(p.input, p.output) for p in task.train if p.output is not None]
        rule = _induce_rule(pairs, radius=self.radius)

        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            test_cands: list[Grid] = []
            if rule is not None:
                applied = _apply_rule(rule, test_pair.input, radius=self.radius)
                if applied is not None:
                    test_cands.append(applied)

            # Fallbacks
            while len(test_cands) < 2:
                test_cands.append(test_pair.input.copy())
            candidates.append(test_cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["InductionSolver"]
