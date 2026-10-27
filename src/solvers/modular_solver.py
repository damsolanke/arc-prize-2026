"""ModularSolver: output[r,c] depends on (r,c) modulo some small k.

Examples: checkerboard, stripe, tile-patterns based on position.
Rules tested:
  - (r+c) % 2 == 0 → color_a else color_b
  - r % 2, c % 2 combinations
  - (r+c) % 3 variations
Different mechanism than transforms or structural rules.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _checker_2(g: Grid, color_a: int, color_b: int) -> Grid:
    h, w = g.shape
    out = np.zeros_like(g)
    for r in range(h):
        for c in range(w):
            out[r, c] = color_a if (r + c) % 2 == 0 else color_b
    return out


def _stripe_h_2(g: Grid, color_a: int, color_b: int) -> Grid:
    h, w = g.shape
    out = np.zeros_like(g)
    for r in range(h):
        out[r, :] = color_a if r % 2 == 0 else color_b
    return out


def _stripe_v_2(g: Grid, color_a: int, color_b: int) -> Grid:
    h, w = g.shape
    out = np.zeros_like(g)
    for c in range(w):
        out[:, c] = color_a if c % 2 == 0 else color_b
    return out


def _fits_rule_with_colors(fn: Callable, task: Task) -> tuple[int, int] | None:
    """Try all (color_a, color_b) pairs; return first pair that fits all train pairs."""
    for ca in range(10):
        for cb in range(10):
            if ca == cb:
                continue
            ok = True
            for p in task.train:
                if p.output is None:
                    continue
                try:
                    pred = fn(p.input, ca, cb)
                except Exception:
                    ok = False
                    break
                if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
                    ok = False
                    break
            if ok:
                return (ca, cb)
    return None


MODULAR_RULES = (
    ("checker", _checker_2),
    ("stripe_h", _stripe_h_2),
    ("stripe_v", _stripe_v_2),
)


class ModularSolver(BaseSolver):
    name = "modular_solver"

    def solve(self, task: Task) -> Prediction:
        found = None
        for rule_name, fn in MODULAR_RULES:
            colors = _fits_rule_with_colors(fn, task)
            if colors is not None:
                found = (fn, colors)
                break

        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if found is not None:
                fn, (ca, cb) = found
                try:
                    cands.append(fn(test_pair.input, ca, cb))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["ModularSolver"]
