"""CellularAutomatonSolver: try simple CA rules.

Each rule is a function (cell, neighborhood_count_by_color) → new_cell.
We enumerate a small family of rules and check which fits train pairs.

Orthogonal to induction (which memorizes neighborhoods) because CA rules
generalize via counting, not exact pattern matching.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _count_neighbors(g: Grid, r: int, c: int, color: int, diagonal: bool = True) -> int:
    h, w = g.shape
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if not diagonal and abs(dr) + abs(dc) > 1:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and g[rr, cc] == color:
                count += 1
    return count


def _apply_ca_rule(g: Grid, rule: Callable[[int, dict], int]) -> Grid:
    h, w = g.shape
    out = np.zeros_like(g)
    for r in range(h):
        for c in range(w):
            counts = {}
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        col = int(g[rr, cc])
                        counts[col] = counts.get(col, 0) + 1
            out[r, c] = rule(int(g[r, c]), counts)
    return out


def _rule_any_neighbor_becomes(g_color: int, counts: dict) -> int:
    """Cell becomes most common neighbor color."""
    if not counts:
        return g_color
    return max(counts, key=counts.get)


def _rule_zero_if_isolated(g_color: int, counts: dict) -> int:
    """Cell becomes 0 if no same-color neighbor, else unchanged."""
    if g_color == 0:
        return 0
    return g_color if counts.get(g_color, 0) > 0 else 0


def _rule_grow_if_neighbor(g_color: int, counts: dict) -> int:
    """If empty (0) but has colored neighbor, become that color."""
    if g_color != 0:
        return g_color
    non_zero = {c: n for c, n in counts.items() if c != 0}
    if not non_zero:
        return 0
    return max(non_zero, key=non_zero.get)


CA_RULES: tuple[tuple[str, Callable], ...] = (
    ("any_neighbor", _rule_any_neighbor_becomes),
    ("zero_if_isolated", _rule_zero_if_isolated),
    ("grow_if_neighbor", _rule_grow_if_neighbor),
)


def _fits_train(rule, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _apply_ca_rule(p.input, rule)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class CellularAutomatonSolver(BaseSolver):
    name = "ca_solver"

    def solve(self, task: Task) -> Prediction:
        winners = [rule for _, rule in CA_RULES if _fits_train(rule, task)]
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            for rule in winners[:2]:
                try:
                    cands.append(_apply_ca_rule(test_pair.input, rule))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["CellularAutomatonSolver"]
