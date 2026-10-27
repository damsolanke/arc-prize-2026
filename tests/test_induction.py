"""Tests for src/solvers/induction_solver.py."""
from __future__ import annotations

import numpy as np

from src.eval.loader import Pair, Task
from src.solvers.induction_solver import (
    InductionSolver,
    _induce_rule,
    _neighborhood,
)


def _g(a):
    return np.array(a, dtype=np.int8)


def test_neighborhood_center_only():
    g = _g([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    nb = _neighborhood(g, 1, 1, radius=1)
    assert nb == (1, 2, 3, 4, 5, 6, 7, 8, 9)


def test_neighborhood_pads_out_of_bounds():
    g = _g([[1, 2], [3, 4]])
    nb = _neighborhood(g, 0, 0, radius=1)
    # corner (0,0): pad outside → [0,0,0,0,1,2,0,3,4]
    assert nb == (0, 0, 0, 0, 1, 2, 0, 3, 4)


def test_induce_rule_consistent():
    # simple: swap 1<->2
    pairs = [
        (_g([[1, 2], [2, 1]]), _g([[2, 1], [1, 2]])),
    ]
    rule = _induce_rule(pairs, radius=1)
    assert rule is not None
    # every neighborhood where center=1 should map to 2 in output
    assert len(rule) > 0


def test_induce_rule_inconsistent_returns_none():
    # conflicting rules: same neighborhood maps to different values
    pairs = [
        (_g([[1, 1], [1, 1]]), _g([[2, 2], [2, 2]])),
        (_g([[1, 1], [1, 1]]), _g([[3, 3], [3, 3]])),
    ]
    rule = _induce_rule(pairs, radius=1)
    assert rule is None


def test_induce_rule_shape_mismatch_returns_none():
    pairs = [
        (_g([[1]]), _g([[1, 2]])),
    ]
    assert _induce_rule(pairs) is None


def test_solver_solves_simple_swap():
    # train: swap 1<->2 consistently
    pairs = (
        Pair(_g([[1, 0], [0, 1]]), _g([[2, 0], [0, 2]])),
        Pair(_g([[1, 0, 1]]), _g([[2, 0, 2]])),
    )
    test = (Pair(_g([[1, 0], [0, 1]]), _g([[2, 0], [0, 2]])),)
    task = Task(task_id="swap_1_2", train=pairs, test=test)
    solver = InductionSolver(radius=1)
    pred = solver.solve(task)
    assert pred.candidates[0][0].tolist() == [[2, 0], [0, 2]]


def test_solver_abstains_on_non_rule_task():
    # Task where the rule can't be induced (output depends on global info)
    pairs = (
        Pair(_g([[1, 1], [1, 1]]), _g([[1, 0], [0, 1]])),
    )
    test = (Pair(_g([[1, 1], [1, 1]]), _g([[0, 0], [0, 0]])),)
    task = Task(task_id="not_a_rule", train=pairs, test=test)
    solver = InductionSolver(radius=1)
    pred = solver.solve(task)
    # Won't match target; just shouldn't crash
    assert len(pred.candidates[0]) >= 2
