"""Tests for src/solvers/diff_solver.py and multi_induction_solver.py."""
from __future__ import annotations

import numpy as np

from src.eval.loader import Pair, Task
from src.solvers.diff_solver import DiffSolver, _apply_overlay, _fixed_overlay
from src.solvers.multi_induction_solver import MultiRadiusInductionSolver


def _g(a):
    return np.array(a, dtype=np.int8)


def test_fixed_overlay_consistent():
    # Two pairs that differ from input by the SAME overlay pattern
    pairs = [
        (_g([[0, 0, 0], [0, 1, 0]]), _g([[9, 0, 9], [0, 1, 0]])),
        (_g([[0, 0, 0], [0, 5, 0]]), _g([[9, 0, 9], [0, 5, 0]])),
    ]
    overlay = _fixed_overlay(pairs)
    assert overlay is not None
    assert overlay[0, 0] == 9
    assert overlay[0, 2] == 9
    assert overlay[0, 1] == 0  # unchanged


def test_fixed_overlay_inconsistent_returns_none():
    pairs = [
        (_g([[0, 0]]), _g([[1, 0]])),
        (_g([[0, 0]]), _g([[2, 0]])),  # different overlay
    ]
    assert _fixed_overlay(pairs) is None


def test_apply_overlay():
    overlay = _g([[9, 0, 9]])
    g = _g([[0, 5, 0]])
    out = _apply_overlay(g, overlay)
    assert out is not None
    assert out.tolist() == [[9, 5, 9]]


def test_apply_overlay_shape_mismatch():
    overlay = _g([[0, 0]])
    g = _g([[0]])
    assert _apply_overlay(g, overlay) is None


def test_diff_solver_solves_frame_task():
    # Task: add corners of color 9 everywhere (fixed overlay)
    pairs = (
        Pair(_g([[0, 0, 0], [0, 1, 0]]), _g([[9, 0, 9], [0, 1, 0]])),
        Pair(_g([[0, 0, 0], [0, 5, 0]]), _g([[9, 0, 9], [0, 5, 0]])),
    )
    test = (Pair(_g([[0, 0, 0], [0, 7, 0]]), _g([[9, 0, 9], [0, 7, 0]])),)
    task = Task(task_id="frame", train=pairs, test=test)
    pred = DiffSolver().solve(task)
    assert pred.candidates[0][0].tolist() == [[9, 0, 9], [0, 7, 0]]


def test_multi_radius_induction_picks_smallest_consistent():
    # Trivial: color-swap 1->2, solvable at r=1 (or r=0 once we include identity pass)
    pairs = (
        Pair(_g([[1, 0], [0, 1]]), _g([[2, 0], [0, 2]])),
    )
    test = (Pair(_g([[1, 0], [0, 1]]), _g([[2, 0], [0, 2]])),)
    task = Task(task_id="swap", train=pairs, test=test)
    solver = MultiRadiusInductionSolver(radii=(0, 1, 2))
    pred = solver.solve(task)
    # Expect at least one of the candidates to match
    target = [[2, 0], [0, 2]]
    matches = [c.tolist() == target for c in pred.candidates[0]]
    assert any(matches)
