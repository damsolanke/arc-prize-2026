"""Baseline: try every heuristic in BASIC_TRANSFORMS + inferred color maps.

For each candidate, check if it fits all train pairs (exact-match). If any match,
apply to test inputs; fall back to diverse non-fitting candidates if none fit.
Returns up to 3 candidate outputs per test input.

This is the floor. Expected ARC-AGI-2 eval: ~3-10% depending on heuristic coverage.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction
from .heuristics import (
    BASIC_TRANSFORMS,
    Transform,
    apply_color_map,
    identity,
    infer_consistent_color_map,
    flip_h,
    flip_v,
)
from .object_heuristics import OBJECT_HEURISTICS


def _fits_train(fn: Transform, task: Task) -> bool:
    """True if fn maps every train input to train output exactly."""
    for pair in task.train:
        if pair.output is None:
            continue
        try:
            pred = fn(pair.input)
        except Exception:
            return False
        if pred.shape != pair.output.shape or not np.array_equal(pred, pair.output):
            return False
    return True


def _partial_fit_score(fn: Transform, task: Task) -> float:
    """Fraction of cells correct across all train outputs."""
    total = 0
    correct = 0
    for pair in task.train:
        if pair.output is None:
            continue
        try:
            pred = fn(pair.input)
        except Exception:
            return 0.0
        if pred.shape != pair.output.shape:
            continue
        total += pair.output.size
        correct += int(np.array_equal(pred, pair.output)) * pair.output.size
        # Allow partial credit across cells too
        if pred.shape == pair.output.shape:
            correct_cells = int((pred == pair.output).sum())
            correct = max(correct, correct_cells)
    return correct / total if total else 0.0


class TransformSolver(BaseSolver):
    """Heuristic library + per-task color map inference."""

    name = "transform_solver"

    def _candidate_transforms(self, task: Task) -> list[tuple[str, Transform]]:
        cands: list[tuple[str, Transform]] = list(BASIC_TRANSFORMS) + list(OBJECT_HEURISTICS)

        # Inferred color map (if consistent across train pairs)
        pairs = tuple(
            (p.input, p.output) for p in task.train if p.output is not None
        )
        if pairs:
            m = infer_consistent_color_map(pairs)
            if m:
                cands.append(
                    (f"color_map_{len(m)}", lambda g, mm=m: apply_color_map(g, mm))
                )
        return cands

    def solve(self, task: Task) -> Prediction:
        cands = self._candidate_transforms(task)
        winners: list[Transform] = [fn for _, fn in cands if _fits_train(fn, task)]

        # If no exact fit, pick top 3 by partial-fit score for at-least-diverse guesses
        if not winners:
            scored = sorted(cands, key=lambda kv: _partial_fit_score(kv[1], task), reverse=True)
            winners = [fn for _, fn in scored[:3]]

        # Guarantee 3 diverse candidates
        if len(winners) < 3:
            for alt in (identity, flip_h, flip_v):
                if alt not in winners:
                    winners.append(alt)
                if len(winners) == 3:
                    break

        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            test_cands = []
            for fn in winners[:3]:
                try:
                    test_cands.append(fn(test_pair.input))
                except Exception:
                    test_cands.append(test_pair.input.copy())
            while len(test_cands) < 3:
                test_cands.append(test_pair.input.copy())
            candidates.append(test_cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["TransformSolver"]
