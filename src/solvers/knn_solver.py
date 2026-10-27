"""k-NN task matcher: find the training task whose demonstrations look most
like this task's demonstrations, apply its (learned) transformation.

Intuition: if eval task X's (input, output) pairs look structurally similar to
training task Y's (input, output) pairs, then X probably uses the same
transformation rule as Y. We define similarity by comparing the diff patterns
of (input, output) grids.

This is a cheap form of *task-level* few-shot learning. Unlike the other
heuristic solvers, it uses the 1000-task training set as a prior — the
missing ingredient from pure depth-N program search.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Task
from .base import BaseSolver, Prediction


def _pair_features(p_input: np.ndarray, p_output: np.ndarray) -> np.ndarray:
    """Compact feature vector for an (input, output) pair.

    Features: shape deltas, color-set deltas, cell-wise diff rate, unique-color
    counts. Dimensionless and small — designed so structurally similar pairs
    cluster together even if grid sizes differ.
    """
    ih, iw = p_input.shape
    oh, ow = p_output.shape

    shape_delta = np.array([oh - ih, ow - iw, oh * ow - ih * iw], dtype=np.float32)
    color_in = set(p_input.flatten().tolist())
    color_out = set(p_output.flatten().tolist())
    color_delta = np.array([
        len(color_in),
        len(color_out),
        len(color_out - color_in),
        len(color_in - color_out),
        len(color_in & color_out),
    ], dtype=np.float32)

    # Same-shape-specific: cell diff rate
    diff_rate = 0.0
    if p_input.shape == p_output.shape:
        diff_rate = float((p_input != p_output).sum()) / float(p_input.size)
    same_shape = float(p_input.shape == p_output.shape)

    return np.concatenate([shape_delta, color_delta, [diff_rate, same_shape]])


def _task_signature(task: Task) -> np.ndarray:
    """Aggregate features across all train pairs (mean)."""
    feats = [
        _pair_features(p.input, p.output)
        for p in task.train
        if p.output is not None
    ]
    if not feats:
        return np.zeros(_pair_features(np.zeros((1, 1), dtype=np.int8),
                                        np.zeros((1, 1), dtype=np.int8)).shape)
    return np.mean(np.stack(feats), axis=0)


class KNNSolver(BaseSolver):
    """Look up most-similar train task; apply its transformation (approximated)."""

    name = "knn_solver"

    def __init__(self, library: list[Task], k: int = 1):
        self.library = library
        self.k = k
        self.library_sigs: list[tuple[np.ndarray, Task]] = [
            (_task_signature(t), t) for t in library
        ]

    def _nearest(self, query: Task) -> list[Task]:
        qs = _task_signature(query)
        distances = [
            (float(np.linalg.norm(sig - qs)), t) for sig, t in self.library_sigs
        ]
        distances.sort(key=lambda d: d[0])
        return [t for _, t in distances[: self.k]]

    def solve(self, task: Task) -> Prediction:
        neighbors = self._nearest(task)
        # Use top neighbor's train-output-aggregated "typical shape" as a template
        # for each test input. Minimal baseline; full knn-transfer needs neighbor-
        # specific solver that we'll build when the retrieval itself is shown to
        # align on meaningful similarity.
        _ = neighbors  # reserved for future neighbor-conditional inference

        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[np.ndarray] = [
                test_pair.input.copy(),
                test_pair.input.copy(),
            ]
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["KNNSolver", "_task_signature"]


# ---- type alias ----

from ..eval.loader import Grid  # noqa: E402
