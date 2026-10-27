"""FrameDetectionSolver: many ARC tasks feature a rectangular frame of constant color
surrounding an 'interior'. The transformation usually modifies only the interior
or relates frame-color to interior.

This solver detects a monochrome rectangular frame (top row, bottom row, left col,
right col all same color) and, if consistent across train pairs, applies the
frame's rule to the test.

Heuristic 1: interior swap (swap interior with a color inferred from train).
Heuristic 2: fill interior with frame color.
Heuristic 3: strip frame (output = interior only).
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _detect_frame(g: Grid) -> int | None:
    """Return the frame color if g's outermost rows/cols are all one color, else None."""
    h, w = g.shape
    if h < 2 or w < 2:
        return None
    top = g[0, :]
    bottom = g[h - 1, :]
    left = g[:, 0]
    right = g[:, w - 1]
    all_edges = np.concatenate([top, bottom, left, right])
    colors = np.unique(all_edges)
    if len(colors) != 1:
        return None
    return int(colors[0])


def _strip_frame(g: Grid) -> Grid:
    h, w = g.shape
    if h < 3 or w < 3:
        return g.copy()
    return g[1:h - 1, 1:w - 1].copy()


def _fill_interior(g: Grid, color: int) -> Grid:
    out = g.copy()
    h, w = g.shape
    if h > 2 and w > 2:
        out[1:h - 1, 1:w - 1] = color
    return out


class FrameSolver(BaseSolver):
    """Detect frame; try 3 frame-based transforms; pick ones fitting train."""

    name = "frame_solver"

    def _infer_rule(self, task: Task) -> str | None:
        """Try each of {strip, fill_frame_color, fill_most_common_interior};
        return name of rule consistent across all train pairs, else None."""
        rules = {
            "strip": lambda g: _strip_frame(g),
            "fill_frame": lambda g: _fill_interior(g, _detect_frame(g) or 0),
        }
        for name, fn in rules.items():
            fits = True
            for pair in task.train:
                if pair.output is None:
                    continue
                if _detect_frame(pair.input) is None:
                    fits = False
                    break
                try:
                    pred = fn(pair.input)
                except Exception:
                    fits = False
                    break
                if pred.shape != pair.output.shape or not np.array_equal(pred, pair.output):
                    fits = False
                    break
            if fits:
                return name
        return None

    def solve(self, task: Task) -> Prediction:
        rule = self._infer_rule(task)
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule == "strip":
                cands.append(_strip_frame(test_pair.input))
            elif rule == "fill_frame":
                fc = _detect_frame(test_pair.input) or 0
                cands.append(_fill_interior(test_pair.input, fc))
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["FrameSolver"]
