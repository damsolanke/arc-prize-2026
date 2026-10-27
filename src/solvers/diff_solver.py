"""DiffPatternSolver: learn per-task which input cells change to which output cells,
then apply the same fixed diff pattern at test time.

For tasks where the transformation is "add the same fixed pattern of non-background
cells to every input" (i.e., output = overlay fixed_pattern onto input), this
recovers the pattern by intersecting (input == output) cells across train pairs
and subtracting to find the overlay.

Orthogonal to D8/composition (which are structural transforms) and induction
(which is local neighborhood rules). Targets tasks like "draw a frame around
the input", "add a marker in a fixed position", "overlay a constant stencil".
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _fixed_overlay(pairs: list[tuple[Grid, Grid]]) -> Grid | None:
    """If all pairs share input-shape == output-shape and all (input, output) diffs
    match exactly the same cell-pattern, return the overlay grid (with 0 meaning
    'don't override'). Else None."""
    if not pairs:
        return None
    target_shape: tuple[int, int] | None = None
    overlay: np.ndarray | None = None
    for inp, out in pairs:
        if inp.shape != out.shape:
            return None
        if target_shape is None:
            target_shape = inp.shape
        elif inp.shape != target_shape:
            return None
        diff_mask = inp != out
        # The overlay is where output differs from input. Require output[diff] to be
        # consistent across pairs.
        current = np.zeros_like(inp)
        current[diff_mask] = out[diff_mask]
        if overlay is None:
            overlay = current
        else:
            # Both overlays should agree everywhere except at cells where input
            # covers the overlay with the same color (then diff=False). Simpler:
            # require equality of the overlay grids directly.
            if not np.array_equal(overlay, current):
                return None
    return overlay


def _apply_overlay(g: Grid, overlay: Grid) -> Grid | None:
    if g.shape != overlay.shape:
        return None
    out = g.copy()
    mask = overlay != 0
    out[mask] = overlay[mask]
    return out


class DiffSolver(BaseSolver):
    """Fixed-overlay pattern solver."""

    name = "diff_solver"

    def solve(self, task: Task) -> Prediction:
        pairs = [(p.input, p.output) for p in task.train if p.output is not None]
        overlay = _fixed_overlay(pairs)

        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if overlay is not None:
                applied = _apply_overlay(test_pair.input, overlay)
                if applied is not None:
                    cands.append(applied)
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["DiffSolver"]
