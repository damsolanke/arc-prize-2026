"""Line-drawing solver: detect 2 endpoint pixels; draw line between them in output.

Many ARC tasks have the pattern "2 cells of color X in the input; draw a line
of color Y connecting them in the output." This heuristic handles the simplest
case: horizontal / vertical lines only (no bresenham diagonals in this version).
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _find_endpoints(g: Grid, color: int) -> list[tuple[int, int]]:
    rs, cs = np.where(g == color)
    return list(zip(rs.tolist(), cs.tolist()))


def _draw_hv_line(g: Grid, p1: tuple[int, int], p2: tuple[int, int], color: int) -> Grid:
    out = g.copy()
    r1, c1 = p1
    r2, c2 = p2
    if r1 == r2:
        for c in range(min(c1, c2), max(c1, c2) + 1):
            if out[r1, c] == 0:
                out[r1, c] = color
    elif c1 == c2:
        for r in range(min(r1, r2), max(r1, r2) + 1):
            if out[r, c1] == 0:
                out[r, c1] = color
    # else: diagonal — skip (not implemented)
    return out


class LineSolver(BaseSolver):
    """If all train pairs show exactly 2 endpoint-cells (in input) connected by a line
    (in output), learn endpoint-color + line-color and apply to test."""

    name = "line_solver"

    def _infer_rule(self, task: Task) -> tuple[int, int] | None:
        """Return (endpoint_color, line_color) if pattern is consistent, else None."""
        endpoint_color: int | None = None
        line_color: int | None = None
        for pair in task.train:
            if pair.output is None:
                continue
            if pair.input.shape != pair.output.shape:
                return None

            # Output should have input's cells + a line
            diff = (pair.input != pair.output)
            if not diff.any():
                return None  # no line drawn
            diff_colors = np.unique(pair.output[diff])
            if len(diff_colors) != 1:
                return None  # line must be single color
            lc = int(diff_colors[0])

            # Find all non-zero cells in input that appear unchanged in output
            nonzero_colors = [int(c) for c in np.unique(pair.input) if c != 0]
            # Endpoints: exactly 2 input cells of the *same* color, h/v aligned
            candidates = []
            for col in nonzero_colors:
                pts = _find_endpoints(pair.input, col)
                if len(pts) == 2:
                    p1, p2 = pts
                    if p1[0] == p2[0] or p1[1] == p2[1]:
                        candidates.append((col, p1, p2))
            if not candidates:
                return None

            # Pick the endpoint pair whose connecting line matches output
            match = None
            for col, p1, p2 in candidates:
                if np.array_equal(_draw_hv_line(pair.input, p1, p2, lc), pair.output):
                    match = col
                    break
            if match is None:
                return None
            if endpoint_color is None:
                endpoint_color, line_color = match, lc
            elif endpoint_color != match or line_color != lc:
                return None  # inconsistent across pairs
        if endpoint_color is None:
            return None
        return endpoint_color, line_color  # type: ignore[return-value]

    def solve(self, task: Task) -> Prediction:
        rule = self._infer_rule(task)
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                ep_color, ln_color = rule
                pts = _find_endpoints(test_pair.input, ep_color)
                if len(pts) == 2:
                    p1, p2 = pts
                    if p1[0] == p2[0] or p1[1] == p2[1]:
                        cands.append(_draw_hv_line(test_pair.input, p1, p2, ln_color))
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["LineSolver"]
