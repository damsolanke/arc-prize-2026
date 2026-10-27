"""MovementSolver: detect per-color object movement (dr, dc) and apply.

For tasks where each color moves by a fixed offset (possibly different per
color). Learn offset-per-color from train pairs; apply at test.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _find_color_movements(inp: Grid, out: Grid) -> dict[int, tuple[int, int]] | None:
    """Return dict[color] -> (dr, dc) if each color moved by a consistent offset."""
    if inp.shape != out.shape:
        return None
    mapping: dict[int, tuple[int, int]] = {}
    for color in np.unique(inp):
        if color == 0:
            continue
        ip = list(zip(*np.where(inp == color)))
        op = list(zip(*np.where(out == color)))
        if not ip or len(ip) != len(op):
            return None
        # Try (dr, dc) computed from first input to first output sorted
        ip_sorted = sorted(ip)
        op_sorted = sorted(op)
        dr = op_sorted[0][0] - ip_sorted[0][0]
        dc = op_sorted[0][1] - ip_sorted[0][1]
        # Verify all points moved by the same (dr, dc)
        for (ir, ic), (orr, oc) in zip(ip_sorted, op_sorted):
            if orr - ir != dr or oc - ic != dc:
                return None
        mapping[int(color)] = (int(dr), int(dc))
    return mapping


def _apply_movement(g: Grid, mapping: dict[int, tuple[int, int]]) -> Grid:
    out = np.zeros_like(g)
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            col = int(g[r, c])
            if col == 0:
                continue
            dr, dc = mapping.get(col, (0, 0))
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = col
    return out


def _find_consistent_mapping(task: Task) -> dict[int, tuple[int, int]] | None:
    mapping = None
    for p in task.train:
        if p.output is None:
            continue
        m = _find_color_movements(p.input, p.output)
        if m is None:
            return None
        if mapping is None:
            mapping = m
        else:
            # merge — require consistency
            for color, off in m.items():
                if color in mapping and mapping[color] != off:
                    return None
                mapping[color] = off
    return mapping


class MovementSolver(BaseSolver):
    name = "movement_solver"

    def solve(self, task: Task) -> Prediction:
        mapping = _find_consistent_mapping(task)
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if mapping is not None:
                try:
                    cands.append(_apply_movement(test_pair.input, mapping))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["MovementSolver"]
