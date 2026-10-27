"""XORCombineSolver: input has 2 regions side-by-side; output = XOR/AND/OR of them.

For tasks where the input contains two equally-sized regions separated by a
divider, and the output is the boolean combination of the non-zero masks.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _split_h(g: Grid) -> tuple[Grid, Grid] | None:
    h, w = g.shape
    if w % 2 != 0:
        return None
    mid = w // 2
    return g[:, :mid], g[:, mid:]


def _split_v(g: Grid) -> tuple[Grid, Grid] | None:
    h, w = g.shape
    if h % 2 != 0:
        return None
    mid = h // 2
    return g[:mid, :], g[mid:, :]


def _xor(a, b, output_color):
    mask_a = a != 0
    mask_b = b != 0
    out = np.zeros_like(a)
    out[mask_a ^ mask_b] = output_color
    return out


def _or(a, b, output_color):
    mask = (a != 0) | (b != 0)
    out = np.zeros_like(a)
    out[mask] = output_color
    return out


def _and(a, b, output_color):
    mask = (a != 0) & (b != 0)
    out = np.zeros_like(a)
    out[mask] = output_color
    return out


OPS = (("xor", _xor), ("or", _or), ("and", _and))
SPLITS = (("h", _split_h), ("v", _split_v))


def _try_rule(split_fn, op_fn, output_color, task):
    for p in task.train:
        if p.output is None:
            continue
        splits = split_fn(p.input)
        if splits is None:
            return False
        a, b = splits
        if a.shape != b.shape:
            return False
        try:
            pred = op_fn(a, b, output_color)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class XORCombineSolver(BaseSolver):
    name = "xor_combine_solver"

    def solve(self, task: Task) -> Prediction:
        rule = None
        for sname, sfn in SPLITS:
            for oname, ofn in OPS:
                for color in range(1, 10):
                    if _try_rule(sfn, ofn, color, task):
                        rule = (sfn, ofn, color)
                        break
                if rule:
                    break
            if rule:
                break

        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                sfn, ofn, color = rule
                splits = sfn(test_pair.input)
                if splits is not None:
                    a, b = splits
                    if a.shape == b.shape:
                        try:
                            cands.append(ofn(a, b, color))
                        except Exception:
                            pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["XORCombineSolver"]
