"""DimSolver: output dimensions depend on input count / size statistics.

Some ARC tasks have output shape = f(input properties), e.g.:
- output is NxN where N = count of non-zero cells
- output is HxW where H,W = counts of distinct colors

We infer the (shape_fn, fill_fn) from train pairs and apply to test.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _n_nonzero(g: Grid) -> int:
    return int(np.count_nonzero(g))


def _n_distinct_nonzero(g: Grid) -> int:
    vals = set(g.flatten().tolist()) - {0}
    return len(vals)


def _most_common_color(g: Grid) -> int:
    vals, counts = np.unique(g[g != 0], return_counts=True) if (g != 0).any() else (np.array([0]), np.array([1]))
    return int(vals[counts.argmax()])


# Candidate output shapes parametrized by input
SHAPE_FNS = {
    "count_x_count": lambda g: (_n_nonzero(g), _n_nonzero(g)),
    "distinct_x_distinct": lambda g: (_n_distinct_nonzero(g), _n_distinct_nonzero(g)),
    "count_x_1": lambda g: (_n_nonzero(g), 1),
    "1x1": lambda g: (1, 1),
}

# Candidate fill functions
FILL_FNS = {
    "most_common": _most_common_color,
    "distinct_count": lambda g: _n_distinct_nonzero(g),
    "nonzero_count": lambda g: min(_n_nonzero(g), 9),
}


def _try_rule(shape_fn, fill_fn, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            h, w = shape_fn(p.input)
            fill = fill_fn(p.input)
        except Exception:
            return False
        if (h, w) != p.output.shape:
            return False
        expected = np.full((h, w), fill, dtype=np.int8)
        if not np.array_equal(expected, p.output):
            return False
    return True


class DimSolver(BaseSolver):
    name = "dim_solver"

    def solve(self, task: Task) -> Prediction:
        rule = None
        for s_name, s_fn in SHAPE_FNS.items():
            for f_name, f_fn in FILL_FNS.items():
                if _try_rule(s_fn, f_fn, task):
                    rule = (s_fn, f_fn)
                    break
            if rule:
                break

        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if rule is not None:
                s_fn, f_fn = rule
                try:
                    h, w = s_fn(test_pair.input)
                    fill = f_fn(test_pair.input)
                    cands.append(np.full((h, w), fill, dtype=np.int8))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["DimSolver"]
