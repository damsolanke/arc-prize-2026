"""BlockReplicationSolver: each input cell becomes a scaled block in output.

For tasks like: each 1-pixel in input → 3x3 block of color X in output.
Different from scale_up (which preserves colors) — this MAPS colors to blocks.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction


def _block_replicate(g: Grid, block_factor: int) -> Grid:
    """Each cell becomes a block_factor x block_factor block of same color."""
    h, w = g.shape
    out = np.zeros((h * block_factor, w * block_factor), dtype=np.int8)
    for r in range(h):
        for c in range(w):
            out[r * block_factor:(r + 1) * block_factor,
                c * block_factor:(c + 1) * block_factor] = g[r, c]
    return out


def _fits(factor: int, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = _block_replicate(p.input, factor)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


class BlockReplicateSolver(BaseSolver):
    name = "block_replicate_solver"

    def solve(self, task: Task) -> Prediction:
        factor = None
        for f in range(2, 7):
            if _fits(f, task):
                factor = f
                break
        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            cands: list[Grid] = []
            if factor is not None:
                try:
                    cands.append(_block_replicate(test_pair.input, factor))
                except Exception:
                    pass
            while len(cands) < 2:
                cands.append(test_pair.input.copy())
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["BlockReplicateSolver"]
