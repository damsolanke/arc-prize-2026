"""Augmentation utilities for test-time training.

The D8 group + color permutation is the minimum augmentation set shared by
NVARC, ARChitects, MindsAI, and TRM. Round-trip invertibility is required
because inference must be done in augmented space then inverted back.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..eval.loader import Grid

DIGIT_TOKENS = "0123456789"


# ----- D8 -----

def d8_apply(g: Grid, tid: int) -> Grid:
    if tid == 0:
        return g
    if tid == 1:
        return np.rot90(g, 1)
    if tid == 2:
        return np.rot90(g, 2)
    if tid == 3:
        return np.rot90(g, 3)
    if tid == 4:
        return g[:, ::-1]
    if tid == 5:
        return g[::-1, :]
    if tid == 6:
        return g.T
    if tid == 7:
        return g[::-1, ::-1].T
    raise ValueError(f"bad tid {tid}")


def d8_invert(g: Grid, tid: int) -> Grid:
    """Inverse of d8_apply(_, tid)."""
    inverse = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}
    return d8_apply(g, inverse[tid])


# ----- Color permutation -----

def color_permute(g: Grid, perm: np.ndarray) -> Grid:
    """Relabel colors per perm (length-10 permutation of 0..9)."""
    out = g.copy()
    for src, dst in enumerate(perm):
        out[g == src] = dst
    return out


def color_permute_invert(g: Grid, perm: np.ndarray) -> Grid:
    inverse = np.argsort(perm)
    return color_permute(g, inverse)


@dataclass(frozen=True)
class Aug:
    """One D8 symmetry + one color permutation."""

    tid: int
    perm: tuple[int, ...]

    def apply(self, g: Grid) -> Grid:
        perm_arr = np.array(self.perm, dtype=np.int8)
        return color_permute(d8_apply(g, self.tid), perm_arr)

    def invert(self, g: Grid) -> Grid:
        perm_arr = np.array(self.perm, dtype=np.int8)
        return d8_invert(color_permute_invert(g, perm_arr), self.tid)


def sample_augs(n: int, rng: np.random.Generator) -> list[Aug]:
    augs: list[Aug] = []
    for _ in range(n):
        tid = int(rng.integers(0, 8))
        perm = np.arange(10)
        rng.shuffle(perm)
        augs.append(Aug(tid=tid, perm=tuple(perm.tolist())))
    return augs


# ----- Grid <-> text representation (for LLM-based solvers) -----

def grid_to_text(g: Grid) -> str:
    return "\n".join("".join(str(c) for c in row) for row in g)


def text_to_grid(txt: str) -> Grid | None:
    rows = [r for r in txt.strip().split("\n") if r]
    if not rows:
        return None
    w = len(rows[0])
    out = []
    for r in rows:
        if len(r) != w or not all(c in DIGIT_TOKENS for c in r):
            return None
        out.append([int(c) for c in r])
    return np.array(out, dtype=np.int8)


__all__ = [
    "Aug",
    "d8_apply",
    "d8_invert",
    "color_permute",
    "color_permute_invert",
    "sample_augs",
    "grid_to_text",
    "text_to_grid",
]
