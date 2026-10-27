"""A library of pure grid-transformation heuristics.

Each function takes a single Grid and returns a single Grid. Solvers compose these
by (a) trying each on train pairs, (b) picking ones that fit, (c) applying to test.

These are the *atoms* of the program-space DSL. 2025 ARC-AGI-2 SOTA methods
(NVARC, ARChitects) use neural nets to propose sequences over a similar op set;
we use them directly as a diverse-proposal baseline.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

import numpy as np

from ..eval.loader import Grid

Transform = Callable[[Grid], Grid]

# ----- D8 symmetry group (8 elements) -----

def identity(g: Grid) -> Grid:
    return g


def flip_h(g: Grid) -> Grid:
    return g[:, ::-1]


def flip_v(g: Grid) -> Grid:
    return g[::-1, :]


def rot90(g: Grid) -> Grid:
    return np.rot90(g, 1)


def rot180(g: Grid) -> Grid:
    return np.rot90(g, 2)


def rot270(g: Grid) -> Grid:
    return np.rot90(g, 3)


def transpose(g: Grid) -> Grid:
    return g.T


def anti_transpose(g: Grid) -> Grid:
    return g[::-1, ::-1].T


D8: tuple[Transform, ...] = (
    identity, flip_h, flip_v, rot90, rot180, rot270, transpose, anti_transpose,
)

# ----- Color operations -----

def most_common_color(g: Grid) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[counts.argmax()])


def least_common_color(g: Grid) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[counts.argmin()])


def unique_colors(g: Grid) -> list[int]:
    return sorted(set(g.flatten().tolist()))


def solid_fill_most_common(g: Grid) -> Grid:
    return np.full_like(g, most_common_color(g))


def solid_fill_least_common(g: Grid) -> Grid:
    return np.full_like(g, least_common_color(g))


def solid_fill_zero(g: Grid) -> Grid:
    return np.zeros_like(g)


def invert_colors(g: Grid) -> Grid:
    """Swap most common <-> least common color."""
    out = g.copy()
    mc = most_common_color(g)
    lc = least_common_color(g)
    if mc == lc:
        return out
    out[g == mc] = lc
    out[g == lc] = mc
    return out


# ----- Cropping and bounding box -----

def nonzero_bbox(g: Grid) -> tuple[int, int, int, int]:
    """Return (r0, c0, r1, c1) bounding box of non-zero cells; (-1,-1,-1,-1) if empty."""
    mask = g != 0
    if not mask.any():
        return (-1, -1, -1, -1)
    rs, cs = np.where(mask)
    return int(rs.min()), int(cs.min()), int(rs.max()) + 1, int(cs.max()) + 1


def crop_to_nonzero_bbox(g: Grid) -> Grid:
    r0, c0, r1, c1 = nonzero_bbox(g)
    if r0 < 0:
        return g.copy()
    return g[r0:r1, c0:c1].copy()


def color_bbox(g: Grid, color: int) -> tuple[int, int, int, int]:
    mask = g == color
    if not mask.any():
        return (-1, -1, -1, -1)
    rs, cs = np.where(mask)
    return int(rs.min()), int(cs.min()), int(rs.max()) + 1, int(cs.max()) + 1


def crop_to_color_bbox(color: int) -> Transform:
    def _crop(g: Grid) -> Grid:
        r0, c0, r1, c1 = color_bbox(g, color)
        if r0 < 0:
            return g.copy()
        return g[r0:r1, c0:c1].copy()
    return _crop


# ----- Tiling and scaling -----

def tile_3x3_self(g: Grid) -> Grid:
    return np.tile(g, (3, 3))


def tile_2x2_self(g: Grid) -> Grid:
    return np.tile(g, (2, 2))


def scale_up_2x(g: Grid) -> Grid:
    return np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)


def scale_up_3x(g: Grid) -> Grid:
    return np.repeat(np.repeat(g, 3, axis=0), 3, axis=1)


def scale_down_2x(g: Grid) -> Grid:
    """Downsample by 2 — take top-left of each 2x2 block. Loses info."""
    return g[::2, ::2].copy()


# ----- Symmetry completion -----

def complete_horizontal_symmetry(g: Grid) -> Grid:
    """Make the grid horizontally symmetric by OR-ing with its mirror."""
    h, w = g.shape
    out = g.copy()
    for i in range(h):
        for j in range(w):
            if out[i, j] == 0 and out[i, w - 1 - j] != 0:
                out[i, j] = out[i, w - 1 - j]
    return out


def complete_vertical_symmetry(g: Grid) -> Grid:
    h, w = g.shape
    out = g.copy()
    for i in range(h):
        for j in range(w):
            if out[i, j] == 0 and out[h - 1 - i, j] != 0:
                out[i, j] = out[h - 1 - i, j]
    return out


# ----- Color mapping from pair examples -----

def color_map_from_pair(src: Grid, dst: Grid) -> dict[int, int] | None:
    if src.shape != dst.shape:
        return None
    mapping: dict[int, int] = {}
    for s, d in zip(src.flatten().tolist(), dst.flatten().tolist()):
        if s in mapping and mapping[s] != d:
            return None
        mapping[s] = d
    return mapping


def apply_color_map(g: Grid, mapping: dict[int, int]) -> Grid:
    out = g.copy()
    for src, dst in mapping.items():
        out[g == src] = dst
    return out


def infer_consistent_color_map(pairs: Sequence[tuple[Grid, Grid]]) -> dict[int, int] | None:
    """Infer a color mapping consistent across ALL pairs. None if inconsistent."""
    combined: dict[int, int] = {}
    for src, dst in pairs:
        if src.shape != dst.shape:
            return None
        m = color_map_from_pair(src, dst)
        if m is None:
            return None
        for k, v in m.items():
            if k in combined and combined[k] != v:
                return None
            combined[k] = v
    return combined


# ----- Object/connected-component ops -----

def connected_components(g: Grid, *, ignore: int = 0) -> list[np.ndarray]:
    """Return list of boolean masks, one per 4-connected component of non-ignore cells."""
    from scipy.ndimage import label  # type: ignore[import-untyped]

    mask = g != ignore
    labeled, n = label(mask)
    return [labeled == i for i in range(1, n + 1)]


def largest_component_isolate(g: Grid, *, background: int = 0) -> Grid:
    """Return a copy of g with only the largest non-background component kept."""
    comps = connected_components(g, ignore=background)
    if not comps:
        return g.copy()
    largest = max(comps, key=int.__call__ if False else (lambda m: int(m.sum())))
    out = np.full_like(g, background)
    out[largest] = g[largest]
    return out


def smallest_component_isolate(g: Grid, *, background: int = 0) -> Grid:
    comps = connected_components(g, ignore=background)
    if not comps:
        return g.copy()
    smallest = min(comps, key=lambda m: int(m.sum()))
    out = np.full_like(g, background)
    out[smallest] = g[smallest]
    return out


# ----- Composed library -----

BASIC_TRANSFORMS: tuple[tuple[str, Transform], ...] = (
    ("identity", identity),
    ("flip_h", flip_h),
    ("flip_v", flip_v),
    ("rot90", rot90),
    ("rot180", rot180),
    ("rot270", rot270),
    ("transpose", transpose),
    ("anti_transpose", anti_transpose),
    ("solid_fill_most_common", solid_fill_most_common),
    ("solid_fill_least_common", solid_fill_least_common),
    ("solid_fill_zero", solid_fill_zero),
    ("invert_colors", invert_colors),
    ("crop_to_nonzero_bbox", crop_to_nonzero_bbox),
    ("tile_2x2_self", tile_2x2_self),
    ("tile_3x3_self", tile_3x3_self),
    ("scale_up_2x", scale_up_2x),
    ("scale_up_3x", scale_up_3x),
    ("scale_down_2x", scale_down_2x),
    ("complete_horizontal_symmetry", complete_horizontal_symmetry),
    ("complete_vertical_symmetry", complete_vertical_symmetry),
    ("largest_component_isolate", largest_component_isolate),
    ("smallest_component_isolate", smallest_component_isolate),
)


__all__ = [
    "Transform", "D8", "BASIC_TRANSFORMS",
    "identity", "flip_h", "flip_v", "rot90", "rot180", "rot270", "transpose", "anti_transpose",
    "most_common_color", "least_common_color", "unique_colors",
    "solid_fill_most_common", "solid_fill_least_common", "solid_fill_zero", "invert_colors",
    "nonzero_bbox", "crop_to_nonzero_bbox", "color_bbox", "crop_to_color_bbox",
    "tile_3x3_self", "tile_2x2_self", "scale_up_2x", "scale_up_3x", "scale_down_2x",
    "complete_horizontal_symmetry", "complete_vertical_symmetry",
    "color_map_from_pair", "apply_color_map", "infer_consistent_color_map",
    "connected_components", "largest_component_isolate", "smallest_component_isolate",
]
