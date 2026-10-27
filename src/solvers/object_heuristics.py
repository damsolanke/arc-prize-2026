"""Object-aware heuristics: operations that detect connected components and
do per-object transformations. Covers tasks where the transformation is
"for each object X in input, do Y to produce output-object X'".
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid

try:
    from scipy.ndimage import label
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _connected_components(g: Grid, ignore: int = 0, diagonal: bool = False):
    """Return list of (mask, bbox, color) for each connected component."""
    if not SCIPY_AVAILABLE:
        return []
    structure = np.ones((3, 3)) if diagonal else None
    mask_nonzero = g != ignore
    labeled, n = label(mask_nonzero, structure=structure)
    comps = []
    for i in range(1, n + 1):
        m = labeled == i
        rs, cs = np.where(m)
        bbox = (int(rs.min()), int(cs.min()), int(rs.max()) + 1, int(cs.max()) + 1)
        # assume single-color components (common in ARC)
        colors_in_mask = np.unique(g[m])
        color = int(colors_in_mask[0]) if len(colors_in_mask) > 0 else 0
        comps.append((m, bbox, color))
    return comps


def gravity_down(g: Grid) -> Grid:
    """Let each non-zero cell 'fall' to the bottom of its column (background=0)."""
    out = np.zeros_like(g)
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        non_bg = col[col != 0]
        out[h - len(non_bg):, c] = non_bg
    return out


def gravity_up(g: Grid) -> Grid:
    out = np.zeros_like(g)
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        non_bg = col[col != 0]
        out[:len(non_bg), c] = non_bg
    return out


def gravity_left(g: Grid) -> Grid:
    out = np.zeros_like(g)
    h, w = g.shape
    for r in range(h):
        row = g[r, :]
        non_bg = row[row != 0]
        out[r, :len(non_bg)] = non_bg
    return out


def gravity_right(g: Grid) -> Grid:
    out = np.zeros_like(g)
    h, w = g.shape
    for r in range(h):
        row = g[r, :]
        non_bg = row[row != 0]
        out[r, w - len(non_bg):] = non_bg
    return out


def fill_holes(g: Grid) -> Grid:
    """For each connected non-background component, fill its enclosed holes with its color."""
    if not SCIPY_AVAILABLE:
        return g.copy()
    from scipy.ndimage import binary_fill_holes
    out = g.copy()
    for col in np.unique(g):
        if col == 0:
            continue
        mask = g == col
        filled = binary_fill_holes(mask)
        out[filled & ~mask] = col
    return out


def outline_objects(g: Grid, outline_color: int = 0) -> Grid:
    """Draw a 1-pixel outline around every non-bg cell that borders a bg cell."""
    out = g.copy()
    h, w = g.shape
    bg_mask = g == outline_color
    for i in range(h):
        for j in range(w):
            if g[i, j] == outline_color:
                continue
            # check if any of 4-neighbors is background — if so, keep; else clear
            is_edge = False
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and bg_mask[ni, nj]:
                    is_edge = True
                    break
                if not (0 <= ni < h and 0 <= nj < w):
                    is_edge = True
                    break
            if not is_edge:
                out[i, j] = outline_color
    return out


def count_color_replace(g: Grid) -> Grid:
    """Replace each color with a cell whose value = count of that color.
    Output is always 1x1 with value clamped to 0-9. For 'count the X' tasks.
    """
    vals, counts = np.unique(g, return_counts=True)
    # Pick the most-common non-zero count, cap at 9
    non_bg = [(v, c) for v, c in zip(vals, counts) if v != 0]
    if not non_bg:
        return np.array([[0]], dtype=np.int8)
    count = max(c for _, c in non_bg)
    return np.array([[min(count, 9)]], dtype=np.int8)


def largest_component_color_fill(g: Grid) -> Grid:
    """Return a grid the same shape as input, with largest component's cells
    set to the largest component's color, everything else = 0."""
    comps = _connected_components(g)
    if not comps:
        return np.zeros_like(g)
    largest = max(comps, key=lambda c: int(c[0].sum()))
    out = np.zeros_like(g)
    out[largest[0]] = largest[2]
    return out


def smallest_component_color_fill(g: Grid) -> Grid:
    comps = _connected_components(g)
    if not comps:
        return np.zeros_like(g)
    smallest = min(comps, key=lambda c: int(c[0].sum()))
    out = np.zeros_like(g)
    out[smallest[0]] = smallest[2]
    return out


def crop_to_largest_component(g: Grid) -> Grid:
    comps = _connected_components(g)
    if not comps:
        return g.copy()
    largest = max(comps, key=lambda c: int(c[0].sum()))
    r0, c0, r1, c1 = largest[1]
    return g[r0:r1, c0:c1].copy()


def crop_to_smallest_component(g: Grid) -> Grid:
    comps = _connected_components(g)
    if not comps:
        return g.copy()
    smallest = min(comps, key=lambda c: int(c[0].sum()))
    r0, c0, r1, c1 = smallest[1]
    return g[r0:r1, c0:c1].copy()


OBJECT_HEURISTICS: tuple[tuple[str, object], ...] = (
    ("gravity_down", gravity_down),
    ("gravity_up", gravity_up),
    ("gravity_left", gravity_left),
    ("gravity_right", gravity_right),
    ("fill_holes", fill_holes),
    ("outline_objects", outline_objects),
    ("count_color_replace", count_color_replace),
    ("largest_component_color_fill", largest_component_color_fill),
    ("smallest_component_color_fill", smallest_component_color_fill),
    ("crop_to_largest_component", crop_to_largest_component),
    ("crop_to_smallest_component", crop_to_smallest_component),
)


__all__ = [
    "gravity_down",
    "gravity_up",
    "gravity_left",
    "gravity_right",
    "fill_holes",
    "outline_objects",
    "count_color_replace",
    "largest_component_color_fill",
    "smallest_component_color_fill",
    "crop_to_largest_component",
    "crop_to_smallest_component",
    "OBJECT_HEURISTICS",
]
