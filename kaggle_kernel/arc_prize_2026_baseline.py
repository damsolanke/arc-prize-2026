"""ARC Prize 2026 ARC-AGI-2 — Heuristic + Ensemble Baseline (Paper Track companion).

Self-contained Kaggle kernel submission for arc-prize-2026-arc-agi-2.
Repo: https://github.com/damsolanke/arc-prize-2026

Runs a verified ensemble over 12 heuristic solvers on the 240 hidden test tasks
and writes /kaggle/working/submission.json. Expected LB: ~0% (heuristics alone
cannot solve AGI-2). Purpose: establish submission pipeline for the paper's
cross-family hybrid experiments (September 2026).

MIT-0 license. Dam Solanke 2026.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ==================== DATA ====================

Grid = np.ndarray


@dataclass(frozen=True)
class Pair:
    input: Grid
    output: Grid | None


@dataclass(frozen=True)
class Task:
    task_id: str
    train: tuple[Pair, ...]
    test: tuple[Pair, ...]


def _to_grid(data: list[list[int]]) -> Grid:
    arr = np.array(data, dtype=np.int8)
    if arr.ndim != 2:
        raise ValueError(f"Grid must be 2D, got {arr.shape}")
    return arr


def load_kaggle_test(path: Path) -> list[Task]:
    with open(path) as f:
        challenges = json.load(f)
    tasks = []
    for task_id in sorted(challenges):
        raw = challenges[task_id]
        train = tuple(
            Pair(_to_grid(p["input"]), _to_grid(p["output"])) for p in raw["train"]
        )
        test = tuple(Pair(input=_to_grid(p["input"]), output=None) for p in raw["test"])
        tasks.append(Task(task_id=task_id, train=train, test=test))
    return tasks


# ==================== HEURISTICS ====================

Transform = Callable[[Grid], Grid]


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


def most_common_color(g: Grid) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[counts.argmax()])


def least_common_color(g: Grid) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[counts.argmin()])


def solid_fill_most_common(g: Grid) -> Grid:
    return np.full_like(g, most_common_color(g))


def invert_colors(g: Grid) -> Grid:
    out = g.copy()
    mc, lc = most_common_color(g), least_common_color(g)
    if mc == lc:
        return out
    out[g == mc] = lc
    out[g == lc] = mc
    return out


def crop_to_nonzero_bbox(g: Grid) -> Grid:
    mask = g != 0
    if not mask.any():
        return g.copy()
    rs, cs = np.where(mask)
    return g[rs.min():rs.max() + 1, cs.min():cs.max() + 1].copy()


def tile_2x2_self(g: Grid) -> Grid:
    return np.tile(g, (2, 2))


def tile_3x3_self(g: Grid) -> Grid:
    return np.tile(g, (3, 3))


def scale_up_2x(g: Grid) -> Grid:
    return np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)


def scale_up_3x(g: Grid) -> Grid:
    return np.repeat(np.repeat(g, 3, axis=0), 3, axis=1)


def complete_horizontal_symmetry(g: Grid) -> Grid:
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


def largest_component_isolate(g: Grid) -> Grid:
    from scipy.ndimage import label
    mask = g != 0
    lbl, n = label(mask)
    if n == 0:
        return g.copy()
    sizes = [(lbl == i).sum() for i in range(1, n + 1)]
    biggest = 1 + int(np.argmax(sizes))
    out = np.zeros_like(g)
    out[lbl == biggest] = g[lbl == biggest]
    return out


# ==================== OBJECT-AWARE HEURISTICS ====================

def gravity_down(g: Grid) -> Grid:
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
    from scipy.ndimage import binary_fill_holes
    out = g.copy()
    for col in np.unique(g):
        if col == 0:
            continue
        mask = g == col
        filled = binary_fill_holes(mask)
        out[filled & ~mask] = col
    return out


def _components(g: Grid):
    from scipy.ndimage import label
    labeled, n = label(g != 0)
    comps = []
    for i in range(1, n + 1):
        m = labeled == i
        rs, cs = np.where(m)
        bbox = (int(rs.min()), int(cs.min()), int(rs.max()) + 1, int(cs.max()) + 1)
        colors_in = np.unique(g[m])
        color = int(colors_in[0]) if len(colors_in) > 0 else 0
        comps.append((m, bbox, color))
    return comps


def crop_to_largest_component(g: Grid) -> Grid:
    comps = _components(g)
    if not comps:
        return g.copy()
    largest = max(comps, key=lambda c: int(c[0].sum()))
    r0, c0, r1, c1 = largest[1]
    return g[r0:r1, c0:c1].copy()


def crop_to_smallest_component(g: Grid) -> Grid:
    comps = _components(g)
    if not comps:
        return g.copy()
    smallest = min(comps, key=lambda c: int(c[0].sum()))
    r0, c0, r1, c1 = smallest[1]
    return g[r0:r1, c0:c1].copy()


def smallest_component_color_fill(g: Grid) -> Grid:
    comps = _components(g)
    if not comps:
        return np.zeros_like(g)
    smallest = min(comps, key=lambda c: int(c[0].sum()))
    out = np.zeros_like(g)
    out[smallest[0]] = smallest[2]
    return out


OBJECT_HEURISTICS_NAMED: tuple[tuple[str, Transform], ...] = (
    ("gravity_down", gravity_down),
    ("gravity_up", gravity_up),
    ("gravity_left", gravity_left),
    ("gravity_right", gravity_right),
    ("fill_holes", fill_holes),
    ("crop_to_largest_component", crop_to_largest_component),
    ("crop_to_smallest_component", crop_to_smallest_component),
    ("smallest_component_color_fill", smallest_component_color_fill),
)


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


def infer_consistent_color_map(pairs) -> dict[int, int] | None:
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


BASIC_TRANSFORMS: tuple[tuple[str, Transform], ...] = (
    ("identity", identity),
    ("flip_h", flip_h), ("flip_v", flip_v),
    ("rot90", rot90), ("rot180", rot180), ("rot270", rot270),
    ("transpose", transpose), ("anti_transpose", anti_transpose),
    ("solid_fill_most_common", solid_fill_most_common),
    ("invert_colors", invert_colors),
    ("crop_to_nonzero_bbox", crop_to_nonzero_bbox),
    ("tile_2x2_self", tile_2x2_self),
    ("tile_3x3_self", tile_3x3_self),
    ("scale_up_2x", scale_up_2x),
    ("scale_up_3x", scale_up_3x),
    ("complete_horizontal_symmetry", complete_horizontal_symmetry),
    ("complete_vertical_symmetry", complete_vertical_symmetry),
    ("largest_component_isolate", largest_component_isolate),
) + OBJECT_HEURISTICS_NAMED

CORE_TRANSFORMS = tuple(t for t in BASIC_TRANSFORMS if t[0] not in (
    "tile_2x2_self", "tile_3x3_self", "scale_up_2x", "scale_up_3x",
    "solid_fill_most_common", "largest_component_isolate",
))


def _safe_apply(fn: Transform, g: Grid) -> Grid | None:
    try:
        out = fn(g)
        if out.size > 900:
            return None
        return out
    except Exception:
        return None


def _compose2(f1: Transform, f2: Transform) -> Transform:
    def _c(g: Grid) -> Grid:
        mid = _safe_apply(f1, g)
        if mid is None:
            raise ValueError()
        out = _safe_apply(f2, mid)
        if out is None:
            raise ValueError()
        return out
    return _c


def _fits_train(fn: Transform, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = fn(p.input)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


# ==================== INDUCTION (neighborhood-rule) ====================

def _neighborhood(g: Grid, r: int, c: int, radius: int = 1) -> tuple:
    h, w = g.shape
    n = 2 * radius + 1
    patch = np.full((n, n), 0, dtype=np.int8)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                patch[dr + radius, dc + radius] = g[rr, cc]
    return tuple(patch.flatten().tolist())


def _induce_rule(pairs, radius: int = 1):
    """Return neighborhood -> output-cell mapping if consistent, else None."""
    if not pairs:
        return None
    rule = {}
    for inp, out in pairs:
        if inp.shape != out.shape:
            return None
        h, w = inp.shape
        for r in range(h):
            for c in range(w):
                key = _neighborhood(inp, r, c, radius)
                val = int(out[r, c])
                if key in rule and rule[key] != val:
                    return None
                rule[key] = val
    return rule


def _apply_rule(rule, g: Grid, radius: int = 1):
    out = np.zeros_like(g)
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            key = _neighborhood(g, r, c, radius)
            if key not in rule:
                return None
            out[r, c] = rule[key]
    return out


def apply_induction(task: Task, test_input: Grid, radius: int = 1):
    """Return induction prediction if rule consistent, else None."""
    pairs = [(p.input, p.output) for p in task.train if p.output is not None]
    rule = _induce_rule(pairs, radius=radius)
    if rule is None:
        return None
    return _apply_rule(rule, test_input, radius=radius)


# ==================== CELLULAR AUTOMATA ====================

def _apply_ca_rule_any_neighbor(g: Grid) -> Grid:
    h, w = g.shape
    out = np.zeros_like(g)
    for r in range(h):
        for c in range(w):
            counts = {}
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        col = int(g[rr, cc])
                        counts[col] = counts.get(col, 0) + 1
            if counts:
                out[r, c] = max(counts, key=counts.get)
            else:
                out[r, c] = g[r, c]
    return out


def _apply_ca_grow(g: Grid) -> Grid:
    """Empty cells take the most common non-zero neighbor color."""
    h, w = g.shape
    out = g.copy()
    for r in range(h):
        for c in range(w):
            if g[r, c] != 0:
                continue
            counts = {}
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        col = int(g[rr, cc])
                        if col != 0:
                            counts[col] = counts.get(col, 0) + 1
            if counts:
                out[r, c] = max(counts, key=counts.get)
    return out


def _fits_ca(ca_fn, task: Task) -> bool:
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = ca_fn(p.input)
        except Exception:
            return False
        if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
            return False
    return True


def apply_ca(task: Task, test_input: Grid):
    """Return CA prediction if any rule fits all train pairs, else None."""
    for ca_fn in (_apply_ca_rule_any_neighbor, _apply_ca_grow):
        if _fits_ca(ca_fn, task):
            try:
                return ca_fn(test_input)
            except Exception:
                continue
    return None


# ==================== HISTOGRAM ====================

def _bar_vertical(g: Grid) -> Grid:
    counts = {}
    for c in np.unique(g):
        if c != 0:
            counts[int(c)] = int((g == c).sum())
    if not counts:
        return np.zeros_like(g)
    max_count = max(counts.values())
    colors_sorted = sorted(counts, key=counts.get, reverse=True)
    w = len(colors_sorted)
    out = np.zeros((max_count, w), dtype=np.int8)
    for j, col in enumerate(colors_sorted):
        out[-counts[col]:, j] = col
    return out


def _bar_horizontal(g: Grid) -> Grid:
    counts = {}
    for c in np.unique(g):
        if c != 0:
            counts[int(c)] = int((g == c).sum())
    if not counts:
        return np.zeros_like(g)
    max_count = max(counts.values())
    colors_sorted = sorted(counts, key=counts.get, reverse=True)
    h = len(colors_sorted)
    out = np.zeros((h, max_count), dtype=np.int8)
    for i, col in enumerate(colors_sorted):
        out[i, :counts[col]] = col
    return out


def apply_histogram(task: Task, test_input: Grid):
    for fn in (_bar_vertical, _bar_horizontal):
        fits = True
        for p in task.train:
            if p.output is None:
                continue
            try:
                pred = fn(p.input)
            except Exception:
                fits = False; break
            if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
                fits = False; break
        if fits:
            try:
                return fn(test_input)
            except Exception:
                continue
    return None


# ==================== DIM (shape-from-count) ====================

def _dim_from_count(g: Grid):
    import numpy as np2
    n = int(np2.count_nonzero(g))
    return (n, n) if n > 0 else None


def apply_dim(task: Task, test_input: Grid):
    """Detect: output shape = (count_nonzero, count_nonzero), fill = most_common_nonzero."""
    # Try: output shape = NxN where N = count_nonzero, filled with most-common input color
    def _check_rule(shape_fn, fill_fn):
        for p in task.train:
            if p.output is None:
                continue
            shape = shape_fn(p.input)
            if shape is None:
                return False
            try:
                fill = fill_fn(p.input)
            except Exception:
                return False
            if shape != p.output.shape:
                return False
            expected = np.full(shape, fill, dtype=np.int8)
            if not np.array_equal(expected, p.output):
                return False
        return True

    def _most_common(g):
        if (g != 0).any():
            v, c = np.unique(g[g != 0], return_counts=True)
            return int(v[c.argmax()])
        return 0

    shape_fns = [
        lambda g: (int(np.count_nonzero(g)), int(np.count_nonzero(g))),
        lambda g: (len(set(g.flatten().tolist()) - {0}), len(set(g.flatten().tolist()) - {0})),
        lambda g: (1, 1),
    ]
    fill_fns = [
        _most_common,
        lambda g: len(set(g.flatten().tolist()) - {0}),
        lambda g: min(int(np.count_nonzero(g)), 9),
    ]
    for sf in shape_fns:
        for ff in fill_fns:
            if _check_rule(sf, ff):
                try:
                    h, w = sf(test_input)
                    f = ff(test_input)
                    return np.full((h, w), f, dtype=np.int8)
                except Exception:
                    continue
    return None


# ==================== SUBGRID ====================

def _find_subgrids(g: Grid, divider: int = 0):
    try:
        from scipy.ndimage import label
    except ImportError:
        return []
    mask = g != divider
    labeled, n = label(mask)
    subs = []
    for i in range(1, n + 1):
        m = labeled == i
        rs, cs = np.where(m)
        r0, c0, r1, c1 = int(rs.min()), int(cs.min()), int(rs.max()) + 1, int(cs.max()) + 1
        box = g[r0:r1, c0:c1]
        if (box != divider).all():
            subs.append((r0, c0, r1, c1, box.copy()))
    return subs


def apply_subgrid(task: Task, test_input: Grid):
    for pick_fn in (
        lambda g: max(_find_subgrids(g), key=lambda s: s[4].size, default=None),
        lambda g: min(_find_subgrids(g), key=lambda s: s[4].size, default=None),
    ):
        fits = True
        for p in task.train:
            if p.output is None:
                continue
            try:
                s = pick_fn(p.input)
                if s is None:
                    fits = False; break
                if s[4].shape != p.output.shape or not np.array_equal(s[4], p.output):
                    fits = False; break
            except Exception:
                fits = False; break
        if fits:
            try:
                s = pick_fn(test_input)
                if s is not None:
                    return s[4]
            except Exception:
                continue
    return None


# ==================== PROPAGATION ====================

def _propagate_direction(g: Grid, dr: int, dc: int) -> Grid:
    h, w = g.shape
    out = g.copy()
    for r in range(h):
        for c in range(w):
            if g[r, c] == 0:
                continue
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w:
                if out[rr, cc] == 0:
                    out[rr, cc] = g[r, c]
                rr += dr
                cc += dc
    return out


def apply_propagation(task: Task, test_input: Grid):
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
        fits = True
        for p in task.train:
            if p.output is None:
                continue
            try:
                pred = _propagate_direction(p.input, dr, dc)
            except Exception:
                fits = False; break
            if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
                fits = False; break
        if fits:
            try:
                return _propagate_direction(test_input, dr, dc)
            except Exception:
                continue
    return None


# ==================== SIZE-BASED RECOLOR ====================

def _components_with_sizes(g: Grid):
    try:
        from scipy.ndimage import label
    except ImportError:
        return []
    labeled, n = label(g != 0)
    comps = []
    for i in range(1, n + 1):
        m = labeled == i
        comps.append((m, int(m.sum())))
    return comps


def apply_size_color(task: Task, test_input: Grid):
    def _recolor(g, large, small):
        comps = _components_with_sizes(g)
        if not comps:
            return g.copy()
        out = g.copy()
        sizes = sorted({s for _, s in comps})
        max_s, min_s = sizes[-1], sizes[0]
        for mask, size in comps:
            if size == max_s:
                out[mask] = large
            elif size == min_s:
                out[mask] = small
        return out
    for large in range(1, 10):
        for small in range(10):
            if large == small:
                continue
            fits = True
            for p in task.train:
                if p.output is None:
                    continue
                try:
                    pred = _recolor(p.input, large, small)
                except Exception:
                    fits = False; break
                if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
                    fits = False; break
            if fits:
                try:
                    return _recolor(test_input, large, small)
                except Exception:
                    continue
    return None


# ==================== MOVEMENT (per-color offset) ====================

def apply_box(task: Task, test_input: Grid):
    """Draw bounding box of specific color around components."""
    def _draw(g, color):
        try:
            from scipy.ndimage import label
        except ImportError:
            return None
        out = g.copy()
        labeled, n = label(g != 0)
        for i in range(1, n + 1):
            m = labeled == i
            rs, cs = np.where(m)
            r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
            for c in range(c0, c1 + 1):
                if out[r0, c] == 0:
                    out[r0, c] = color
                if out[r1, c] == 0:
                    out[r1, c] = color
            for r in range(r0, r1 + 1):
                if out[r, c0] == 0:
                    out[r, c0] = color
                if out[r, c1] == 0:
                    out[r, c1] = color
        return out

    for color in range(1, 10):
        fits = True
        for p in task.train:
            if p.output is None:
                continue
            try:
                pred = _draw(p.input, color)
            except Exception:
                fits = False; break
            if pred is None or pred.shape != p.output.shape or not np.array_equal(pred, p.output):
                fits = False; break
        if fits:
            try:
                return _draw(test_input, color)
            except Exception:
                continue
    return None


def apply_grid_divide(task: Task, test_input: Grid):
    """When input divided by uniform-color lines, pick the unique subgrid."""
    def _find(g, divider):
        h, w = g.shape
        row_mask = np.all(g == divider, axis=1)
        col_mask = np.all(g == divider, axis=0)
        if not row_mask.any() or not col_mask.any():
            return []
        rb = [-1] + [i for i in range(h) if row_mask[i]] + [h]
        rs = [(rb[i] + 1, rb[i + 1]) for i in range(len(rb) - 1) if rb[i + 1] > rb[i] + 1]
        cb = [-1] + [j for j in range(w) if col_mask[j]] + [w]
        cs = [(cb[i] + 1, cb[i + 1]) for i in range(len(cb) - 1) if cb[i + 1] > cb[i] + 1]
        return [g[r0:r1, c0:c1].copy() for r0, r1 in rs for c0, c1 in cs if r1 > r0 and c1 > c0]

    def _pick_unique(g, divider):
        subs = _find(g, divider)
        if not subs:
            return None
        from collections import Counter
        keys = [s.tobytes() for s in subs]
        counts = Counter(keys)
        uniq = [k for k, c in counts.items() if c == 1]
        if len(uniq) == 1:
            idx = keys.index(uniq[0])
            return subs[idx]
        return None

    for divider in range(10):
        fits = True
        for p in task.train:
            if p.output is None:
                continue
            sub = _pick_unique(p.input, divider)
            if sub is None or sub.shape != p.output.shape or not np.array_equal(sub, p.output):
                fits = False; break
        if fits:
            sub = _pick_unique(test_input, divider)
            if sub is not None:
                return sub
    return None


def apply_axis_mirror(task: Task, test_input: Grid):
    """Concat input with mirror along axis."""
    mirrors = [
        lambda g: np.concatenate([g, g[:, ::-1]], axis=1),
        lambda g: np.concatenate([g[:, ::-1], g], axis=1),
        lambda g: np.concatenate([g, g[::-1, :]], axis=0),
        lambda g: np.concatenate([g[::-1, :], g], axis=0),
    ]
    for fn in mirrors:
        fits = True
        for p in task.train:
            if p.output is None:
                continue
            try:
                pred = fn(p.input)
            except Exception:
                fits = False; break
            if pred.shape != p.output.shape or not np.array_equal(pred, p.output):
                fits = False; break
        if fits:
            try:
                return fn(test_input)
            except Exception:
                continue
    return None


def apply_movement(task: Task, test_input: Grid):
    def _find_offsets(inp, out):
        if inp.shape != out.shape:
            return None
        mapping = {}
        for color in np.unique(inp):
            if color == 0:
                continue
            ip = sorted(zip(*np.where(inp == color)))
            op = sorted(zip(*np.where(out == color)))
            if not ip or len(ip) != len(op):
                return None
            dr = op[0][0] - ip[0][0]
            dc = op[0][1] - ip[0][1]
            for (ir, ic), (orr, oc) in zip(ip, op):
                if orr - ir != dr or oc - ic != dc:
                    return None
            mapping[int(color)] = (int(dr), int(dc))
        return mapping

    mapping = None
    for p in task.train:
        if p.output is None:
            continue
        m = _find_offsets(p.input, p.output)
        if m is None:
            return None
        if mapping is None:
            mapping = m
        else:
            for c, off in m.items():
                if c in mapping and mapping[c] != off:
                    return None
                mapping[c] = off
    if mapping is None:
        return None
    # Apply to test
    h, w = test_input.shape
    out = np.zeros_like(test_input)
    for r in range(h):
        for c in range(w):
            col = int(test_input[r, c])
            if col == 0:
                continue
            dr, dc = mapping.get(col, (0, 0))
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = col
    return out


def _partial_fit_score(fn: Transform, task: Task) -> float:
    """Best cell-wise match rate across any train pair; fallback when no exact fit."""
    best = 0.0
    for p in task.train:
        if p.output is None:
            continue
        try:
            pred = fn(p.input)
        except Exception:
            continue
        if pred.shape != p.output.shape:
            continue
        rate = float((pred == p.output).sum()) / float(p.output.size)
        if rate > best:
            best = rate
    return best


# ==================== SOLVER ====================

def solve_task(task: Task) -> list[list[Grid]]:
    """Try single heuristics + 2-op compositions + color-map. Returns per-test-input
    list of up to 2 candidate grids (attempt_1, attempt_2)."""
    cands: list[tuple[str, Transform]] = list(BASIC_TRANSFORMS)
    for n1, f1 in CORE_TRANSFORMS:
        for n2, f2 in CORE_TRANSFORMS:
            if n1 == "identity" or n2 == "identity":
                continue
            cands.append((f"{n1}>{n2}", _compose2(f1, f2)))

    pairs = tuple((p.input, p.output) for p in task.train if p.output is not None)
    if pairs:
        m = infer_consistent_color_map(pairs)
        if m:
            color_fn = lambda g, mm=m: apply_color_map(g, mm)  # noqa: E731
            cands.append(("color_map", color_fn))

    winners: list[Transform] = [fn for _, fn in cands if _fits_train(fn, task)]
    if not winners:
        # Fallback: rank by best cell-wise match rate; take top 2.
        # Matches local composition_solver recovery (+3 tasks in train).
        scored = sorted(cands, key=lambda kv: _partial_fit_score(kv[1], task), reverse=True)
        winners = [fn for _, fn in scored[:2]]
    while len(winners) < 2:
        winners.append(winners[-1])

    # Build induction predictor (disjoint family) — use as attempt_2 when composition
    # has a train-fit winner AND induction also has a consistent rule. This is the
    # +8 disjoint-tasks finding from paper §5.1 ported into the submission.
    out_per_test: list[list[Grid]] = []
    for test_pair in task.test:
        per_test: list[Grid] = []
        # attempt_1: composition winner
        try:
            per_test.append(winners[0](test_pair.input))
        except Exception:
            per_test.append(test_pair.input.copy())
        # attempt_2: priority = 12 orthogonal families ordered by disjoint-contribution
        for candidate_fn in (
            lambda: apply_induction(task, test_pair.input, radius=1),
            lambda: apply_axis_mirror(task, test_pair.input),
            lambda: apply_ca(task, test_pair.input),
            lambda: apply_histogram(task, test_pair.input),
            lambda: apply_dim(task, test_pair.input),
            lambda: apply_subgrid(task, test_pair.input),
            lambda: apply_propagation(task, test_pair.input),
            lambda: apply_size_color(task, test_pair.input),
            lambda: apply_movement(task, test_pair.input),
            lambda: apply_box(task, test_pair.input),
            lambda: apply_grid_divide(task, test_pair.input),
        ):
            pred = candidate_fn()
            if pred is not None:
                per_test.append(pred)
                break
        else:
            try:
                per_test.append(winners[1](test_pair.input))
            except Exception:
                per_test.append(test_pair.input.copy())
        while len(per_test) < 2:
            per_test.append(per_test[-1].copy())
        out_per_test.append(per_test[:2])
    return out_per_test


# ==================== MAIN ====================

def _find_test_challenges() -> Path:
    """Find arc-agi_test_challenges.json wherever Kaggle mounted it."""
    candidates = [
        Path("/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"),
        Path("/kaggle/input/arc-prize-2026/arc-agi_test_challenges.json"),
    ]
    for p in candidates:
        if p.exists():
            return p
    # Glob-search /kaggle/input
    ki = Path("/kaggle/input")
    if ki.exists():
        for match in ki.rglob("arc-agi_test_challenges.json"):
            return match
        print("/kaggle/input contents:")
        for sub in ki.iterdir():
            print(f"  {sub}")
            try:
                for sub2 in sub.iterdir():
                    print(f"    {sub2.name}")
            except Exception:
                pass
    # Local fallback
    local = Path(__file__).resolve().parents[1] / "data" / "arc-prize-2026-arc-agi-2" / "arc-agi_test_challenges.json"
    if local.exists():
        return local
    raise FileNotFoundError("arc-agi_test_challenges.json not found in any candidate path")


def main() -> None:
    in_path = _find_test_challenges()
    out_path = Path("/kaggle/working/submission.json")
    if not out_path.parent.exists():
        out_path = Path("/tmp/submission.json")

    print(f"input: {in_path}")
    print(f"output: {out_path}")

    t0 = time.perf_counter()
    tasks = load_kaggle_test(in_path)
    print(f"loaded {len(tasks)} test tasks in {time.perf_counter()-t0:.2f}s")

    submission: dict[str, list[dict]] = {}
    t0 = time.perf_counter()
    for i, task in enumerate(tasks):
        candidates = solve_task(task)
        submission[task.task_id] = [
            {
                "attempt_1": cands[0].astype(int).tolist(),
                "attempt_2": cands[1].astype(int).tolist(),
            }
            for cands in candidates
        ]
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tasks)}", flush=True)
    wall = time.perf_counter() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(submission))
    print(f"wrote {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
    print(f"wall: {wall:.2f}s ({wall/len(tasks)*1000:.1f}ms/task)")
    print(f"tasks covered: {len(submission)}/{len(tasks)}")


if __name__ == "__main__":
    main()
