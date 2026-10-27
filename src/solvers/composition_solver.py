"""2-op composition solver: try every pair of basic heuristics; pick those fitting train.

Expanding the program-space search from depth-1 (single heuristic) to depth-2
(compose two heuristics). Depth-1 on AGI-2 train solved 19/1000. Depth-2 adds
hundreds more candidate programs but most are pathological (e.g. scale_up_3x
then scale_up_3x = 9x grid which exceeds 30x30 cap for many tasks).
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Grid, Task
from .base import BaseSolver, Prediction
from .heuristics import (
    BASIC_TRANSFORMS,
    Transform,
    apply_color_map,
    complete_horizontal_symmetry,
    complete_vertical_symmetry,
    crop_to_nonzero_bbox,
    infer_consistent_color_map,
    invert_colors,
    identity,
    flip_h,
    flip_v,
    rot90,
    rot180,
    rot270,
    transpose,
)
from .object_heuristics import OBJECT_HEURISTICS


# A smaller set of "core" transforms that compose well (don't explode grid size)
CORE_TRANSFORMS: tuple[tuple[str, Transform], ...] = (
    ("identity", identity),
    ("flip_h", flip_h),
    ("flip_v", flip_v),
    ("rot90", rot90),
    ("rot180", rot180),
    ("rot270", rot270),
    ("transpose", transpose),
    ("crop_to_nonzero_bbox", crop_to_nonzero_bbox),
    ("complete_horizontal_symmetry", complete_horizontal_symmetry),
    ("complete_vertical_symmetry", complete_vertical_symmetry),
    ("invert_colors", invert_colors),
)


MAX_GRID_SIZE = 900  # 30x30 is the ARC maximum


def _safe_apply(fn: Transform, g: Grid) -> Grid | None:
    try:
        out = fn(g)
        if out.size > MAX_GRID_SIZE:
            return None
        return out
    except Exception:
        return None


def _compose2(f1: Transform, f2: Transform) -> Transform:
    def _composed(g: Grid) -> Grid:
        mid = _safe_apply(f1, g)
        if mid is None:
            raise ValueError("first transform blew up")
        out = _safe_apply(f2, mid)
        if out is None:
            raise ValueError("second transform blew up")
        return out
    return _composed


def _fits_train(fn: Transform, task: Task) -> bool:
    for pair in task.train:
        if pair.output is None:
            continue
        try:
            pred = fn(pair.input)
        except Exception:
            return False
        if pred.shape != pair.output.shape or not np.array_equal(pred, pair.output):
            return False
    return True


def _partial_fit_score(fn: Transform, task: Task) -> float:
    """Best cell-wise match rate across any train pair. Used as a fallback
    ranking when no transform fits all train pairs exactly."""
    best = 0.0
    for pair in task.train:
        if pair.output is None:
            continue
        try:
            pred = fn(pair.input)
        except Exception:
            continue
        if pred.shape != pair.output.shape:
            continue
        rate = float((pred == pair.output).sum()) / float(pair.output.size)
        if rate > best:
            best = rate
    return best


class CompositionSolver(BaseSolver):
    """Try single heuristics + 2-op compositions + inferred color map."""

    name = "composition_solver"

    def _candidate_transforms(self, task: Task) -> list[tuple[str, Transform]]:
        cands: list[tuple[str, Transform]] = list(BASIC_TRANSFORMS) + list(OBJECT_HEURISTICS)

        # 2-op compositions from CORE_TRANSFORMS
        for n1, f1 in CORE_TRANSFORMS:
            for n2, f2 in CORE_TRANSFORMS:
                if n1 == "identity" or n2 == "identity":
                    continue  # redundant
                cands.append((f"{n1}>{n2}", _compose2(f1, f2)))

        # Color map + geometry composition
        pairs = tuple((p.input, p.output) for p in task.train if p.output is not None)
        if pairs:
            m = infer_consistent_color_map(pairs)
            if m:
                color_fn = lambda g, mm=m: apply_color_map(g, mm)  # noqa: E731
                cands.append(("color_map", color_fn))
                # color_map then flip/rotate
                for n, f in CORE_TRANSFORMS:
                    if n in ("identity", "invert_colors"):
                        continue
                    cands.append((f"color_map>{n}", _compose2(color_fn, f)))
                    cands.append((f"{n}>color_map", _compose2(f, color_fn)))
        return cands

    def solve(self, task: Task) -> Prediction:
        cands = self._candidate_transforms(task)
        # Preserve iteration order (BASIC + OBJECT first, then depth-2 compositions,
        # then color_map). Simpler programs therefore fill the top-2 slots first —
        # important under the 2026 N_ATTEMPTS=2 rule.
        winners: list[tuple[str, Transform]] = [(n, fn) for n, fn in cands if _fits_train(fn, task)]

        if not winners:
            # Fallback: pick top-3 by partial-fit score (matches transform_solver's
            # behavior so composition never underperforms transform on no-fit tasks).
            scored = sorted(cands, key=lambda kv: _partial_fit_score(kv[1], task), reverse=True)
            winners = scored[:3]
        while len(winners) < 3:
            winners.append(winners[-1])

        candidates: list[list[Grid]] = []
        for test_pair in task.test:
            test_cands: list[Grid] = []
            for _, fn in winners[:3]:
                try:
                    out = fn(test_pair.input)
                    if out.size > MAX_GRID_SIZE:
                        test_cands.append(test_pair.input.copy())
                    else:
                        test_cands.append(out)
                except Exception:
                    test_cands.append(test_pair.input.copy())
            candidates.append(test_cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


__all__ = ["CompositionSolver"]
