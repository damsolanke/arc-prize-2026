"""Diverse single-strategy solvers for use as inner solvers in EnsembleSolver.

Each targets a different task family — they're intentionally narrow. Together
they give a heterogeneous ensemble whose vote tends to hit the right answer
when the task matches one of the strategies.
"""
from __future__ import annotations

import numpy as np

from ..eval.loader import Task
from .base import BaseSolver, Prediction
from .heuristics import (
    D8,
    apply_color_map,
    complete_horizontal_symmetry,
    complete_vertical_symmetry,
    crop_to_nonzero_bbox,
    infer_consistent_color_map,
    invert_colors,
    largest_component_isolate,
    scale_up_2x,
    scale_up_3x,
    smallest_component_isolate,
    solid_fill_most_common,
    tile_2x2_self,
    tile_3x3_self,
)


def _fits_train(fn, task: Task) -> bool:
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


class IdentitySolver(BaseSolver):
    """Trivial — always predict test input unchanged. Calibration floor."""

    name = "identity_solver"

    def solve(self, task: Task) -> Prediction:
        candidates = [
            [p.input.copy(), p.input.copy(), p.input.copy()] for p in task.test
        ]
        return Prediction(task_id=task.task_id, candidates=candidates)


class D8Solver(BaseSolver):
    """Try all 8 D8 transforms, pick ones that fit train pairs."""

    name = "d8_solver"

    def solve(self, task: Task) -> Prediction:
        winners = [fn for fn in D8 if _fits_train(fn, task)][:3]
        if not winners:
            winners = list(D8[:3])
        while len(winners) < 3:
            winners.append(winners[-1])
        candidates = []
        for test_pair in task.test:
            test_cands = []
            for fn in winners:
                try:
                    test_cands.append(fn(test_pair.input))
                except Exception:
                    test_cands.append(test_pair.input.copy())
            candidates.append(test_cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


class ColorMapSolver(BaseSolver):
    """Infer per-task color mapping; apply if consistent across train pairs."""

    name = "color_map_solver"

    def solve(self, task: Task) -> Prediction:
        pairs = tuple((p.input, p.output) for p in task.train if p.output is not None)
        m = infer_consistent_color_map(pairs) if pairs else None

        candidates = []
        for test_pair in task.test:
            if m:
                g_mapped = apply_color_map(test_pair.input, m)
                g_mapped2 = invert_colors(test_pair.input)
                candidates.append([g_mapped, g_mapped2, test_pair.input.copy()])
            else:
                # fallback
                candidates.append([
                    test_pair.input.copy(),
                    invert_colors(test_pair.input),
                    solid_fill_most_common(test_pair.input),
                ])
        return Prediction(task_id=task.task_id, candidates=candidates)


class SymmetrySolver(BaseSolver):
    """Complete horizontal or vertical symmetry."""

    name = "symmetry_solver"

    def solve(self, task: Task) -> Prediction:
        # Figure out which symmetry direction fits training best
        h_fit = _fits_train(complete_horizontal_symmetry, task)
        v_fit = _fits_train(complete_vertical_symmetry, task)
        candidates = []
        for test_pair in task.test:
            cands = []
            try:
                cands.append(complete_horizontal_symmetry(test_pair.input))
            except Exception:
                cands.append(test_pair.input.copy())
            try:
                cands.append(complete_vertical_symmetry(test_pair.input))
            except Exception:
                cands.append(test_pair.input.copy())
            cands.append(test_pair.input.copy())
            # reorder so fitting comes first
            if v_fit and not h_fit:
                cands = [cands[1], cands[0], cands[2]]
            candidates.append(cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


class TilingSolver(BaseSolver):
    """Tile or scale the input (common pattern-completion trick)."""

    name = "tiling_solver"

    def solve(self, task: Task) -> Prediction:
        cands_fns = [tile_2x2_self, tile_3x3_self, scale_up_2x, scale_up_3x]
        # rank by fit on train
        fit_scores = []
        for fn in cands_fns:
            fit = _fits_train(fn, task)
            fit_scores.append((fn, fit))
        fit_scores.sort(key=lambda x: (not x[1]))
        winners = [fn for fn, _ in fit_scores[:3]]

        candidates = []
        for test_pair in task.test:
            test_cands = []
            for fn in winners:
                try:
                    test_cands.append(fn(test_pair.input))
                except Exception:
                    test_cands.append(test_pair.input.copy())
            candidates.append(test_cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


class ObjectIsolationSolver(BaseSolver):
    """Isolate largest/smallest connected component or crop to bbox."""

    name = "object_isolation_solver"

    def solve(self, task: Task) -> Prediction:
        fns = [largest_component_isolate, smallest_component_isolate, crop_to_nonzero_bbox]
        fit_scores = []
        for fn in fns:
            fit = _fits_train(fn, task)
            fit_scores.append((fn, fit))
        fit_scores.sort(key=lambda x: (not x[1]))
        winners = [fn for fn, _ in fit_scores[:3]]

        candidates = []
        for test_pair in task.test:
            test_cands = []
            for fn in winners:
                try:
                    test_cands.append(fn(test_pair.input))
                except Exception:
                    test_cands.append(test_pair.input.copy())
            candidates.append(test_cands[:3])
        return Prediction(task_id=task.task_id, candidates=candidates)


def default_ensemble_inner() -> list[BaseSolver]:
    """Return a diverse set of single-strategy solvers for EnsembleSolver.

    InductionSolver at radius=1 outperforms MultiRadiusInductionSolver in verified
    ensembles (+1 task), because r=0 induction fits train on too many tasks where
    it doesn't generalize, contaminating the vote. Keep multi-radius separate.
    """
    from .axis_mirror_solver import AxisMirrorSolver
    from .box_solver import BoxSolver
    from .xor_combine_solver import XORCombineSolver
    from .ca_solver import CellularAutomatonSolver
    from .color_count_solver import ColorCountSolver
    from .diff_solver import DiffSolver
    from .dim_solver import DimSolver
    from .grid_divide_solver import GridDivideSolver
    from .histogram_solver import HistogramSolver
    from .holes_solver import HolesSolver
    from .induction_solver import InductionSolver
    from .line_solver import LineSolver
    from .movement_solver import MovementSolver
    from .nested_solver import NestedSolver
    from .propagation_solver import PropagationSolver
    from .quadsym_solver import QuadSymmetrySolver
    from .size_color_solver import SizeColorSolver
    from .subgrid_solver import SubgridSolver
    return [
        D8Solver(),
        ColorMapSolver(),
        SymmetrySolver(),
        TilingSolver(),
        ObjectIsolationSolver(),
        IdentitySolver(),
        InductionSolver(radius=1),
        LineSolver(),
        ColorCountSolver(),
        DiffSolver(),
        CellularAutomatonSolver(),
        HistogramSolver(),
        DimSolver(),
        SubgridSolver(),
        PropagationSolver(),
        SizeColorSolver(),
        MovementSolver(),
        BoxSolver(),
        GridDivideSolver(),
        AxisMirrorSolver(),
        XORCombineSolver(),
        NestedSolver(),
        QuadSymmetrySolver(),
        HolesSolver(),
    ]


__all__ = [
    "IdentitySolver",
    "D8Solver",
    "ColorMapSolver",
    "SymmetrySolver",
    "TilingSolver",
    "ObjectIsolationSolver",
    "default_ensemble_inner",
]
