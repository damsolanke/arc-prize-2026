"""Compute structural features of ARC tasks for taxonomy + baseline routing.

These features let us answer: "what kind of task is this?" without a human
hand-labeling every task. They're used in §5.6 of the paper (Task-Family
Taxonomy) and can serve as a dispatcher that routes tasks to the appropriate
heuristic solver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loader import Task


@dataclass(frozen=True)
class TaskFeatures:
    task_id: str
    n_train: int
    n_test: int

    # shape signals
    same_shape_io: bool              # all train pairs: input shape == output shape
    output_larger: bool              # all: output.size > input.size
    output_smaller: bool             # all: output.size < input.size
    output_exact_scale_2x: bool      # all: output.shape == 2 * input.shape
    output_exact_scale_3x: bool
    output_exact_half: bool

    # color signals
    same_colors_io: bool             # all: set(in) == set(out)
    output_adds_colors: bool         # all: set(out) > set(in) (strict superset)
    output_removes_colors: bool      # all: set(in) > set(out)
    n_unique_input_colors: int       # max across train pairs
    n_unique_output_colors: int

    # grid size signals
    max_in_size: int
    max_out_size: int
    avg_in_size: float
    avg_out_size: float

    # object-count signals (4-connected components of nonzero cells)
    avg_in_components: float
    avg_out_components: float

    def family(self) -> str:
        """Coarse categorization — each task gets at most one label for the taxonomy."""
        if self.same_shape_io and self.same_colors_io:
            return "pixel_transform"  # shape + palette preserved
        if self.same_shape_io and not self.same_colors_io:
            return "color_transform"
        if self.output_exact_scale_2x or self.output_exact_scale_3x:
            return "tiling_or_scaling"
        if self.output_smaller:
            return "cropping_or_selection"
        if self.output_larger:
            return "expansion_or_generation"
        return "other"


def _count_components(g: np.ndarray, ignore: int = 0) -> int:
    try:
        from scipy.ndimage import label
        mask = g != ignore
        _, n = label(mask)
        return n
    except ImportError:
        return 0


def compute_features(task: Task) -> TaskFeatures:
    train = [p for p in task.train if p.output is not None]
    if not train:
        raise ValueError(f"task {task.task_id} has no usable train pairs")

    # shape
    same_shape = all(p.input.shape == p.output.shape for p in train)
    all_larger = all(p.output.size > p.input.size for p in train)
    all_smaller = all(p.output.size < p.input.size for p in train)
    scale_2x = all(
        p.output.shape[0] == 2 * p.input.shape[0]
        and p.output.shape[1] == 2 * p.input.shape[1]
        for p in train
    )
    scale_3x = all(
        p.output.shape[0] == 3 * p.input.shape[0]
        and p.output.shape[1] == 3 * p.input.shape[1]
        for p in train
    )
    exact_half = all(
        p.input.shape[0] == 2 * p.output.shape[0]
        and p.input.shape[1] == 2 * p.output.shape[1]
        for p in train
    )

    # color
    in_sets = [set(p.input.flatten().tolist()) for p in train]
    out_sets = [set(p.output.flatten().tolist()) for p in train]
    same_colors = all(a == b for a, b in zip(in_sets, out_sets))
    adds_colors = all(b > a for a, b in zip(in_sets, out_sets)) and not same_colors
    removes_colors = all(a > b for a, b in zip(in_sets, out_sets)) and not same_colors

    max_in_colors = max(len(s) for s in in_sets)
    max_out_colors = max(len(s) for s in out_sets)

    max_in_size = max(p.input.size for p in train)
    max_out_size = max(p.output.size for p in train)
    avg_in_size = sum(p.input.size for p in train) / len(train)
    avg_out_size = sum(p.output.size for p in train) / len(train)

    avg_in_comp = sum(_count_components(p.input) for p in train) / len(train)
    avg_out_comp = sum(_count_components(p.output) for p in train) / len(train)

    return TaskFeatures(
        task_id=task.task_id,
        n_train=len(task.train),
        n_test=len(task.test),
        same_shape_io=same_shape,
        output_larger=all_larger,
        output_smaller=all_smaller,
        output_exact_scale_2x=scale_2x,
        output_exact_scale_3x=scale_3x,
        output_exact_half=exact_half,
        same_colors_io=same_colors,
        output_adds_colors=adds_colors,
        output_removes_colors=removes_colors,
        n_unique_input_colors=max_in_colors,
        n_unique_output_colors=max_out_colors,
        max_in_size=max_in_size,
        max_out_size=max_out_size,
        avg_in_size=avg_in_size,
        avg_out_size=avg_out_size,
        avg_in_components=avg_in_comp,
        avg_out_components=avg_out_comp,
    )


__all__ = ["TaskFeatures", "compute_features"]
