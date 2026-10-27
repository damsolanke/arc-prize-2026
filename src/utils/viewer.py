"""Render ARC tasks as matplotlib figures. Used for paper figures + debugging."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from ..eval.loader import Grid, Task

ARC_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
]
ARC_CMAP = mcolors.ListedColormap(ARC_COLORS)
ARC_NORM = mcolors.BoundaryNorm(boundaries=np.arange(11) - 0.5, ncolors=10)


def render_grid(ax, grid: Grid, title: str = "") -> None:
    ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#555555", linestyle="-", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def render_task(task: Task, predictions: Sequence[Grid] | None = None):
    n_train = len(task.train)
    n_test = len(task.test)
    n_pred = 3 if predictions else 0
    cols = max(n_train, n_test + n_pred)
    fig, axes = plt.subplots(2 + (1 if predictions else 0), cols,
                              figsize=(cols * 1.6, 4 if predictions else 3))
    axes = np.atleast_2d(axes)

    for i, pair in enumerate(task.train):
        render_grid(axes[0, i], pair.input, f"train_in_{i}")
        render_grid(axes[1, i], pair.output, f"train_out_{i}")
    for j in range(len(task.train), cols):
        for r in range(2):
            axes[r, j].axis("off")

    for i, pair in enumerate(task.test):
        col = i
        render_grid(axes[0, col], pair.input, f"test_in_{i}")
        if pair.output is not None:
            render_grid(axes[1, col], pair.output, f"test_out_{i}")

    if predictions:
        for j, p in enumerate(predictions[:3]):
            render_grid(axes[2, j], p, f"pred_{j}")
        for j in range(len(predictions), cols):
            axes[2, j].axis("off")

    fig.suptitle(f"task {task.task_id}", fontsize=11)
    fig.tight_layout()
    return fig


def save_task_figure(task: Task, out_path: Path | str, predictions: Sequence[Grid] | None = None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = render_task(task, predictions)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
