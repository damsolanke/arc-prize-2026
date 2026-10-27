"""Produce submission.json from a solver over the Kaggle test_challenges set.

Output format (official sample_submission.json):
    {
      "task_id": [
        {"attempt_1": [[rows...]], "attempt_2": [[rows...]]},
        ... one per test input
      ],
      ...
    }

Usage:
    python scripts/make_submission.py \
        --solver composition \
        --out submission.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.eval import load_kaggle_test, load_kaggle_evaluation  # noqa: E402
from src.solvers import (  # noqa: E402
    ColorMapSolver,
    CompositionSolver,
    D8Solver,
    EnsembleSolver,
    TransformSolver,
    VerifiedEnsembleSolver,
    default_ensemble_inner,
)

SOLVERS = {
    "d8": D8Solver,
    "color_map": ColorMapSolver,
    "transform": TransformSolver,
    "composition": CompositionSolver,
    "ensemble_plain": lambda: EnsembleSolver(
        inner=default_ensemble_inner() + [TransformSolver(), CompositionSolver()],
        name="ensemble_plain",
    ),
    "ensemble_verified": lambda: VerifiedEnsembleSolver(
        inner=default_ensemble_inner() + [TransformSolver(), CompositionSolver()],
        name="ensemble_verified",
    ),
}


def grid_to_list(g: np.ndarray) -> list[list[int]]:
    return g.astype(int).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="composition", choices=sorted(SOLVERS))
    ap.add_argument("--split", default="test", choices=["test", "evaluation"])
    ap.add_argument("--out", default="submission.json", type=Path)
    args = ap.parse_args()

    if args.split == "test":
        tasks = load_kaggle_test()
    else:
        tasks = load_kaggle_evaluation()
    print(f"loaded {len(tasks)} {args.split} tasks")

    solver = SOLVERS[args.solver]()
    print(f"solver: {solver.name}")

    submission: dict[str, list[dict]] = {}
    t0 = time.perf_counter()
    for t in tasks:
        pred = solver.solve(t)
        entries: list[dict] = []
        for i, cands in enumerate(pred.candidates):
            attempt_1 = grid_to_list(cands[0]) if len(cands) >= 1 else [[0, 0], [0, 0]]
            attempt_2 = grid_to_list(cands[1]) if len(cands) >= 2 else attempt_1
            entries.append({"attempt_1": attempt_1, "attempt_2": attempt_2})
        submission[t.task_id] = entries
    wall = time.perf_counter() - t0

    args.out.write_text(json.dumps(submission))
    print(f"wrote {args.out} ({args.out.stat().st_size/1024:.1f} KB)")
    print(f"wall: {wall:.2f}s ({wall/len(tasks)*1000:.1f}ms/task)")
    print(f"tasks covered: {len(submission)} / {len(tasks)}")


if __name__ == "__main__":
    main()
