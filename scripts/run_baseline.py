"""Run all baselines + the diverse Explicit Ensemble on ARC-AGI-2 (120 + 1000 tasks).

Produces the floor row of the paper's ablation table.

Usage:
    python scripts/run_baseline.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.loader import EVAL_DIR, TRAIN_DIR, load_split  # noqa: E402
from src.eval.scorer import score_submission  # noqa: E402
from src.solvers import (  # noqa: E402
    ColorMapSolver,
    CompositionSolver,
    D8Solver,
    EnsembleSolver,
    IdentitySolver,
    ObjectIsolationSolver,
    SymmetrySolver,
    TilingSolver,
    TransformSolver,
    VerifiedEnsembleSolver,
    default_ensemble_inner,
)


def run_solver(solver, tasks, split_name: str) -> dict:
    preds: dict[str, list] = {}
    t0 = time.perf_counter()
    for t in tasks:
        preds[t.task_id] = solver.solve(t).candidates
    wall = time.perf_counter() - t0

    score = score_submission(tasks, preds)
    res = {
        "solver": solver.name,
        "split": split_name,
        "solved": score.solved,
        "total": score.total,
        "accuracy": round(score.accuracy, 4),
        "wall_s": round(wall, 3),
        "ms_per_task": round(wall / len(tasks) * 1000, 2),
    }
    print(
        f"  {solver.name:<40s} {score.solved:4d}/{score.total:<4d} "
        f"({score.accuracy*100:5.2f}%)   {wall:6.2f}s  ({res['ms_per_task']:5.1f}ms/task)"
    )
    return res


def main() -> None:
    solvers = [
        IdentitySolver(),
        D8Solver(),
        ColorMapSolver(),
        SymmetrySolver(),
        TilingSolver(),
        ObjectIsolationSolver(),
        TransformSolver(),
        CompositionSolver(),
        EnsembleSolver(inner=default_ensemble_inner(), name="ensemble_6_diverse_plain"),
        VerifiedEnsembleSolver(inner=default_ensemble_inner(), name="ensemble_6_diverse_verified"),
        EnsembleSolver(
            inner=default_ensemble_inner() + [TransformSolver(), CompositionSolver()],
            name="ensemble_8_with_transform_comp_plain",
        ),
        VerifiedEnsembleSolver(
            inner=default_ensemble_inner() + [TransformSolver(), CompositionSolver()],
            name="ensemble_8_with_transform_comp_verified",
        ),
    ]

    results = []
    for split_name, split_dir in [("eval", EVAL_DIR), ("train", TRAIN_DIR)]:
        tasks = list(load_split(split_dir))
        print(f"\n{'='*72}\n{split_name}: {len(tasks)} tasks\n{'='*72}")
        for solver in solvers:
            results.append(run_solver(solver, tasks, split_name))

    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"baseline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nresults -> {out_file}")


if __name__ == "__main__":
    main()
