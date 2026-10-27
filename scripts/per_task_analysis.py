"""Per-task breakdown: which solver solves each task, plus oracle upper bound.

Used for paper §5 figures:
  - Bar chart: solver coverage (unique tasks solved by each)
  - Oracle: union of all solvers' solved tasks = upper bound of an oracle selector
  - Task-family breakdown: which family benefits most from which solver

Writes:
  experiments/results/per_task_breakdown.json  — per (task_id, solver) solved flag
  experiments/results/oracle_upper_bound.json  — oracle solver union analysis
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.loader import EVAL_DIR, TRAIN_DIR, load_split  # noqa: E402
from src.eval.scorer import score_submission  # noqa: E402
from src.eval.task_features import compute_features  # noqa: E402
from src.solvers import (  # noqa: E402
    ColorMapSolver,
    CompositionSolver,
    D8Solver,
    IdentitySolver,
    ObjectIsolationSolver,
    SymmetrySolver,
    TilingSolver,
    TransformSolver,
)


def main() -> None:
    solvers = {
        "identity": IdentitySolver(),
        "d8": D8Solver(),
        "color_map": ColorMapSolver(),
        "symmetry": SymmetrySolver(),
        "tiling": TilingSolver(),
        "object": ObjectIsolationSolver(),
        "transform": TransformSolver(),
        "composition": CompositionSolver(),
    }

    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_task = defaultdict(dict)
    oracle = defaultdict(dict)

    for split_name, split_dir in [("eval", EVAL_DIR), ("train", TRAIN_DIR)]:
        tasks = list(load_split(split_dir))
        print(f"\n{'='*70}\n{split_name}: {len(tasks)} tasks\n{'='*70}")

        per_solver_solved: dict[str, set[str]] = {}
        for solver_name, solver in solvers.items():
            preds = {t.task_id: solver.solve(t).candidates for t in tasks}
            score = score_submission(tasks, preds)
            solved_ids = {r.task_id for r in score.per_task if r.solved}
            per_solver_solved[solver_name] = solved_ids
            for t in tasks:
                per_task[split_name].setdefault(t.task_id, {})[solver_name] = (
                    t.task_id in solved_ids
                )
            print(f"  {solver_name:<12s} solved {len(solved_ids):4d} / {len(tasks)}")

        # Union = oracle that picks best solver per task
        all_solved = set()
        for s in per_solver_solved.values():
            all_solved |= s
        print(f"  {'ORACLE (union)':<12s} solved {len(all_solved):4d} / {len(tasks)}")

        # Family breakdown: per family, per solver, how many tasks solved
        feats = {t.task_id: compute_features(t) for t in tasks}
        fam_solver: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        fam_counts: dict[str, int] = defaultdict(int)
        for t in tasks:
            fam = feats[t.task_id].family()
            fam_counts[fam] += 1
            for solver_name, solved_set in per_solver_solved.items():
                if t.task_id in solved_set:
                    fam_solver[fam][solver_name] += 1

        print(f"\n  by family × solver (solved / total):")
        print(f"  {'family':<28s} {'total':>6s}  " + "  ".join(
            f"{name:>10s}" for name in solvers
        ))
        for fam in sorted(fam_counts):
            row = [f"  {fam:<28s} {fam_counts[fam]:6d} "]
            for name in solvers:
                row.append(f"  {fam_solver[fam][name]:10d}")
            print("".join(row))

        oracle[split_name] = {
            "total_tasks": len(tasks),
            "per_solver_solved_counts": {n: len(s) for n, s in per_solver_solved.items()},
            "oracle_union_size": len(all_solved),
            "oracle_accuracy": len(all_solved) / len(tasks),
            "family_counts": dict(fam_counts),
            "family_x_solver_solved": {
                fam: dict(solvers_dict) for fam, solvers_dict in fam_solver.items()
            },
        }

    (out_dir / "per_task_breakdown.json").write_text(json.dumps(per_task, indent=2))
    (out_dir / "oracle_upper_bound.json").write_text(json.dumps(oracle, indent=2))
    print(f"\nwrote per-task breakdown to {out_dir}/")


if __name__ == "__main__":
    main()
