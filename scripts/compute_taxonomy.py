"""Compute taxonomy features for every ARC-AGI-2 eval + training task.

Writes:
  experiments/results/taxonomy_eval.json
  experiments/results/taxonomy_train.json
  experiments/results/taxonomy_summary.md

This is the per-task feature map underlying the E5 task-family ablation.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.loader import EVAL_DIR, TRAIN_DIR, load_split  # noqa: E402
from src.eval.task_features import compute_features  # noqa: E402


def main() -> None:
    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for split_name, split_dir in [("eval", EVAL_DIR), ("train", TRAIN_DIR)]:
        tasks = list(load_split(split_dir))
        feats = [compute_features(t) for t in tasks]
        out_file = out_dir / f"taxonomy_{split_name}.json"
        out_file.write_text(json.dumps([asdict(f) for f in feats], indent=2))

        fam_counter = Counter(f.family() for f in feats)
        same_shape = sum(1 for f in feats if f.same_shape_io)
        same_colors = sum(1 for f in feats if f.same_colors_io)
        larger = sum(1 for f in feats if f.output_larger)
        smaller = sum(1 for f in feats if f.output_smaller)
        scale2 = sum(1 for f in feats if f.output_exact_scale_2x)
        scale3 = sum(1 for f in feats if f.output_exact_scale_3x)
        exact_half = sum(1 for f in feats if f.output_exact_half)

        print(f"\n{split_name}: {len(feats)} tasks")
        print(f"  same_shape_io:  {same_shape:4d} ({same_shape/len(feats)*100:5.1f}%)")
        print(f"  same_colors_io: {same_colors:4d} ({same_colors/len(feats)*100:5.1f}%)")
        print(f"  output_larger:  {larger:4d} ({larger/len(feats)*100:5.1f}%)")
        print(f"  output_smaller: {smaller:4d} ({smaller/len(feats)*100:5.1f}%)")
        print(f"  scale_2x:       {scale2:4d}  ({scale2/len(feats)*100:5.1f}%)")
        print(f"  scale_3x:       {scale3:4d}  ({scale3/len(feats)*100:5.1f}%)")
        print(f"  exact_half:     {exact_half:4d}  ({exact_half/len(feats)*100:5.1f}%)")
        print("  family distribution:")
        for fam, c in fam_counter.most_common():
            print(f"    {fam:<28s} {c:4d}  ({c/len(feats)*100:5.1f}%)")

        summary_rows.append({
            "split": split_name,
            "total": len(feats),
            "same_shape_io": same_shape,
            "same_colors_io": same_colors,
            "output_larger": larger,
            "output_smaller": smaller,
            "scale_2x": scale2,
            "scale_3x": scale3,
            "family_distribution": dict(fam_counter),
        })

    summary_file = out_dir / "taxonomy_summary.json"
    summary_file.write_text(json.dumps(summary_rows, indent=2))
    print(f"\nwrote taxonomy files to {out_dir}")


if __name__ == "__main__":
    main()
