# Reproducing this paper's numbers

Single source of truth for Paper Track reproducibility. Every number in the paper (our Kaggle writeup for the ARC Prize 2026 Paper Track) maps to a data file, a command or a Kaggle kernel below. Section labels follow the 2026-10-20 revision of the paper.

## Environment

```bash
# Local work: any machine with Python 3.11+; leaderboard runs need Kaggle 4 x L4
git clone https://github.com/damsolanke/arc-prize-2026
cd arc-prize-2026
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install scipy      # paper/make_figures.py
# rsvg-convert (librsvg) renders the cover PNG; without it only the SVG is written
```

## Data files behind the paper

| File | Contents | Source |
|---|---|---|
| `paper/evidence/leaderboard_runs.csv` | All 29 ARC-AGI-2 submissions: reference, UTC time, kernel slug, `scriptVersionId`, system label, SFT data, public score, dev reading | Kaggle `list_submissions` (`python scripts/sdk_kaggle.py subs` prints the latest 12), read 2026-10-20 15:15 UTC; dev readings from `paper/evidence/dev_runs.csv` and, for kernel v4, our notes on its June dev run |
| `paper/evidence/dev_runs.csv` | One row per logged 16-task dev run: solved test outputs, seconds per task, tasks that hit the 1,200 s cap, workers that report the adapter merge | `python scripts/sdk_kaggle.py log <slug>` (latest session of each kernel), fetched 2026-10-03 |
| `paper/evidence/public_lb_2026-10-03_scores.csv` | Rank, score and submission count for all 2,377 teams, names removed | Kaggle `download_leaderboard`, 2026-10-03 14:04 UTC |

## Sections 3 to 5: statistics and the cover image

```bash
python paper/make_figures.py
```

It prints every derived number under the marker that tags it in the paper source (U1 to U7, U10 to U12, U15) and writes `paper/figures/cover.svg` and `cover.png` (3200 x 1800). The other markers come from outside the script: U8 (rank and field) and U9 (private scores) from the final leaderboard, U13 (submission ID) and U14 (licence line) from the citation and licence decisions. On the 2026-10-20 data it prints, among others:

| Marker | Quantity | Value |
|---|---|---|
| U1 | Runs per system, June run first | base 10 (28.06, then nine October reruns); SFT 11 (29.86, then ten October reruns) |
| U2 | Base system (`arc-agi-2-nvarc-4b-l4x4`, scriptVersionId 327994206): n, mean ± SD, range | 10, 29.34 ± 1.21, 27.64 to 31.39 |
| U2 | SFT system (`arc-agi-2-sft-infer-v3`, scriptVersionId 328952350): n, mean ± SD, range | 11, 29.86 ± 1.46, 27.78 to 31.39 |
| U3 | Pooled run-to-run SD (pooled-variance estimate, 19 df) | 1.35 points = 1.6 tasks |
| U4 | SFT minus base, pooled-t 95% CI (Welch interval also printed) | +0.53, -0.71 to +1.76 (p 0.38; Welch -0.70 to +1.75) |
| U5 | June sweep z = delta / (SD x sqrt 2), delta against the base June run 28.06 | +0.14, -0.87, +0.94, -0.44, +0.43 |
| U6 | Minimum detectable difference, 80% power, two-sided 5%, (1.96 + 0.84) x SD x sqrt(2/n) | 5.3, 3.1, 1.9, 1.4 points (6.4, 3.7, 2.3, 1.7 tasks) at n = 1, 3, 8, 15 |
| U7 | Runs per system for a one-task (0.83-point) gain: 2 x (2.8 x SD / 0.833)^2, rounded up; the same for a two-task gain and at the SD of one-submission teams (1.25); days at one submission per day; round length 2026-03-25 to 2026-11-02 | 42; 11; 36; 84 days; 222 days |
| U10 | Expected best of 60 standard normal draws (numerical integration), times SD 1.25 and 1.35; observed gap between the 60-plus and one-submission bins | 2.32 SD = 2.9 to 3.1 points; 3.07 |
| U11 | Largest range between runs of one kernel, in points and tasks; largest June sweep delta as a share of it | 3.75 = 4.5 tasks; 1.80 = 0.48 |
| U12 | June run against the October runs of the same kernel, in October SD units; October-only pooled SD | base: October n 9, mean 29.48, SD 1.19, June 28.06 at -1.19 SD; SFT: October n 10, mean 29.86, SD 1.54, June 29.86 at 0.00 SD; pooled 1.39 (17 df) |
| U15 | Rank span of our best score and of the June best in the 2026-10-03 leaderboard export | 31.39: 256 to 376; 29.86: 833 to 853 |

One task is 100/120 = 0.83 points: all 2,377 public scores are multiples of 100/720 (none is an odd multiple of 100/1440), which fits 120 public tasks whose multi-output tasks earn partial credit. The script also computes the other leaderboard numbers in §5 from `public_lb_2026-10-03_scores.csv`: 683 teams from 30 to under 32, and the best-score-by-submission-count table for teams scoring 26 to 35 (233 one-submission teams at 29.23, SD 1.25; 58 teams with 60 or more at 32.29). Ties are ordered by submission time, so 31.39 is a 121-way tie spanning ranks 256 to 376.

## Kernels and submissions

Shared settings, from each `kernel-metadata.json`: competition source `arc-prize-2026-arc-agi-2`; model `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`; kernel source `sorokin/pip-install-unsloth-flash-patch`; internet off; docker image `gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868`; machine `NvidiaL4` (4 x L4) except the T4 run. Every directory sits under `kaggle_kernel/`; the T4 kernel `arc_agi2_nvarc` is not part of this repository.

| Configuration | SFT kernel (dir) | SFT flags | Inference kernel (dir) | scriptVersionId | Submissions (public score) | Dev, of 16 |
|---|---|---|---|---|---|---:|
| T4 settings | none | none | `arc_agi2_nvarc` | 327950122 | 53780259 (10.42) | n/a |
| Base | none | none | `arc_agi2_nvarc_l4x4` (v4) | 327994206 | 10 runs: 53790200 (28.06) in June, nine October reruns 27.64 to 31.39 (best 56784515) | 2.33 |
| train | `arc_agi2_sft` | `SFT_AUG=1` | `arc_agi2_sft_infer` | 328557646 | 53841875 (28.33) | 2.33 |
| train, 3 colour permutations | `arc_agi2_sft_v2` | `SFT_AUG=3` | `arc_agi2_sft_infer_v2` | 328688887 | 53863350 (26.39) | 2.33 |
| train + eval (submitted) | `arc_agi2_sft_v3` | `SFT_AUG=1`, eval appended | `arc_agi2_sft_infer_v3` (v1) | 328952350 | 11 runs: 53895939 (29.86) in June, ten October reruns 27.78 to 31.39 (best 56925138 and 57413553) | 2.33 |
| train + eval + ARC-GEN | `arc_agi2_sft_syn` | `GEN_TASKS=400 GEN_PER=3` | `arc_agi2_sft_infer_syn` | 329309802 | 53927991 (27.22) | 3.33 |
| train + eval x3 | `arc_agi2_sft_evup` | `EVAL_REPEAT=3` | `arc_agi2_sft_infer_evup` | 329539348 | 53960956 (28.89) | 4.33 |

All SFT runs use `SFT_EPOCHS=1 SFT_GA=8`, LoRA r 256, alpha 32, rsLoRA, lr 5e-5 cosine, bf16. Code checks behind §2:

- `arc_agi2_nvarc_l4x4` differs from the public notebook `kaiserm/nvarc-winning-solution-2025-4xl4s` in three places only: `_find_model_dir()`, a guard in the dev-mode benchmark printout, and the 16-task dev sample. In rerun mode it computes the same thing.
- The five `arc_agi2_sft_infer*` notebooks have identical code cells and add one 16-line adapter-merge block to `arc_agi2_nvarc_l4x4`. Each dev log has four "[SFT-DEPLOY] merged adapter" lines, one per worker.
- `arc_agi2_sft_v3`, `_syn` and `_evup` append the 120 evaluation tasks with solutions to the training data (`_evup` three times); `arc_agi2_sft` and `_v2` use the 1,000 training tasks only.

Procedure for one row (each `kernel-metadata.json` names the owner `YOUR_KAGGLE_USERNAME`, which `sdk_kaggle.py` replaces with the account in `~/.kaggle/kaggle.json`):

```bash
python scripts/sdk_kaggle.py push kaggle_kernel/arc_agi2_sft_v3        # SFT, 2.5 h for 1,000 tasks
python scripts/sdk_kaggle.py push kaggle_kernel/arc_agi2_sft_infer_v3  # dev mode: first 16 eval tasks, prints "acc: X/16"
python scripts/sdk_kaggle.py log arc-agi-2-sft-infer-v3
python scripts/sdk_kaggle.py submit arc-agi-2-sft-infer-v3 1 "<description>"   # one per UTC day
python scripts/sdk_kaggle.py subs
```

The dev reading for base kernel v4 (2.33) comes from our notes on its June dev run; the latest logged session of that kernel is v5 (DFS cutoff 0.15, 784 s per task, not submitted), which solved the same three outputs.

## Section 3: noise sources in the code

| Claim | Where |
|---|---|
| Scoring views seeded with `hash(bk) % 1024**2`, `bk` a string | `arc_solver.py` as written by the `%%writefile arc_solver.py` cell of `arc_agi2_nvarc_l4x4.ipynb`: `bk = subkey.split(".")[0]` and `aug_dataset.augment(seed=hash(bk) % 1024**2)`; no notebook sets `PYTHONHASHSEED`. The same line is in five of the six 2026 public notebooks we read (all but luxluxshan, which uses `zlib.crc32` and `PYTHONHASHSEED=0`) |
| 8 scoring views, 16 decoding views, 128 TTT copies | `augment()` builds 8 geometric views (transpose x 4 rotations) times n colour permutations: n=1 for scoring, n=2 for decoding, n=16 for TTT |
| 1,200 s per task, 540 s per DFS batch, 12 h minus 600 s overall | `spend_time > 1200`, `time.time() - start_time < 540`, `global_end_time = time.time() + 12 * 3600 - 600` |
| Two or three dev tasks hit the 1,200 s cap in every logged dev run | `paper/evidence/dev_runs.csv`, column `tasks_hitting_1200s_cap` |
| Ranking by decode count minus mean NLL | `getter_kgmon` in `arc_decoder.py`; `calc_scores` returns the negative summed log-probability |

## Section 4: contamination evidence

| Claim | Where |
|---|---|
| Summaries of all 120 evaluation tasks (29 by hand, 91 by LLM) fed the synthetic pipeline | NVARC paper, section 2.1 (`github.com/1ytic/NVARC/blob/main/nvarc_2025.pdf`) |
| "NVARC full" uses training and evaluation puzzles; 1,785,960 of 3,255,481 samples (54.9%) | NVARC paper, Table 1 |
| NVARC calls the evaluation descriptions "leaked"; their validation loss still correlated with the public board | NVARC paper, section 3.5 |
| The base checkpoint was trained on that mix | NVARC repo `ARChitects/run_sft_4b.sh` (`EXP_NAME="qwen3_4b_grids15_sft139"`, 12,716 steps x global batch 256 = 3,255,296 samples) and `ARChitects/sft_mg.yaml` (`train_dataset_path` includes `data/grids_v15/nvarc_full`) |
| 103k synthetic puzzles with about 30 input-output pairs each, public since November 2025 | NVARC paper, section 2.4; Kaggle dataset `sorokin/nvarc-synthetic-puzzles` (5.79 GB, version 1 dated 2025-11-13) |
| Dev readings and the solved outputs behind them | `paper/evidence/dev_runs.csv` |

## Heuristic submissions (0.00)

Three heuristic-ensemble submissions (16 to 23 April 2026, kernel `kaggle_kernel/arc_prize_2026_baseline.py`) scored 0.00. Local heuristic results: `python scripts/run_baseline.py` (about 80 s of solver time; writes `experiments/results/baseline_<timestamp>.json`). Task taxonomy: `python scripts/compute_taxonomy.py`.

## Determinism

The heuristic experiments are deterministic. The Kaggle kernels fix the numpy seeds for TTT (1) and decoding views (2) and the trainer seed (42), but the scoring views take their seed from Python's per-process salted `hash`, and the GPU kernels and wall-clock caps add further run-to-run variation (paper §3). Leaderboard rows other than the two resubmitted kernels have one run each.

## Contact

Dam Solanke
GitHub: [damsolanke](https://github.com/damsolanke)
Paper Track submission: Kaggle competition `arc-prize-2026-paper-track`
