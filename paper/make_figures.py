"""Rerun statistics and the cover image for paper/main.md.

Reads paper/evidence/leaderboard_runs.csv (one row per scored ARC-AGI-2 submission) and
paper/evidence/public_lb_2026-10-03_scores.csv (score and submission count per team, no names),
prints every number the paper takes from them, keyed by the <!--Un--> markers in main.md,
and writes paper/figures/cover.svg plus a 2x PNG through rsvg-convert.

Usage: python paper/make_figures.py
"""
import csv
import math
import shutil
import statistics as st
import subprocess
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from scipy import integrate, stats

HERE = Path(__file__).parent
RUNS = HERE / "evidence" / "leaderboard_runs.csv"
LB = HERE / "evidence" / "public_lb_2026-10-03_scores.csv"
OUT = HERE / "figures"
TASK = 100 / 120  # one fully solved task on the 120-task public split, in points
Z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)  # two-sided 5%, 80% power

ROUND_DAYS = (date(2026, 11, 2) - date(2026, 3, 25)).days  # ARC-AGI-2 start to final submission deadline
BEST_OF = 60  # submission count of the heaviest bin in the best-of-N table


def r2(x):
    """Round half up to 2 decimals, so 30.485 prints as 30.49 rather than float's 30.48."""
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def expected_max(n):
    """Mean of the largest of n independent standard normal draws."""
    f = lambda x: x * n * stats.norm.pdf(x) * stats.norm.cdf(x) ** (n - 1)
    return integrate.quad(f, -10, 10)[0]


SWEEP_LABELS = {"sft_train": "train", "sft_aug3": "3 colour perms", "sft_syn": "+ARC-GEN", "sft_evup": "eval x3"}


def load_runs():
    rows = list(csv.DictReader(RUNS.open()))
    for r in rows:
        r["score"] = float(r["public_score"])
    return rows


def summarize(rows):
    out = {}
    for sysname in ("base", "sft"):
        xs = [r for r in rows if r["system"] == sysname]
        xs.sort(key=lambda r: r["submitted_utc"])
        sc = [r["score"] for r in xs]
        out[sysname] = dict(runs=xs, scores=sc, n=len(sc), mean=st.mean(sc), sd=st.stdev(sc) if len(sc) > 1 else float("nan"),
                            lo=min(sc), hi=max(sc))
    b, s = out["base"], out["sft"]
    df_p = (b["n"] - 1) + (s["n"] - 1)
    pooled = math.sqrt(((b["n"] - 1) * b["sd"] ** 2 + (s["n"] - 1) * s["sd"] ** 2) / df_p)
    diff = s["mean"] - b["mean"]
    # both systems share one pipeline and its noise sources, so the paper uses the pooled-variance t interval
    se_p = pooled * math.sqrt(1 / b["n"] + 1 / s["n"])
    half = stats.t.ppf(0.975, df_p) * se_p
    p = 2 * stats.t.sf(abs(diff) / se_p, df_p)
    se_w = math.sqrt(b["sd"] ** 2 / b["n"] + s["sd"] ** 2 / s["n"])
    df_w = se_w ** 4 / ((b["sd"] ** 2 / b["n"]) ** 2 / (b["n"] - 1) + (s["sd"] ** 2 / s["n"]) ** 2 / (s["n"] - 1))
    half_w = stats.t.ppf(0.975, df_w) * se_w
    out["pooled_sd"], out["pooled_df"] = pooled, df_p
    out["diff"], out["diff_ci"], out["diff_p"] = diff, (diff - half, diff + half), p
    out["welch_ci"], out["welch_df"] = (diff - half_w, diff + half_w), df_w
    out["mdd"] = {n: Z * pooled * math.sqrt(2 / n) for n in (1, 3, 8, 15)}
    out["n_for"] = {k: math.ceil(2 * (Z * pooled / (k * TASK)) ** 2) for k in (1, 2)}
    june_base = [r for r in b["runs"] if r["submitted_utc"] < "2026-07"][0]["score"]
    sweep = []
    for r in rows:
        if r["system"] in SWEEP_LABELS or (r["system"] == "sft" and r["submitted_utc"] < "2026-07"):
            d = r["score"] - june_base
            sweep.append(dict(label=SWEEP_LABELS.get(r["system"], "train + eval"), score=r["score"], delta=d,
                              z=d / (pooled * math.sqrt(2)), probe=r["probe_of_16"], system=r["system"]))
    out["sweep"], out["june_base"] = sweep, june_base
    return out


def leaderboard():
    rows = [(float(r["score"]), int(r["submission_count"])) for r in csv.DictReader(LB.open())]
    band = [(s, n) for s, n in rows if 26 <= s <= 35]
    table = []
    for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 9), (10, 24), (25, 59), (60, 10 ** 9)):
        xs = [s for s, n in band if lo <= n <= hi]
        table.append((lo, hi, len(xs), st.mean(xs), st.stdev(xs)))
    scores = [s for s, _ in rows]

    def rank_span(x):  # ties are ordered by submission time, so a score maps to a span of ranks
        above = sum(v > x for v in scores)
        return above + 1, above + sum(v == x for v in scores)

    return dict(teams=len(rows), table=table, scores=scores, rank_span=rank_span,
                band_30_32=sum(30 <= v < 32 for v in scores), one_sub_sd=table[0][4],
                gap=table[-1][3] - table[0][3])


def report(s, lb):
    b, f = s["base"], s["sft"]
    fmt = lambda xs: ", ".join(f"{x:.2f}" for x in xs)
    print("Values for paper/main.md, keyed by its <!--Un--> markers")
    print(f"U1  abstract and Limitations: base runs n={b['n']} ({fmt(b['scores'])}); SFT runs n={f['n']} ({fmt(f['scores'])})")
    for name, d in (("base", b), ("SFT ", f)):
        print(f"U2  Table 1 {name}: n={d['n']}, mean +- SD {r2(d['mean'])} +- {r2(d['sd'])}, range {d['lo']:.2f} to {d['hi']:.2f}")
    print(f"U3  pooled run SD: {s['pooled_sd']:.2f} points ({s['pooled_sd'] / TASK:.1f} tasks), df={s['pooled_df']}")
    lo, hi = s["diff_ci"]
    wl, wh = s["welch_ci"]
    print(f"U4  SFT minus base: {s['diff']:+.2f}, 95% CI {lo:+.2f} to {hi:+.2f} (pooled t, df {s['pooled_df']}, p={s['diff_p']:.2f}); "
          f"Welch {wl:+.2f} to {wh:+.2f} (df {s['welch_df']:.1f})")
    print(f"U5  Table 2, z = delta / (pooled SD x sqrt 2), delta against the base June run {s['june_base']:.2f}:")
    for w in s["sweep"]:
        print(f"      {w['label']:<16} {w['score']:.2f}  delta {w['delta']:+.2f}  z {w['z']:+.2f}  dev {w['probe']}")
    print("U6  minimum detectable difference (80% power, two-sided 5%), points / tasks:")
    for n, m in s["mdd"].items():
        print(f"      n={n:>2} per system: {m:.1f} / {m / TASK:.1f}")
    n1_low = math.ceil(2 * (Z * lb["one_sub_sd"] / TASK) ** 2)
    print(f"U7  runs per system for a one-task gain: {s['n_for'][1]} (days at one per day: {2 * s['n_for'][1]}, round {ROUND_DAYS} days); "
          f"two-task gain: {s['n_for'][2]}; at the one-submission SD {lb['one_sub_sd']:.2f}: {n1_low}")
    em = expected_max(BEST_OF)
    print(f"U10 best of {BEST_OF} runs minus the mean: {em:.2f} SD = {em * lb['one_sub_sd']:.1f} to {em * s['pooled_sd']:.1f} points "
          f"(SD {lb['one_sub_sd']:.2f} to {s['pooled_sd']:.2f}); observed gap {lb['table'][-1][3]:.2f} - {lb['table'][0][3]:.2f} = {lb['gap']:.2f}")
    big = max(abs(w["delta"]) for w in s["sweep"])
    rng = max(b["hi"] - b["lo"], f["hi"] - f["lo"])
    print(f"U11 largest same-kernel range: {rng:.2f} points = {rng / TASK:.1f} tasks (subtitle); largest |sweep delta| {big:.2f} "
          f"= {big / rng:.2f} of it")
    # drift check: June runs against October runs of the same kernel
    for key in ("base", "sft"):
        june = [r["score"] for r in s[key]["runs"] if r["submitted_utc"] < "2026-07"]
        octo = [r["score"] for r in s[key]["runs"] if r["submitted_utc"] >= "2026-10"]
        if len(octo) >= 3:
            m, sd = st.mean(octo), st.stdev(octo)
            print(f"U12 {key}: October n={len(octo)} mean {r2(m)} SD {sd:.2f}; June {june} sits {(june[0] - m) / sd:+.2f} SD from it")
        else:
            print(f"U12 {key}: October n={len(octo)}, too few for a drift check; June {june}, October {octo}")
    octs = [[r["score"] for r in s[k]["runs"] if r["submitted_utc"] >= "2026-10"] for k in ("base", "sft")]
    if all(len(o) >= 2 for o in octs):
        dfo = sum(len(o) - 1 for o in octs)
        print(f"U12 October-only pooled SD: {math.sqrt(sum((len(o) - 1) * st.variance(o) for o in octs) / dfo):.2f} (df {dfo})")
    best = max(b["scores"] + f["scores"])
    first_june_best = max(r["score"] for r in b["runs"] + f["runs"] if r["submitted_utc"] < "2026-07")
    print(f"U15 our best public score {best:.2f}: ranks {lb['rank_span'](best)} of {lb['teams']} in this export; "
          f"the June best {first_june_best:.2f} would rank {lb['rank_span'](first_june_best)}")
    print(f"LB  teams={lb['teams']}; teams from 30 to under 32: {lb['band_30_32']}; best score by submission count, teams at 26 to 35:")
    for lo_, hi_, n, m, sd in lb["table"]:
        print(f"      {lo_}-{hi_ if hi_ < 10 ** 9 else 'up'}: {n} teams, mean {m:.2f}, SD {sd:.2f}")


# ---------------------------------------------------------------- cover image
FONT = "Geist, 'Helvetica Neue', 'Inter', Arial, sans-serif"
INK, BODY, MUTED, FAINT, GRID = "#0F172A", "#334155", "#64748B", "#94A3B8", "#E2E8F0"
BASE_C, SFT_C, SWEEP_C, ACCENT = "#2A78D6", "#EB6834", "#8A94A6", "#4F46E5"
W, H = 1600, 900


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def cover(s):
    parts = []

    def text(x, y, t, size=15, fill=INK, weight=500, anchor="start", spacing=None):
        ls = f' letter-spacing="{spacing}"' if spacing is not None else ""
        parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
                     f'font-weight="{weight}" text-anchor="{anchor}"{ls}>{esc(t)}</text>')

    b, f = s["base"], s["sft"]
    allsc = b["scores"] + f["scores"] + [w["score"] for w in s["sweep"]]
    band_lo, band_hi = b["mean"] - 2 * s["pooled_sd"], b["mean"] + 2 * s["pooled_sd"]
    xmin = math.floor(min(min(allsc), band_lo) - 0.4)
    xmax = math.ceil(max(max(allsc), band_hi) + 0.4)
    x0, x1 = 470, 1520
    sx = lambda v: x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)
    rows = [("base", 318), ("sft", 438), ("sweep", 558)]
    top, bottom = 262, 612

    parts.append(f'<rect width="{W}" height="{H}" fill="#FBFCFE"/>')
    text(80, 84, "ARC PRIZE 2026  ·  PAPER TRACK  ·  ARC-AGI-2", size=15, fill=ACCENT, weight=700, spacing=3)
    text(78, 146, "One score per day", size=54, weight=800, spacing=-0.8)
    text(80, 186, "Identical reruns of two NVARC-style systems on the ARC-AGI-2 public leaderboard (120 tasks)",
         size=21, fill=MUTED)

    # noise band: base mean +- 2 pooled SD, where most single runs of the unchanged base system should land
    parts.append(f'<rect x="{sx(band_lo):.1f}" y="{top}" width="{sx(band_hi) - sx(band_lo):.1f}" height="{bottom - top}" '
                 f'rx="10" fill="{BASE_C}" fill-opacity="0.08"/>')
    text((sx(band_lo) + sx(band_hi)) / 2, top - 14,
         f"One run of the unchanged base system: mean ± 2 SD ({band_lo:.1f} to {band_hi:.1f}), SD pooled over {b['n'] + f['n']} runs",
         size=14, fill=MUTED, weight=600, anchor="middle")

    for v in range(xmin, xmax + 1):
        parts.append(f'<line x1="{sx(v):.1f}" y1="{top}" x2="{sx(v):.1f}" y2="{bottom}" stroke="{GRID}" stroke-width="1"/>')
        text(sx(v), bottom + 30, f"{v}", size=15, fill=MUTED, anchor="middle")
    parts.append(f'<line x1="{x0}" y1="{bottom}" x2="{x1}" y2="{bottom}" stroke="#CBD5E1" stroke-width="1"/>')
    text((x0 + x1) / 2, bottom + 60, "Public leaderboard score (%)", size=15, fill=MUTED, weight=600, anchor="middle")
    # one-task scale bar, right of the axis title
    tw = sx(xmin + TASK) - sx(xmin)
    bx = x1 - tw
    parts.append(f'<path d="M{bx:.1f} {bottom + 49} v6 H{x1} v-6" fill="none" stroke="{MUTED}" stroke-width="1.5"/>')
    text(bx - 10, bottom + 60, "one task", size=13, fill=MUTED, weight=600, anchor="end")

    labels = {"base": ("Base system", f"NVARC Qwen3-4B + per-task TTT, kernel v4 (n = {b['n']})"),
              "sft": ("SFT system", f"base + LoRA SFT on train + eval, kernel v1 (n = {f['n']})"),
              "sweep": ("June sweep", "four other adapters, one run each")}
    for key, y in rows:
        head, sub = labels[key]
        text(80, y - 4, head, size=20, weight=800)
        text(80, y + 20, sub, size=14, fill=MUTED)

    def dots(scores, y, color, hollow=False):
        seen = {}
        for v in sorted(scores):
            k = round(v, 2)
            off = seen.get(k, 0)
            seen[k] = off + 1
            yy = y + (0, -15, 15, -30, 30, -45, 45)[min(off, 6)]  # equal scores stack alternately up and down
            fill = "#FBFCFE" if hollow else color
            parts.append(f'<circle cx="{sx(v):.1f}" cy="{yy:.1f}" r="9" fill="{fill}" stroke="#FFFFFF" stroke-width="2.5"/>')
            if hollow:
                parts.append(f'<circle cx="{sx(v):.1f}" cy="{yy:.1f}" r="7.5" fill="none" stroke="{color}" stroke-width="2.5"/>')

    for key, y in rows[:2]:
        d, color = s[key], (BASE_C if key == "base" else SFT_C)
        parts.append(f'<line x1="{sx(d["mean"]):.1f}" y1="{y - 26}" x2="{sx(d["mean"]):.1f}" y2="{y + 26}" stroke="{INK}" stroke-width="2"/>')
        text(sx(d["mean"]), y + 46, f"mean {r2(d['mean'])}", size=13, fill=BODY, weight=600, anchor="middle")
        dots(d["scores"], y, color)
        # bracket over the spread of identical runs
        lo, hi = sx(d["lo"]), sx(d["hi"])
        parts.append(f'<path d="M{lo:.1f} {y - 22} v-8 H{hi:.1f} v8" fill="none" stroke="{FAINT}" stroke-width="1.5"/>')
        text((lo + hi) / 2, y - 38, f"{d['lo']:.2f} to {d['hi']:.2f}: {(d['hi'] - d['lo']) / TASK:.1f} tasks apart, same code",
             size=13, fill=BODY, weight=600, anchor="middle")

    ysw = rows[2][1]
    dots([w["score"] for w in s["sweep"] if w["system"] != "sft"], ysw, SWEEP_C, hollow=True)
    for i, w in enumerate(sorted((w for w in s["sweep"] if w["system"] != "sft"), key=lambda w: w["score"])):
        text(sx(w["score"]), ysw + (-22 if i % 2 == 0 else 34), w["label"], size=13, fill=BODY, weight=600, anchor="middle")

    # stat tiles
    big = max(abs(w["delta"]) for w in s["sweep"])
    rng = max(b["hi"] - b["lo"], f["hi"] - f["lo"])
    tiles = [(f"{rng:.2f}", "points from worst to best run, same code"),
             (f"{big:.2f}", "points: largest effect in our June sweep"),
             (f"{s['n_for'][1]}", "runs per arm to detect a one-task gain")]
    for i, (num, line) in enumerate(tiles):
        x = 80 + i * 370
        parts.append(f'<rect x="{x}" y="704" width="350" height="124" rx="14" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.5"/>')
        text(x + 24, 768, num, size=46, weight=800)
        text(x + 24, 804, line, size=14.5, fill=BODY, weight=600)
    text(1200, 748, f"Run-to-run SD {s['pooled_sd']:.1f} points,", size=16, fill=BODY, weight=600)
    text(1200, 772, f"pooled over {b['n'] + f['n']} identical runs.", size=16, fill=BODY, weight=600)
    text(1200, 796, "Power 80%, two-sided 5%.", size=15, fill=MUTED)
    text(1200, 818, "One task = 0.83 points.", size=15, fill=MUTED)
    text(80, 872, "Dam Solanke  ·  Kaggle ARC Prize 2026, ARC-AGI-2 public leaderboard, June to October 2026", size=14, fill=FAINT)

    OUT.mkdir(parents=True, exist_ok=True)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
           f'{"".join(parts)}</svg>\n')
    (OUT / "cover.svg").write_text(svg)
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-z", "2", "-o", str(OUT / "cover.png"), str(OUT / "cover.svg")], check=True)
    print("wrote", OUT / "cover.svg")


if __name__ == "__main__":
    summary = summarize(load_runs())
    report(summary, leaderboard())
    cover(summary)
