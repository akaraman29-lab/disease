"""
bootstrap_evaluate.py
=====================
Bootstrap 95% confidence intervals for all 6 evaluation strategies.
Reads evaluation_results.csv (one row per case) produced by evaluate_inference.py.

Each bootstrap iteration resamples rows with replacement and recomputes
every statistic. 2000 iterations gives stable 95% CIs.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DISEASE  = Path("/Users/azrakaraman/Desktop/disease")
CSV_PATH = DISEASE / "evaluation_results.csv"
OUT_CSV  = DISEASE / "bootstrap_results.csv"

B    = 2000   # bootstrap iterations
SEED = 42
RNG  = np.random.default_rng(SEED)

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading evaluation_results.csv ...")
df = pd.read_csv(CSV_PATH)
N  = len(df)
print(f"  {N} evaluable cases\n")

# Pre-cast columns used repeatedly
top1    = df["top1_correct"].astype(bool).values
top2    = df["top2_correct"].astype(bool).values
top3    = df["top3_correct"].astype(bool).values
top5    = df["top5_correct"].astype(bool).values
n_cands = df["n_candidates"].values
rr      = df["reciprocal_rank"].values
np_val  = df["norm_prob"].values
rank    = df["true_parent_rank"].values   # NaN when not in candidate set

# Rank histogram bin membership (per case, as 0/1 arrays)
RANK_LABELS = [str(i) for i in range(1, 11)] + ["11+"]
rank_bins = {}
for r in range(1, 11):
    rank_bins[str(r)] = (rank == r).astype(float)
rank_bins["11+"] = (rank >= 11).astype(float)
found = ~np.isnan(rank)   # True when true parent is in candidate set

# ── Helper: compute all statistics on an index array ──────────────────────────
def compute_stats(idx):
    t1   = top1[idx]
    t2   = top2[idx]
    t3   = top3[idx]
    t5   = top5[idx]
    nc   = n_cands[idx]
    rrv  = rr[idx]
    npv  = np_val[idx]
    fnd  = found[idx]
    n    = len(idx)

    # Strategy 1
    tp   = t1.sum()
    fp   = ((nc > 0) & ~t1).sum()
    fn   = (nc == 0).sum()
    p1   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r1   = tp / n
    f1   = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0.0

    # Strategy 2
    top2_rate = t2.mean()
    top3_rate = t3.mean()
    top5_rate = t5.mean()

    # Strategy 3
    any_rate  = fnd.mean()

    # Strategy 4 — % of found cases in each rank bin
    any_found = fnd.sum()
    hist = {}
    for label in RANK_LABELS:
        col = rank_bins[label][idx]
        hist[label] = col.sum() / any_found if any_found > 0 else 0.0

    # Strategy 5
    mrr = rrv.mean()

    # Strategy 6
    exp_prec = npv.mean()

    return {
        "s1_precision" : p1,
        "s1_recall"    : r1,
        "s1_f1"        : f1,
        "s2_top2"      : top2_rate,
        "s2_top3"      : top3_rate,
        "s2_top5"      : top5_rate,
        "s3_any"       : any_rate,
        **{f"s4_rank{lbl}": v for lbl, v in hist.items()},
        "s5_mrr"       : mrr,
        "s6_exp_prec"  : exp_prec,
    }

# ── Point estimates ────────────────────────────────────────────────────────────
point = compute_stats(np.arange(N))

# ── Bootstrap ─────────────────────────────────────────────────────────────────
print(f"Running {B} bootstrap iterations ...")
boot_records = []
for _ in range(B):
    idx = RNG.integers(0, N, size=N)
    boot_records.append(compute_stats(idx))

boot = pd.DataFrame(boot_records)
lo   = boot.quantile(0.025)
hi   = boot.quantile(0.975)
print("  Done.\n")

# ── Print results ──────────────────────────────────────────────────────────────
W = 62

def fmt(key):
    v  = point[key]
    l  = lo[key]
    h  = hi[key]
    return f"{v:.4f}  [{l:.4f}, {h:.4f}]"

print("=" * W)
print(f"  BOOTSTRAP RESULTS  ({B} iterations, 95% CI)")
print("=" * W)

print("\n  Strategy 1 — Top-1  [point estimate  (95% CI)]")
print(f"    Precision : {fmt('s1_precision')}")
print(f"    Recall    : {fmt('s1_recall')}")
print(f"    F1        : {fmt('s1_f1')}")

print("\n  Strategy 2 — Top-k recall")
print(f"    Top-1 : {fmt('s1_recall')}")
print(f"    Top-2 : {fmt('s2_top2')}")
print(f"    Top-3 : {fmt('s2_top3')}")
print(f"    Top-5 : {fmt('s2_top5')}")

print("\n  Strategy 3 — Any candidate")
print(f"    Found in candidate set : {fmt('s3_any')}")

print("\n  Strategy 4 — Rank histogram (% of found cases)")
print(f"    {'Rank':<6}  {'Point':>8}  {'95% CI'}")
print(f"    {'────':<6}  {'─────':>8}  {'──────────────────'}")
for lbl in RANK_LABELS:
    key = f"s4_rank{lbl}"
    v, l, h = point[key], lo[key], hi[key]
    print(f"    {lbl:<6}  {v:>7.1%}  [{l:.1%}, {h:.1%}]")

print("\n  Strategy 5 — Mean Reciprocal Rank")
print(f"    MRR : {fmt('s5_mrr')}")

print("\n  Strategy 6 — Expected precision (probability-weighted)")
print(f"    Mean norm. score : {fmt('s6_exp_prec')}")

print("\n" + "=" * W)

# ── Save full bootstrap distributions ─────────────────────────────────────────
summary_rows = []
for key in point.keys():
    summary_rows.append({
        "statistic"    : key,
        "point_estimate": point[key],
        "ci_lo"        : lo[key],
        "ci_hi"        : hi[key],
    })
pd.DataFrame(summary_rows).to_csv(OUT_CSV, index=False)
print(f"\n  Full CI table saved: {OUT_CSV}")
