"""
tie_and_indegree_analysis.py
============================
Two analyses:

1. Tie analysis — for each case, how many candidates share the top score?
   Since scores are keyed on (hamming, datediff) only, any two candidates
   sampled the same number of days ago with the same mutation distance get
   identical scores. Top-1 'correct' is therefore partly luck of ordering.

2. In-degree over time — scatter/ribbon plot of n_candidates per case vs
   sample_time, showing the needle-in-a-haystack problem growing as the
   epidemic matures.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

DISEASE  = Path("/Users/azrakaraman/Desktop/disease")
PKL_PATH = DISEASE / "infectors_dict.pickle"
GT_PATH  = DISEASE / "ground_truth.csv"
OUT_DIR  = DISEASE

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data ...")
with open(PKL_PATH, "rb") as f:
    infectors_dict = pickle.load(f)

gt = pd.read_csv(GT_PATH)
gt["strain"] = "case_" + gt["case_id"].astype(str)
sample_time  = gt.set_index("strain")["sample_time"].to_dict()

# ── Per-case stats ─────────────────────────────────────────────────────────────
rows = []
for sid, cands in infectors_dict.items():
    t = sample_time.get(sid, np.nan)
    if cands is None or len(cands) == 0:
        rows.append({"strain": sid, "sample_time": t,
                     "n_candidates": 0, "n_tied_top": 0, "top_score": np.nan})
        continue

    scores    = np.array([v[0] for v in cands.values()])
    top_score = scores.max()
    n_tied    = int((scores == top_score).sum())
    rows.append({"strain": sid, "sample_time": t,
                 "n_candidates": len(cands), "n_tied_top": n_tied,
                 "top_score": top_score})

df = pd.DataFrame(rows)
has_cands = df[df.n_candidates > 0].copy()

# ── 1. Tie analysis — print summary ───────────────────────────────────────────
print("\n── Tie analysis (cases with ≥1 candidate) ──────────────────────")
print(f"  Cases with candidates   : {len(has_cands):,}")
print(f"  Cases where top-1 is a  : {(has_cands.n_tied_top == 1).sum():,}  "
      f"({100*(has_cands.n_tied_top == 1).mean():.1f}%)  unique winner")
print(f"  Cases with 2 tied        : {(has_cands.n_tied_top == 2).sum():,}  "
      f"({100*(has_cands.n_tied_top == 2).mean():.1f}%)")
print(f"  Cases with 3–5 tied      : {((has_cands.n_tied_top >= 3) & (has_cands.n_tied_top <= 5)).sum():,}  "
      f"({100*((has_cands.n_tied_top >= 3) & (has_cands.n_tied_top <= 5)).mean():.1f}%)")
print(f"  Cases with >5 tied       : {(has_cands.n_tied_top > 5).sum():,}  "
      f"({100*(has_cands.n_tied_top > 5).mean():.1f}%)")
print(f"  Median tied at top       : {has_cands.n_tied_top.median():.0f}")
print(f"  Mean   tied at top       : {has_cands.n_tied_top.mean():.1f}")
print(f"  Max    tied at top       : {has_cands.n_tied_top.max()}")

# ── 2. Plots ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Needle-in-a-haystack: candidate set size and score ties",
             fontsize=13, fontweight="bold", y=1.01)

# ── Panel A: in-degree over time (binned median + ribbon) ─────────────────────
ax = axes[0]
bin_width = 5   # days
t_max  = int(np.ceil(has_cands.sample_time.max()))
bins   = np.arange(0, t_max + bin_width, bin_width)
labels = (bins[:-1] + bins[1:]) / 2
has_cands["time_bin"] = pd.cut(has_cands.sample_time, bins=bins, labels=labels)
grouped = has_cands.groupby("time_bin", observed=True)["n_candidates"]
med  = grouped.median()
p25  = grouped.quantile(0.25)
p75  = grouped.quantile(0.75)
p90  = grouped.quantile(0.90)
x    = med.index.astype(float)

ax.fill_between(x, p25, p75, alpha=0.25, color="steelblue", label="IQR")
ax.fill_between(x, p75, p90, alpha=0.12, color="steelblue", label="75th–90th pct")
ax.plot(x, med, color="steelblue", linewidth=2, label="Median")
ax.set_xlabel("Sample time (days)", fontsize=11)
ax.set_ylabel("Number of candidates (in-degree)", fontsize=11)
ax.set_title("A — Candidate set size over time", fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(0, t_max)
ax.set_ylim(bottom=0)

# ── Panel B: distribution of n_candidates (log-scale histogram) ───────────────
ax = axes[1]
max_c = int(has_cands.n_candidates.max())
bin_edges = np.arange(0, min(max_c + 2, 200))
ax.hist(has_cands.n_candidates.clip(upper=198), bins=bin_edges,
        color="steelblue", edgecolor="white", linewidth=0.3)
ax.set_xlabel("Number of candidates per case", fontsize=11)
ax.set_ylabel("Count (cases)", fontsize=11)
ax.set_title("B — In-degree distribution", fontsize=11)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
# Vertical line at median
med_nc = has_cands.n_candidates.median()
ax.axvline(med_nc, color="crimson", linewidth=1.5, linestyle="--",
           label=f"Median = {med_nc:.0f}")
ax.legend(fontsize=9)

# ── Panel C: distribution of n_tied_top ───────────────────────────────────────
ax = axes[2]
tie_counts = has_cands.n_tied_top.clip(upper=50)
bins_tie   = np.arange(0.5, 52.5, 1)
ax.hist(tie_counts, bins=bins_tie,
        color="darkorange", edgecolor="white", linewidth=0.3)
ax.set_xlabel("Number of candidates tied at top score", fontsize=11)
ax.set_ylabel("Count (cases)", fontsize=11)
ax.set_title("C — Score ties at top-1", fontsize=11)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
med_tie = has_cands.n_tied_top.median()
ax.axvline(med_tie, color="crimson", linewidth=1.5, linestyle="--",
           label=f"Median = {med_tie:.0f}")
pct_unique = 100 * (has_cands.n_tied_top == 1).mean()
ax.text(0.97, 0.97, f"{pct_unique:.1f}% unique winner",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        color="crimson")
ax.legend(fontsize=9)

plt.tight_layout()
out_path = OUT_DIR / "tie_and_indegree.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n  Plot saved: {out_path}")
plt.show()
