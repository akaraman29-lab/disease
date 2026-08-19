"""
evaluate_inference.py
=====================
Five evaluation strategies for Jacob's transmission inference model.
Works entirely from infectors_dict.pickle + ground_truth.csv — no re-running the model.

Strategies
----------
1. Top-1          : true parent has the single highest score
2. Top-k          : true parent appears in top k=2, 3, 5 by score
3. Any candidate  : true parent appears anywhere in the candidate set
4. Rank histogram : distribution of true parent's rank (for found cases)
5. MRR            : mean reciprocal rank across all evaluable cases
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

DISEASE  = Path("/Users/azrakaraman/Desktop/disease")
GT_PATH  = DISEASE / "ground_truth.csv"
PKL_PATH = DISEASE / "infectors_dict.pickle"
OUT_CSV  = DISEASE / "evaluation_results.csv"

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data ...")
gt = pd.read_csv(GT_PATH)
gt["strain"] = "case_" + gt["case_id"].astype(str)

with open(PKL_PATH, "rb") as f:
    infectors_dict = pickle.load(f)

# ── Build true parent lookup ───────────────────────────────────────────────────
# Exclude index case (parent_id == -1)
true_parent = {
    row.strain: f"case_{int(row.parent_id)}"
    for _, row in gt.iterrows()
    if row.parent_id != -1
}

sequenced   = set(infectors_dict.keys())
# Only evaluate cases whose true parent is also sequenced
eval_strains = [s for s in true_parent if true_parent[s] in sequenced]
print(f"  Evaluable cases: {len(eval_strains)}")

# ── Build ranked candidate list per case ──────────────────────────────────────
# infectors_dict[sid] = {candidate_strain: (prob, (hamming, datediff))} or None
def get_ranked(sid):
    """Return list of candidate strains sorted by score descending."""
    cands = infectors_dict.get(sid)
    if not cands:
        return []
    return sorted(cands, key=lambda k: cands[k][0], reverse=True)

# ── Per-case evaluation ────────────────────────────────────────────────────────
rows = []
for sid in eval_strains:
    true_p = true_parent[sid]
    ranked = get_ranked(sid)

    if true_p in ranked:
        rank = ranked.index(true_p) + 1   # 1-based
    else:
        rank = None

    # Strategy 6a: normalised probability assigned to the true parent
    cands = infectors_dict.get(sid)
    if cands and true_p in cands:
        total_score = sum(v[0] for v in cands.values())
        norm_prob   = cands[true_p][0] / total_score if total_score > 0 else 0.0
    else:
        norm_prob   = 0.0

    rows.append({
        "strain"          : sid,
        "true_parent"     : true_p,
        "n_candidates"    : len(ranked),
        "true_parent_rank": rank,
        "top1_correct"    : rank == 1,
        "top2_correct"    : rank is not None and rank <= 2,
        "top3_correct"    : rank is not None and rank <= 3,
        "top5_correct"    : rank is not None and rank <= 5,
        "reciprocal_rank" : (1 / rank) if rank is not None else 0.0,
        "norm_prob"       : norm_prob,
    })

results = pd.DataFrame(rows)
results.to_csv(OUT_CSV, index=False)
print(f"  Saved: {OUT_CSV}\n")

N = len(results)

# ── Strategy 1: Top-1 ─────────────────────────────────────────────────────────
tp1 = results.top1_correct.sum()
fp1 = ((results.n_candidates > 0) & ~results.top1_correct).sum()
fn1 = (results.n_candidates == 0).sum()
p1  = tp1 / (tp1 + fp1) if (tp1 + fp1) > 0 else 0
r1  = tp1 / N
f1  = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0

# ── Strategy 2: Top-k ─────────────────────────────────────────────────────────
topk = {k: results[f"top{k}_correct"].sum() / N
        for k in [2, 3, 5]}

# ── Strategy 3: Any candidate ─────────────────────────────────────────────────
any_found = (results.true_parent_rank.notna()).sum()
any_rate  = any_found / N

# ── Strategy 4: Rank histogram ────────────────────────────────────────────────
found     = results[results.true_parent_rank.notna()].true_parent_rank.astype(int)
bins      = list(range(1, 11)) + [float("inf")]
labels    = [str(i) for i in range(1, 11)] + ["11+"]
hist      = pd.cut(found, bins=[0]+list(range(1,11))+[10000],
                   labels=labels, right=True).value_counts().sort_index()

# ── Strategy 5: MRR ───────────────────────────────────────────────────────────
mrr = results.reciprocal_rank.mean()

# ── Strategy 6a: Expected precision (probability-weighted) ────────────────────
# For each case, normalise raw scores to sum to 1, take the normalised score
# of the true parent. Average over all cases (0 if true parent not in candidates).
exp_precision = results.norm_prob.mean()

# ── Print summary ─────────────────────────────────────────────────────────────
W = 52
print("=" * W)
print(f"  EVALUATION SUMMARY  ({N} evaluable cases)")
print("=" * W)

print("\n  Strategy 1 — Top-1 (baseline)")
print(f"    True positives  : {tp1:>6}")
print(f"    False positives : {fp1:>6}")
print(f"    False negatives : {fn1:>6}")
print(f"    Precision       : {p1:.4f}")
print(f"    Recall          : {r1:.4f}")
print(f"    F1              : {f1:.4f}")

print("\n  Strategy 2 — Top-k recall")
print(f"    Top-1  : {tp1/N:.4f}  ({tp1}/{N})")
for k in [2, 3, 5]:
    n = results[f"top{k}_correct"].sum()
    print(f"    Top-{k}  : {n/N:.4f}  ({n}/{N})")

print("\n  Strategy 3 — Any candidate")
print(f"    Found in candidate set : {any_rate:.4f}  ({any_found}/{N})")

print("\n  Strategy 4 — Rank distribution (when true parent is found)")
print(f"    {'Rank':<8} {'Count':>6}  {'%':>7}")
print(f"    {'────':<8} {'─────':>6}  {'──────':>7}")
for label, count in hist.items():
    pct = 100 * count / any_found if any_found > 0 else 0
    print(f"    {label:<8} {count:>6}  {pct:>6.1f}%")

print("\n  Strategy 5 — Mean Reciprocal Rank")
print(f"    MRR : {mrr:.4f}")

print("\n  Strategy 6a — Expected precision (probability-weighted)")
print(f"    Mean normalised score of true parent : {exp_precision:.4f}")
print(f"    Interpretation: on average the model assigns {exp_precision*100:.1f}% of its")
print(f"    probability mass to the correct infector")

print("\n" + "=" * W)
