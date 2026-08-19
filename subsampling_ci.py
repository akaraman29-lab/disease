"""
subsampling_ci.py
=================
Tests how subsampling affects model performance.

Fixed: epidemic realisation (seed=4, existing sequences from ground_truth.csv
       and phastSim_output FASTA)
Varied: which cases are subsampled (20 random seeds × 7 subsampling rates)

For each subsample:
  - Run inference only on subsampled cases
  - Evaluate only pairs where BOTH infector and infectee are in the subsample
  - Record precision, recall, F1, and n_evaluable pairs

Outputs:
  - subsampling_ci_results.csv
  - subsampling_ci_plot.png   mean ± 95% CI vs subsampling rate
"""

import sys
import numpy as np
import pandas as pd
import regex
from Bio import SeqIO
from pathlib import Path
import matplotlib.pyplot as plt

DISEASE = Path("/Users/azrakaraman/Desktop/disease")
TF_DIR  = DISEASE / "Danish_Transmission" / "src" / "network_construction"
sys.path.insert(0, str(TF_DIR))
import transmission_functions as tf  # type: ignore

FASTA_IN = DISEASE / "phastSim_output" / "sars-cov-2_simulation_output.fasta"
GT_PATH  = DISEASE / "ground_truth.csv"
OUT_CSV  = DISEASE / "subsampling_ci_results.csv"
OUT_PLOT = DISEASE / "subsampling_ci_plot.png"

SR_INFERENCE = 0.091; TIME_WIN = 14; MAX_HAMM = 2

SUBSAMPLE_RATES = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
SEEDS           = [4, 42, 123, 456, 789, 1234, 9999,
                   2, 7, 13, 21, 50, 77, 100, 200, 300, 500, 777, 2000, 5000]
N_SEEDS = len(SEEDS)

# ── Precompute inference table ─────────────────────────────────────────────────
print("Precomputing inference probability table ...")
ndays_max = 21; nsubs_max = 9
prob_mat = np.zeros((nsubs_max + 1, ndays_max))
for s in range(nsubs_max + 1):
    for d in range(ndays_max):
        prob_mat[s, d] = tf.scenario1_probability(d, s, SR_INFERENCE, shift=0)
valid_idxs = {(0, 0)}
for d in range(ndays_max):
    cum = np.cumsum(tf.substitution_probability(np.arange(nsubs_max + 1), d, SR_INFERENCE))
    valid_idxs |= {(int(j), d) for j in np.argwhere(cum <= 0.95).flatten()}

def clean_seq(seq):
    return regex.sub(r"[^ACTG]", "-", str(seq).upper())

# ── Load full dataset once ─────────────────────────────────────────────────────
print("Loading ground truth and sequences ...")
gt = pd.read_csv(GT_PATH)
gt["strain"] = "case_" + gt["case_id"].astype(str)
true_parent  = {row.strain: f"case_{int(row.parent_id)}" for _, row in gt.iterrows()}

raw = {r.id: str(r.seq) for r in SeqIO.parse(FASTA_IN, "fasta")}
gt_seq = (gt[gt.strain.isin(raw)].copy()
            .sort_values("sample_time").reset_index(drop=True))

# Build full sequence matrix once
print("Building sequence matrix ...")
all_strains  = gt_seq.strain.values
all_sdays    = np.round(gt_seq.sample_time.values).astype(int)
seqs_np      = np.stack([np.frombuffer(clean_seq(raw[s]).encode(), dtype=np.uint8)
                          for s in all_strains])
var_mask     = np.any(seqs_np != seqs_np[0], axis=0)
seqs_var_full = seqs_np[:, var_mask]
strain_to_idx = {s: i for i, s in enumerate(all_strains)}
print(f"  {len(all_strains)} cases  |  {var_mask.sum()} variable sites")

# ── Inference on a subsample ───────────────────────────────────────────────────
def run_inference_subsample(subsample_strains):
    """Run inference restricted to subsample_strains only."""
    # Get indices into full matrix
    idxs     = np.array([strain_to_idx[s] for s in subsample_strains])
    strains  = np.array(subsample_strains)
    sdays    = all_sdays[idxs]
    seqs_var = seqs_var_full[idxs]

    # Sort by sample time (required for candidate window logic)
    order    = np.argsort(sdays, kind="stable")
    strains  = strains[order]
    sdays    = sdays[order]
    seqs_var = seqs_var[order]

    leaf_set   = set(strains)
    # Evaluable: true parent also in subsample
    eval_cases = [s for s in strains if true_parent.get(s, "case_-1") in leaf_set]

    if len(eval_cases) == 0:
        return 0.0, 0.0, 0.0, 0

    infectors = {}
    for i in range(len(strains)):
        sid = strains[i]; t_i = sdays[i]
        jm  = np.where((sdays < t_i) & (sdays >= t_i - TIME_WIN))[0]
        if not len(jm): infectors[sid] = None; continue
        h = np.sum(seqs_var[jm] != seqs_var[i], axis=1)
        jm = jm[h <= MAX_HAMM]; h = h[h <= MAX_HAMM].astype(int)
        if not len(jm): infectors[sid] = None; continue
        dd = (t_i - sdays[jm]).astype(int)
        vm = np.array([(int(hh), int(d)) in valid_idxs for hh, d in zip(h, dd)])
        if not vm.any(): infectors[sid] = None; continue
        jm = jm[vm]; h = h[vm]; dd = dd[vm]
        probs = prob_mat[np.clip(h, 0, nsubs_max).astype(int),
                         np.clip(dd, 0, ndays_max - 1).astype(int)]
        infectors[sid] = {strains[j]: float(p) for j, p in zip(jm, probs)}

    tp = fp = fn = 0; n = len(eval_cases)
    for sid in eval_cases:
        tp_ = true_parent[sid]; c = infectors.get(sid)
        if c:
            if max(c, key=lambda k: c[k]) == tp_: tp += 1
            else: fp += 1
        else: fn += 1

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / n         if n > 0        else 0.0
    f = 2*p*r / (p+r)  if (p + r) > 0 else 0.0
    return p, r, f, n

# ── Load existing results ──────────────────────────────────────────────────────
if OUT_CSV.exists():
    existing = pd.read_csv(OUT_CSV)
    done     = set(zip(existing.rate, existing.seed))
    results  = existing.to_dict("records")
    print(f"Resuming — {len(done)}/{len(SUBSAMPLE_RATES)*N_SEEDS} runs done.")
else:
    done = set(); results = []

total = len(SUBSAMPLE_RATES) * N_SEEDS
run_n = len(done)

# ── Main sweep ─────────────────────────────────────────────────────────────────
all_strains_list = list(all_strains)
N_total = len(all_strains_list)

for rate in SUBSAMPLE_RATES:
    for seed in SEEDS:
        if (rate, seed) in done:
            continue

        run_n += 1
        rng = np.random.default_rng(seed)
        n_sample = max(2, int(round(rate * N_total)))
        subsample = list(rng.choice(all_strains_list, size=n_sample, replace=False))

        prec, rec, f1, n_eval = run_inference_subsample(subsample)

        print(f"[{run_n}/{total}]  rate={rate:.0%}  seed={seed:>4}  "
              f"n={n_sample:>5}  eval={n_eval:>5}  "
              f"P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

        results.append({"rate": rate, "seed": seed, "n_sampled": n_sample,
                        "n_evaluable": n_eval,
                        "precision": prec, "recall": rec, "f1": f1})
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)

print(f"\nAll done. {len(results)} runs saved to {OUT_CSV}")

# ── Plot ───────────────────────────────────────────────────────────────────────
df  = pd.read_csv(OUT_CSV)
agg = df.groupby("rate")[["precision", "recall", "f1", "n_evaluable"]].agg(["mean", "std"])
agg.columns = ["_".join(c) for c in agg.columns]
rates = agg.index.values

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Model performance vs subsampling rate\n"
             f"(mean ± 95% CI across {N_SEEDS} random subsamples, epidemic fixed)",
             fontsize=12, fontweight="bold")

for ax, metric, color, label in [
    (axes[0], "precision", "steelblue",  "Precision"),
    (axes[1], "recall",    "darkorange", "Recall"),
    (axes[2], "f1",        "seagreen",   "F1 score"),
]:
    mean = agg[f"{metric}_mean"]
    ci   = 1.96 * agg[f"{metric}_std"] / np.sqrt(N_SEEDS)

    ax.plot(rates * 100, mean, marker="o", linewidth=2, color=color)
    ax.fill_between(rates * 100, mean - ci, mean + ci, alpha=0.25, color=color)
    ax.axvline(100, color="grey", linestyle="--", linewidth=1, label="100% (baseline)")
    ax.set_xlabel("Subsampling rate (%)", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)

# Add secondary panel for n_evaluable
ax2 = axes[2].twinx()
ax2.plot(rates * 100, agg["n_evaluable_mean"], color="crimson",
         linestyle=":", linewidth=1.5, marker="s", markersize=5, label="Evaluable pairs")
ax2.set_ylabel("Mean evaluable pairs", fontsize=9, color="crimson")
ax2.tick_params(axis="y", colors="crimson")
ax2.legend(fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {OUT_PLOT}")
