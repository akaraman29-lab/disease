"""
subsampling_realistic_ci.py
===========================
Realistic evaluation of subsampling: any prediction made when the true
infector is absent from the subsample counts as a false positive.

Original evaluation (subsampling_ci.py):
  - Only evaluate pairs where BOTH infector and infectee are in subsample
  - Ignores all cases with absent infectors entirely

Realistic evaluation (this script):
  - For cases where infector IS present: TP / FP / FN as before
  - For cases where infector is ABSENT: any prediction = FP
  - Recall denominator = cases where infector is present (true findable links)

This gives a harsher, more realistic picture of precision at low sampling rates.

Outputs:
  - subsampling_realistic_results.csv
  - subsampling_realistic_plot.png    comparison of original vs realistic evaluation
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
OUT_CSV  = DISEASE / "subsampling_realistic_results.csv"
OUT_PLOT = DISEASE / "subsampling_realistic_plot.png"

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

print("Building sequence matrix ...")
all_strains  = gt_seq.strain.values
all_sdays    = np.round(gt_seq.sample_time.values).astype(int)
seqs_np      = np.stack([np.frombuffer(clean_seq(raw[s]).encode(), dtype=np.uint8)
                          for s in all_strains])
var_mask     = np.any(seqs_np != seqs_np[0], axis=0)
seqs_var_full = seqs_np[:, var_mask]
strain_to_idx = {s: i for i, s in enumerate(all_strains)}
print(f"  {len(all_strains)} cases  |  {var_mask.sum()} variable sites")

# ── Inference on a subsample (realistic evaluation) ───────────────────────────
def run_inference_realistic(subsample_strains):
    """
    Runs inference on subsample_strains.

    Realistic evaluation:
      - Cases with infector PRESENT: TP / FP / FN as normal
      - Cases with infector ABSENT:  any prediction → FP; no prediction → ignored
      - Recall denominator = cases where infector is present
    """
    idxs     = np.array([strain_to_idx[s] for s in subsample_strains])
    strains  = np.array(subsample_strains)
    sdays    = all_sdays[idxs]
    seqs_var = seqs_var_full[idxs]

    order    = np.argsort(sdays, kind="stable")
    strains  = strains[order]
    sdays    = sdays[order]
    seqs_var = seqs_var[order]

    leaf_set = set(strains)

    # Run inference for every case in the subsample
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

    # Evaluate ALL cases, not just evaluable ones
    tp = fp = fn = 0
    n_infector_present = 0  # recall denominator

    for sid in strains:
        tp_        = true_parent.get(sid, "case_-1")
        c          = infectors.get(sid)
        inf_present = tp_ in leaf_set

        if inf_present:
            n_infector_present += 1
            if c:
                if max(c, key=lambda k: c[k]) == tp_:
                    tp += 1
                else:
                    fp += 1   # wrong guess, infector was there
            else:
                fn += 1       # no candidates, infector was there
        else:
            # Infector absent from subsample
            if c:
                fp += 1       # spurious prediction — infector not even in dataset

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / n_infector_present if n_infector_present > 0 else 0.0
    f = 2*p*r / (p+r) if (p + r) > 0 else 0.0
    return p, r, f, n_infector_present, tp, fp, fn


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
all_strains_list = list(all_strains)
N_total = len(all_strains_list)

# ── Main sweep ─────────────────────────────────────────────────────────────────
for rate in SUBSAMPLE_RATES:
    for seed in SEEDS:
        if (rate, seed) in done:
            continue

        run_n += 1
        rng = np.random.default_rng(seed)
        n_sample = max(2, int(round(rate * N_total)))
        subsample = list(rng.choice(all_strains_list, size=n_sample, replace=False))

        prec, rec, f1, n_pres, tp, fp, fn = run_inference_realistic(subsample)

        print(f"[{run_n}/{total}]  rate={rate:.0%}  seed={seed:>4}  "
              f"n={n_sample:>5}  infector_present={n_pres:>5}  "
              f"TP={tp:>5}  FP={fp:>5}  FN={fn:>4}  "
              f"P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

        results.append({"rate": rate, "seed": seed, "n_sampled": n_sample,
                        "n_infector_present": n_pres,
                        "tp": tp, "fp": fp, "fn": fn,
                        "precision": prec, "recall": rec, "f1": f1})
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)

print(f"\nAll done. {len(results)} runs saved to {OUT_CSV}")

# ── Plot: comparison of original vs realistic ──────────────────────────────────
df_real = pd.read_csv(OUT_CSV)
agg_real = df_real.groupby("rate")[["precision","recall","f1"]].agg(["mean","std"])
agg_real.columns = ["_".join(c) for c in agg_real.columns]

orig_path = DISEASE / "subsampling_ci_results.csv"
if orig_path.exists():
    df_orig = pd.read_csv(orig_path)
    agg_orig = df_orig.groupby("rate")[["precision","recall","f1"]].agg(["mean","std"])
    agg_orig.columns = ["_".join(c) for c in agg_orig.columns]
    has_orig = True
else:
    has_orig = False

rates = agg_real.index.values

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"Original vs realistic evaluation  (mean ± 95% CI, {N_SEEDS} seeds)\n"
    "Realistic: predictions with absent infector count as FP",
    fontsize=12, fontweight="bold")

for ax, metric, color, label in [
    (axes[0], "precision", "steelblue",  "Precision"),
    (axes[1], "recall",    "darkorange", "Recall"),
    (axes[2], "f1",        "seagreen",   "F1 score"),
]:
    mean_r = agg_real[f"{metric}_mean"]
    ci_r   = 1.96 * agg_real[f"{metric}_std"] / np.sqrt(N_SEEDS)

    ax.plot(rates * 100, mean_r, marker="o", linewidth=2, color=color,
            label="Realistic")
    ax.fill_between(rates * 100, mean_r - ci_r, mean_r + ci_r, alpha=0.25, color=color)

    if has_orig:
        mean_o = agg_orig[f"{metric}_mean"]
        ci_o   = 1.96 * agg_orig[f"{metric}_std"] / np.sqrt(N_SEEDS)
        ax.plot(rates * 100, mean_o, marker="s", linewidth=2, color=color,
                linestyle="--", alpha=0.6, label="Original (evaluable only)")
        ax.fill_between(rates * 100, mean_o - ci_o, mean_o + ci_o,
                        alpha=0.10, color=color)

    ax.set_xlabel("Subsampling rate (%)", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {OUT_PLOT}")
