"""
mutation_rate_sweep.py
======================
Varies the simulated mutation rate (phastSim --scale) across a range of
values while keeping the inference model's assumed SR fixed at 0.091.

For each rate:
  1. Re-runs phastSim on the existing transmission_tree.nwk
  2. Runs Jacob's inference model (row-by-row, precomputed table)
  3. Records precision, recall, F1

Outputs:
  - mutation_rate_sweep_results.csv   — one row per rate
  - mutation_rate_sweep.png           — recall and F1 vs mutation rate
"""

import sys
import subprocess
import shutil
import numpy as np
import pandas as pd
import regex
from Bio import SeqIO, SeqRecord, Seq
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ──────────────────────────────────────────────────────────────────────
DISEASE   = Path("/Users/azrakaraman/Desktop/disease")
TF_DIR    = DISEASE / "Danish_Transmission" / "src" / "network_construction"
sys.path.insert(0, str(TF_DIR))
import transmission_functions as tf  # type: ignore

REF       = Path("/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/phastSim/example/MN908947.3.fasta")
NWK       = DISEASE / "transmission_tree.nwk"
GT_PATH   = DISEASE / "ground_truth.csv"
OUT_CSV   = DISEASE / "mutation_rate_sweep_results.csv"
OUT_PLOT  = DISEASE / "mutation_rate_sweep.png"
PHAST_TMP = DISEASE / "phastSim_sweep_tmp"

GENOME_LEN   = 29903
SR_INFERENCE = 0.091   # fixed — inference model always assumes the Denmark rate
TIME_WIN     = 14
MAX_HAMM     = 2
SEED         = 4

# Mutation rates to sweep (subs/genome/day)
MU_RATES = [0.005, 0.01, 0.025, 0.05, 0.091, 0.15, 0.25, 0.5, 1.0]

# ── Load ground truth once ─────────────────────────────────────────────────────
print("Loading ground truth ...")
gt = pd.read_csv(GT_PATH)
gt["strain"] = "case_" + gt["case_id"].astype(str)
true_parent  = {row.strain: f"case_{int(row.parent_id)}"
                for _, row in gt.iterrows()}

# ── Precompute inference lookup table once (SR_INFERENCE is fixed) ─────────────
print(f"Precomputing probability table (SR_inference = {SR_INFERENCE}) ...")
ndays_max = 21
nsubs_max = 9

prob_mat = np.zeros((nsubs_max + 1, ndays_max))
for sub in range(nsubs_max + 1):
    for d in range(ndays_max):
        prob_mat[sub, d] = tf.scenario1_probability(d, sub, SR_INFERENCE, shift=0)

valid_idxs = {(0, 0)}
for d in range(ndays_max):
    cum = np.cumsum(tf.substitution_probability(np.arange(nsubs_max + 1), d, SR_INFERENCE))
    valid_idxs |= {(int(j), d) for j in np.argwhere(cum <= 0.95).flatten()}

# ── Helper: clean sequence ─────────────────────────────────────────────────────
def clean_seq(seq):
    return regex.sub(r'[^ACTG]', '-', str(seq).upper())

# ── Helper: run inference on a FASTA, return (precision, recall, f1) ──────────
def run_inference(fasta_path):
    fasta_id_set = {r.id for r in SeqIO.parse(fasta_path, "fasta")}
    raw          = {r.id: str(r.seq) for r in SeqIO.parse(fasta_path, "fasta")}

    gt_leaves = (gt[gt.strain.isin(fasta_id_set)]
                   .copy()
                   .sort_values("sample_time")
                   .reset_index(drop=True))

    seqs_list = [np.frombuffer(clean_seq(raw[s]).encode(), dtype=np.uint8)
                 for s in gt_leaves.strain]
    seqs_np   = np.stack(seqs_list)
    ref_row   = seqs_np[0]
    var_mask  = np.any(seqs_np != ref_row, axis=0)
    seqs_var  = seqs_np[:, var_mask]

    strains      = gt_leaves.strain.values
    sample_days  = np.round(gt_leaves.sample_time.values).astype(int)
    leaf_set     = set(strains)
    eval_cases   = [s for s in strains if true_parent.get(s, "case_-1") in leaf_set]

    infectors = {}
    for i in range(len(strains)):
        t_i    = sample_days[i]
        sid    = strains[i]
        j_mask = (sample_days < t_i) & (sample_days >= t_i - TIME_WIN)
        j_idxs = np.where(j_mask)[0]
        if len(j_idxs) == 0:
            infectors[sid] = None
            continue
        hamming = np.sum(seqs_var[j_idxs] != seqs_var[i], axis=1)
        keep    = hamming <= MAX_HAMM
        if not keep.any():
            infectors[sid] = None
            continue
        j_idxs    = j_idxs[keep]
        hamming   = hamming[keep].astype(int)
        datediffs = (t_i - sample_days[j_idxs]).astype(int)
        valid_mask = np.array([(int(h), int(d)) in valid_idxs
                               for h, d in zip(hamming, datediffs)])
        if not valid_mask.any():
            infectors[sid] = None
            continue
        j_idxs    = j_idxs[valid_mask]
        hamming   = hamming[valid_mask]
        datediffs = datediffs[valid_mask]
        h_clip    = np.clip(hamming,   0, nsubs_max).astype(int)
        d_clip    = np.clip(datediffs, 0, ndays_max - 1).astype(int)
        probs     = prob_mat[h_clip, d_clip]
        infectors[sid] = {strains[j]: float(p) for j, p in zip(j_idxs, probs)}

    tp = fp = fn = 0
    n  = len(eval_cases)
    for sid in eval_cases:
        true_p = true_parent[sid]
        cands  = infectors.get(sid)
        if cands:
            inferred = max(cands, key=lambda k: cands[k])
            if inferred == true_p:
                tp += 1
            else:
                fp += 1
        else:
            fn += 1

    # Recall = TP / N: every evaluable case is a true link;
    # wrong guesses and missing candidates both count as failures.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / n         if n > 0           else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    n_var     = int(var_mask.sum())
    return precision, recall, f1, n_var

# ── Main sweep ─────────────────────────────────────────────────────────────────
PHAST_TMP.mkdir(exist_ok=True)
results = []

for mu in MU_RATES:
    scale = mu / GENOME_LEN
    print(f"\n{'─'*60}")
    print(f"  mu = {mu:.4f} subs/genome/day  (scale = {scale:.3e})")

    # 1. Run phastSim
    run_dir = PHAST_TMP / f"mu_{mu:.4f}".replace(".", "p")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir()

    cmd = [
        "phastSim",
        "--outpath",   str(run_dir) + "/",
        "--reference", str(REF),
        "--treeFile",  str(NWK),
        "--scale",     str(scale),
        "--seed",      str(SEED),
        "--createFasta",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  phastSim failed: {res.stderr[:300]}")
        continue

    fasta_path = next(run_dir.glob("*.fasta"), None)
    if fasta_path is None:
        print("  No FASTA output found, skipping.")
        continue

    # 2. Run inference
    print(f"  Running inference ...")
    precision, recall, f1, n_var = run_inference(fasta_path)

    print(f"  Variable sites : {n_var:,}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
    print(f"  F1             : {f1:.4f}")

    results.append({
        "mu_sim"         : mu,
        "scale"          : scale,
        "n_variable_sites": n_var,
        "precision"      : precision,
        "recall"         : recall,
        "f1"             : f1,
    })

    # Save progress after each rate in case of interruption
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)

shutil.rmtree(PHAST_TMP)
print(f"\n\nSweep complete. Results saved: {OUT_CSV}")

# ── Plot ───────────────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Model performance vs simulated mutation rate\n"
    f"(inference SR fixed at {SR_INFERENCE} subs/genome/day)",
    fontsize=12, fontweight="bold"
)

for ax, metric, color, label in [
    (axes[0], "recall",    "steelblue", "Recall"),
    (axes[1], "f1",        "darkorange", "F1 score"),
]:
    ax.plot(df_res.mu_sim, df_res[metric],
            marker="o", linewidth=2, markersize=7, color=color)
    # Mark the true SARS-CoV-2 rate
    ax.axvline(SR_INFERENCE, color="grey", linestyle="--", linewidth=1.2,
               label=f"True rate (0.091)")
    ax.set_xlabel("Simulated mutation rate (subs/genome/day)", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

# Add n_variable_sites as a secondary x-axis label on panel A
ax2 = axes[0].twiny()
ax2.set_xlim(axes[0].get_xlim())
ax2.set_xscale("log")
tick_mus = df_res.mu_sim.values
ax2.set_xticks(tick_mus)
ax2.set_xticklabels([f"{int(r):,}" for r in df_res.n_variable_sites.values],
                    fontsize=7, rotation=45)
ax2.set_xlabel("Variable sites in simulated sequences", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
print(f"Plot saved: {OUT_PLOT}")
plt.show()
