"""
Adapter: runs Jacob's transmission inference model on our simulated outbreak.

Inputs  (in /Users/azrakaraman/Desktop/disease/):
  - phastSim_output/sars-cov-2_simulation_output.fasta   genome sequences
  - ground_truth.csv                                      sample times + true parents

Outputs (same folder):
  - sequences_clean.fasta        uppercase, non-ACTG replaced with '-'
  - infectors_dict.pickle        {strain: {candidate_strain: (prob, (hamming, datediff))}}
  - inference_results.csv        per-case inferred vs true parent

Precision / recall / F1 are printed at the end.

Approach: row-by-row (nodewise). For each case i, we only load the sequences
of candidates in the time window — no full N×N matrix is ever built.
"""

import sys
import numpy as np
import pandas as pd
import scipy as sp
import pickle
import regex
from Bio import SeqIO, SeqRecord, Seq
from pathlib import Path
from tqdm import tqdm
import hammingdist

# ── Paths ──────────────────────────────────────────────────────────────────────
DISEASE     = Path("/Users/azrakaraman/Desktop/disease")

# ── Point Python to Jacob's transmission_functions.py ─────────────────────────
TF_DIR = DISEASE / "Danish_Transmission" / "src" / "network_construction"
sys.path.insert(0, str(TF_DIR))
import transmission_functions as tf #type: ignore 
FASTA_IN    = DISEASE / "phastSim_output" / "sars-cov-2_simulation_output.fasta"
FASTA_CLEAN = DISEASE / "sequences_clean.fasta"
GT_PATH     = DISEASE / "ground_truth.csv"
OUT_PICKLE  = DISEASE / "infectors_dict.pickle"
OUT_CSV     = DISEASE / "inference_results.csv"

SR        = 0.091   # substitution rate (matches simulation)
TIME_WIN  = 14      # candidate window in days (Jacob uses 14)
MAX_HДMM  = 2       # Hamming threshold

# ── 1. Load ground truth ───────────────────────────────────────────────────────
print("Loading ground truth ...")
gt = pd.read_csv(GT_PATH)
gt["strain"] = "case_" + gt["case_id"].astype(str)

fasta_id_set = {r.id for r in SeqIO.parse(FASTA_IN, "fasta")}
gt_leaves = (gt[gt.strain.isin(fasta_id_set)]
               .copy()
               .sort_values("sample_time")
               .reset_index(drop=True))
N = len(gt_leaves)
print(f"  {N} cases with sequences")

# ── 2. Write clean FASTA ───────────────────────────────────────────────────────
print("Writing clean FASTA ...")

def clean_seq(seq):
    return regex.sub(r'[^ACTG]', '-', str(seq).upper())

raw = {r.id: str(r.seq) for r in SeqIO.parse(FASTA_IN, "fasta")}
clean_records = [
    SeqRecord.SeqRecord(Seq.Seq(clean_seq(raw[s])), id=s, name=s, description="")
    for s in gt_leaves.strain
]
SeqIO.write(clean_records, FASTA_CLEAN, "fasta")
print(f"  Saved: {FASTA_CLEAN}")

# ── 3. Load all sequences as numpy byte arrays (variable sites only) ───────────
print("Loading sequences into memory ...")
seqs_list  = [np.frombuffer(clean_seq(raw[s]).encode(), dtype=np.uint8)
              for s in gt_leaves.strain]
seqs_np    = np.stack(seqs_list)                     # (N, L)

# Compress to variable sites — dramatically speeds up Hamming computation
ref        = seqs_np[0]
var_mask   = np.any(seqs_np != ref, axis=0)
seqs_var   = seqs_np[:, var_mask]                    # (N, V)
print(f"  {var_mask.sum()} variable positions across all sequences")

# ── 4. Precompute valid (hamming, datediff) pairs and prob table ───────────────
ndays_max = 21
nsubs_max = 9

prob_mat = np.zeros((nsubs_max + 1, ndays_max))
for sub in range(nsubs_max + 1):
    for d in range(ndays_max):
        prob_mat[sub, d] = tf.scenario1_probability(d, sub, SR, shift=0)

valid_idxs = {(0, 0)}
for d in range(ndays_max):
    cum = np.cumsum(tf.substitution_probability(np.arange(nsubs_max + 1), d, SR))
    valid_idxs |= {(int(j), d) for j in np.argwhere(cum <= 0.95).flatten()}

# ── 5. Row-by-row inference ────────────────────────────────────────────────────
print(f"\nRunning inference ({N} cases) ...")
print(f"  Time window: 0–{TIME_WIN} days  |  Hamming threshold: <= {MAX_HДMM}")

strains      = gt_leaves.strain.values
sample_times = gt_leaves.sample_time.values          # float days
sample_days  = np.round(sample_times).astype(int)

infectors_dict = {}

for i in tqdm(range(N)):
    t_i   = sample_days[i]
    sid   = strains[i]

    # Candidates: sampled strictly before i, within TIME_WIN days
    j_mask = (sample_days < t_i) & (sample_days >= t_i - TIME_WIN)
    j_idxs = np.where(j_mask)[0]

    if len(j_idxs) == 0:
        infectors_dict[sid] = None
        continue

    # Hamming distances (vectorised over variable sites only)
    hamming = np.sum(seqs_var[j_idxs] != seqs_var[i], axis=1)   # (M,)

    # Keep only pairs with Hamming <= MAX_HAMM
    keep = hamming <= MAX_HДMM
    if not keep.any():
        infectors_dict[sid] = None
        continue

    j_idxs  = j_idxs[keep]
    hamming = hamming[keep].astype(int)
    datediffs = (t_i - sample_days[j_idxs]).astype(int)

    # Keep only (hamming, datediff) pairs in valid_idxs
    valid_mask = np.array([(int(h), int(d)) in valid_idxs
                           for h, d in zip(hamming, datediffs)])
    if not valid_mask.any():
        infectors_dict[sid] = None
        continue

    j_idxs    = j_idxs[valid_mask]
    hamming   = hamming[valid_mask]
    datediffs = datediffs[valid_mask]

    # Score each candidate via precomputed table (instant lookup)
    h_clip = np.clip(hamming,   0, nsubs_max).astype(int)
    d_clip = np.clip(datediffs, 0, ndays_max - 1).astype(int)
    probs  = prob_mat[h_clip, d_clip]

    infectors_dict[sid] = {
        strains[j]: (float(p), (int(h), int(d)))
        for j, p, h, d in zip(j_idxs, probs, hamming, datediffs)
    }

with open(OUT_PICKLE, "wb") as f:
    pickle.dump(infectors_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"\n  Saved: {OUT_PICKLE}")

# ── 6. Evaluate ────────────────────────────────────────────────────────────────
print("\nEvaluating ...")

true_parent = {row.strain: f"case_{int(row.parent_id)}" for _, row in gt.iterrows()}
leaf_set    = set(strains)

eval_cases  = [s for s in strains if true_parent.get(s, "case_-1") in leaf_set]

tp = fp = fn = 0
rows = []

for sid in eval_cases:
    true_p   = true_parent[sid]
    cands    = infectors_dict.get(sid)

    if cands:
        inferred = max(cands, key=lambda k: cands[k][0])
        correct  = (inferred == true_p)
        tp += correct
        fp += (not correct)
    else:
        inferred = None
        correct  = False
        fn += 1

    rows.append({"strain": sid, "true_parent": true_p,
                 "inferred_parent": inferred, "correct": correct})

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"  Saved: {OUT_CSV}")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1        = (2 * precision * recall / (precision + recall)
             if (precision + recall) > 0 else 0.0)

print(f"\n{'='*52}")
print(f"  Results  ({len(eval_cases)} evaluable pairs)")
print(f"{'='*52}")
print(f"  True  positives : {tp:>6}")
print(f"  False positives : {fp:>6}")
print(f"  False negatives : {fn:>6}")
print(f"  {'─'*40}")
print(f"  Precision       : {precision:.4f}")
print(f"  Recall          : {recall:.4f}")
print(f"  F1 score        : {f1:.4f}")
print(f"{'='*52}")
