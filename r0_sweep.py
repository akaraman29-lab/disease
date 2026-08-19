"""
r0_sweep.py
===========
Varies R0 (transmission rate) in the epidemic simulation while keeping
all inference model parameters fixed.

Higher R0 → more secondary cases → more candidates per case → more false
positives and lower precision. Recall may stay relatively stable.

Outputs:
  - r0_sweep_results.csv
  - r0_sweep.png
"""

import sys, subprocess, shutil, heapq
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

REF     = Path("/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/phastSim/example/MN908947.3.fasta")
OUT_CSV = DISEASE / "r0_sweep_results.csv"
OUT_PLOT= DISEASE / "r0_sweep.png"
TMP_DIR = DISEASE / "r0_sweep_tmp"

# Fixed parameters
SEED         = 4
TARGET_CASES = 10_000
K_DISP       = 0.3          # overdispersion fixed
GT_MEAN      = 4.87         # generation time fixed (Denmark paper)
GT_VAR       = 1.98
GT_SHAPE     = GT_MEAN**2 / GT_VAR
GT_SCALE     = GT_VAR / GT_MEAN
MU           = 0.091
GENOME_LEN   = 29903
SCALE        = MU / GENOME_LEN
SR_INFERENCE = 0.091
TIME_WIN     = 14
MAX_HAMM     = 2

# R0 values to sweep
R0_VALUES = [1.2, 1.5, 2.0, 2.5, 3.0, 4.0]

# ── Precompute inference lookup table ──────────────────────────────────────────
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

# ── Epidemic simulation ────────────────────────────────────────────────────────
def simulate_epidemic(r0):
    nb_p = K_DISP / (K_DISP + r0)
    rng  = np.random.default_rng(SEED)
    heap = [(0.0, 0, -1)]; next_id = 1; records = []
    while heap and len(records) < TARGET_CASES:
        inf_time, case_id, parent_id = heapq.heappop(heap)
        records.append({"case_id": case_id, "parent_id": parent_id,
                        "infection_time": round(inf_time, 4)})
        for _ in range(int(rng.negative_binomial(K_DISP, nb_p))):
            heapq.heappush(heap, (inf_time + float(rng.gamma(GT_SHAPE, GT_SCALE)),
                                  next_id, case_id))
            next_id += 1
    df = pd.DataFrame(records)
    depth = {-1: -1}
    for _, row in df.sort_values("infection_time").iterrows():
        depth[row.case_id] = depth[row.parent_id] + 1
    df["generation"]  = df.case_id.map(depth)
    df["sample_time"] = df["infection_time"]
    return df

# ── Newick builder ─────────────────────────────────────────────────────────────
def csv_to_newick(df):
    children = {-1: []}
    for cid in df.case_id: children[int(cid)] = []
    for _, row in df.iterrows():
        children[int(row.parent_id)].append(int(row.case_id))
    times = dict(zip(df.case_id.astype(int), df.infection_time)); times[-1] = 0.0
    parents = dict(zip(df.case_id.astype(int), df.parent_id.astype(int)))
    def newick(cid):
        kids = children[cid]; bl = max(times[cid] - times[parents.get(cid,-1)], 1e-8)
        if not kids: return f"case_{cid}:{bl:.6f}"
        return f"({','.join(newick(k) for k in kids)},case_{cid}:0.000001):{bl:.6f}"
    roots = children[-1]
    return (newick(roots[0]) + ";" if len(roots) == 1
            else "(" + ",".join(newick(r) for r in roots) + "):0.0;")

# ── Inference ──────────────────────────────────────────────────────────────────
def clean_seq(seq):
    return regex.sub(r"[^ACTG]", "-", str(seq).upper())

def run_inference(fasta_path, gt):
    true_parent = {f"case_{int(r.case_id)}": f"case_{int(r.parent_id)}"
                   for _, r in gt.iterrows()}
    raw = {r.id: str(r.seq) for r in SeqIO.parse(fasta_path, "fasta")}
    gt_leaves = (gt[gt.apply(lambda r: f"case_{int(r.case_id)}" in raw, axis=1)]
                   .copy().sort_values("sample_time").reset_index(drop=True))
    gt_leaves["strain"] = "case_" + gt_leaves.case_id.astype(str)
    seqs_np  = np.stack([np.frombuffer(clean_seq(raw[s]).encode(), dtype=np.uint8)
                         for s in gt_leaves.strain])
    var_mask = np.any(seqs_np != seqs_np[0], axis=0)
    seqs_var = seqs_np[:, var_mask]
    strains  = gt_leaves.strain.values
    sdays    = np.round(gt_leaves.sample_time.values).astype(int)
    leaf_set = set(strains)
    eval_cases = [s for s in strains if true_parent.get(s, "case_-1") in leaf_set]

    infectors = {}; tp_all = 0; fp_all = 0; fn_all = 0
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
                         np.clip(dd, 0, ndays_max-1).astype(int)]
        infectors[sid] = {strains[j]: float(p) for j, p in zip(jm, probs)}

    tp = fp = fn = 0; n = len(eval_cases)
    n_cands = []
    for sid in eval_cases:
        tp_ = true_parent[sid]; c = infectors.get(sid)
        if c:
            n_cands.append(len(c))
            if max(c, key=lambda k: c[k]) == tp_: tp += 1
            else: fp += 1
        else:
            n_cands.append(0); fn += 1

    p   = tp/(tp+fp)   if (tp+fp)>0 else 0.0
    r   = tp/n         if n>0       else 0.0
    f   = 2*p*r/(p+r)  if (p+r)>0  else 0.0
    med_cands = float(np.median(n_cands))
    return p, r, f, tp, fp, fn, med_cands

# ── Main sweep ─────────────────────────────────────────────────────────────────
if OUT_CSV.exists():
    done_df  = pd.read_csv(OUT_CSV)
    done     = set(done_df.r0)
    results  = done_df.to_dict("records")
    print(f"Resuming — {len(done)} values done.")
else:
    done = set(); results = []

TMP_DIR.mkdir(exist_ok=True)

for r0 in R0_VALUES:
    if r0 in done:
        print(f"  skip R0={r0} (done)"); continue

    print(f"\n{'─'*55}")
    print(f"  R0={r0}  (k={K_DISP}  →  mean offspring = {r0:.1f})")

    gt = simulate_epidemic(r0)
    print(f"  Epidemic: {len(gt)} cases | {gt.infection_time.max():.1f} days")

    nwk_path = TMP_DIR / "tree.nwk"
    nwk_path.write_text(csv_to_newick(gt))

    run_dir = TMP_DIR / "phast"
    if run_dir.exists(): shutil.rmtree(run_dir)
    run_dir.mkdir()
    print("  Running phastSim ...")
    res = subprocess.run(
        ["phastSim","--outpath",str(run_dir)+"/","--reference",str(REF),
         "--treeFile",str(nwk_path),"--scale",str(SCALE),
         "--seed",str(SEED),"--createFasta"],
        capture_output=True, text=True)
    if res.returncode != 0:
        print("  phastSim FAILED:", res.stderr[:200]); continue
    fasta = next(run_dir.glob("*.fasta"))

    print("  Running inference ...")
    prec, rec, f1, tp, fp, fn, med_cands = run_inference(fasta, gt)
    shutil.rmtree(run_dir)
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}  median_candidates={med_cands:.0f}")

    results.append({"r0": r0, "precision": prec, "recall": rec, "f1": f1,
                    "tp": tp, "fp": fp, "fn": fn, "median_candidates": med_cands})
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)

shutil.rmtree(TMP_DIR, ignore_errors=True)
print(f"\nSweep complete. Saved: {OUT_CSV}")

# ── Plot ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(OUT_CSV).sort_values("r0")
print(df[["r0","tp","fp","fn","precision","recall","f1","median_candidates"]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Model performance vs R0\n"
             "(mutation rate, generation time, and inference model fixed)",
             fontsize=12, fontweight="bold")

# Left: precision, recall, F1
ax = axes[0]
ax.plot(df.r0, df.precision, marker="o", linewidth=2, color="steelblue",  label="Precision")
ax.plot(df.r0, df.recall,    marker="s", linewidth=2, color="darkorange", label="Recall")
ax.plot(df.r0, df.f1,        marker="^", linewidth=2, color="seagreen",   label="F1")
ax.axvline(2.0, color="grey", linestyle="--", linewidth=1.2, label="True R0 (2.0)")
ax.set_xlabel("R0", fontsize=11); ax.set_ylabel("Score", fontsize=11)
ax.set_title("Precision / Recall / F1", fontsize=11)
ax.set_ylim(0, 1); ax.legend(fontsize=9)

# Right: median candidate set size (the haystack)
ax = axes[1]
ax.plot(df.r0, df.median_candidates, marker="o", linewidth=2, color="crimson")
ax.axvline(2.0, color="grey", linestyle="--", linewidth=1.2, label="True R0 (2.0)")
ax.set_xlabel("R0", fontsize=11); ax.set_ylabel("Median candidates per case", fontsize=11)
ax.set_title("Haystack size vs R0", fontsize=11)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {OUT_PLOT}")
