"""
r0_sweep_ci.py
==============
Re-runs the R0 sweep with 7 different random seeds per R0 value to get
confidence intervals on precision, recall, and F1.

6 R0 values × 7 seeds = 42 runs. Saves after every run.

Outputs:
  - r0_sweep_ci_results.csv    one row per (r0, seed)
  - r0_sweep_ci_plot.png       mean ± 95% CI for precision/recall/F1
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
OUT_CSV = DISEASE / "r0_sweep_ci_results.csv"
OUT_PLOT= DISEASE / "r0_sweep_ci_plot.png"
TMP_DIR = DISEASE / "r0_ci_tmp"

TARGET_CASES = 10_000; K_DISP = 0.3
GT_MEAN = 4.87; GT_VAR = 1.98
GT_SHAPE = GT_MEAN**2 / GT_VAR; GT_SCALE = GT_VAR / GT_MEAN
MU = 0.091; GENOME_LEN = 29903; SCALE = MU / GENOME_LEN
SR_INFERENCE = 0.091; TIME_WIN = 14; MAX_HAMM = 2

R0_VALUES = [1.2, 1.5, 2.0, 2.5, 3.0, 4.0]
SEEDS     = [4, 42, 123, 456, 789, 1234, 9999,
             2, 7, 13, 21, 50, 77, 100, 200, 300, 500, 777, 2000, 5000]

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

# ── Epidemic simulation ────────────────────────────────────────────────────────
def simulate_epidemic(r0, seed):
    nb_p = K_DISP / (K_DISP + r0)
    rng  = np.random.default_rng(seed)
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
        kids = children[cid]; bl = max(times[cid] - times[parents.get(cid, -1)], 1e-8)
        if not kids: return f"case_{cid}:{bl:.6f}"
        return f"({','.join(newick(k) for k in kids)},case_{cid}:0.000001):{bl:.6f}"
    roots = children[-1]
    return (newick(roots[0]) + ";" if len(roots) == 1
            else "(" + ",".join(newick(r) for r in roots) + "):0.0;")

# ── Inference ──────────────────────────────────────────────────────────────────
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
    return p, r, f

# ── Load existing results ──────────────────────────────────────────────────────
if OUT_CSV.exists():
    existing = pd.read_csv(OUT_CSV)
    done     = set(zip(existing.r0, existing.seed))
    results  = existing.to_dict("records")
    print(f"Resuming — {len(done)}/42 runs done.")
else:
    done = set(); results = []

TMP_DIR.mkdir(exist_ok=True)
total = len(R0_VALUES) * len(SEEDS)
run_n = len(done)

# ── Main sweep ─────────────────────────────────────────────────────────────────
for r0 in R0_VALUES:
    for seed in SEEDS:
        if (r0, seed) in done:
            print(f"  skip R0={r0} seed={seed}")
            continue

        run_n += 1
        print(f"\n[{run_n}/{total}]  R0={r0}  seed={seed}")

        gt = simulate_epidemic(r0, seed)

        nwk_path = TMP_DIR / "tree.nwk"
        nwk_path.write_text(csv_to_newick(gt))

        run_dir = TMP_DIR / "phast"
        if run_dir.exists(): shutil.rmtree(run_dir)
        run_dir.mkdir()

        res = subprocess.run(
            ["phastSim", "--outpath", str(run_dir) + "/", "--reference", str(REF),
             "--treeFile", str(nwk_path), "--scale", str(SCALE),
             "--seed", str(seed), "--createFasta"],
            capture_output=True, text=True)
        if res.returncode != 0:
            print("  phastSim FAILED:", res.stderr[:200]); continue

        fasta = next(run_dir.glob("*.fasta"))
        prec, rec, f1 = run_inference(fasta, gt)
        shutil.rmtree(run_dir)

        print(f"  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")
        results.append({"r0": r0, "seed": seed,
                        "precision": prec, "recall": rec, "f1": f1})
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)

shutil.rmtree(TMP_DIR, ignore_errors=True)
print(f"\nAll done. {len(results)} runs saved to {OUT_CSV}")

# ── Plot ───────────────────────────────────────────────────────────────────────
df  = pd.read_csv(OUT_CSV)
agg = df.groupby("r0")[["precision", "recall", "f1"]].agg(["mean", "std"]).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
n_seeds_done = df.groupby("r0")["seed"].count().min()
fig.suptitle(f"Model performance vs R0  (mean ± 95% CI across {n_seeds_done} epidemic realisations)",
             fontsize=12, fontweight="bold")

for ax, metric, color, label in [
    (axes[0], "precision", "steelblue",  "Precision"),
    (axes[1], "recall",    "darkorange", "Recall"),
    (axes[2], "f1",        "seagreen",   "F1 score"),
]:
    mean = agg[(metric, "mean")]
    ci   = 1.96 * agg[(metric, "std")] / np.sqrt(len(SEEDS))
    r0s  = agg["r0"]

    ax.plot(r0s, mean, marker="o", linewidth=2, color=color)
    ax.fill_between(r0s, mean - ci, mean + ci, alpha=0.25, color=color)
    ax.axvline(2.0, color="grey", linestyle="--", linewidth=1.2, label="True R0 (2.0)")
    ax.set_xlabel("R0", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {OUT_PLOT}")
