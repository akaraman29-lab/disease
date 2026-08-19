"""
generation_time_sweep.py
========================
Varies the true generation time mean (Gamma distribution) in the epidemic
simulation while keeping all inference model parameters fixed.

Pipeline per generation time:
  1. Simulate epidemic  → ground_truth
  2. Build Newick tree
  3. Run phastSim       → sequences (fixed scale = 0.091 / 29903)
  4. Run inference      → precision, recall, F1

GT_VAR is held fixed at 1.98 (from Denmark paper).
Inference model's assumed generation time is also held fixed.

Outputs:
  - generation_time_sweep_results.csv
  - generation_time_sweep.png
"""

import sys, subprocess, shutil, heapq
import numpy as np
import pandas as pd
import regex
from Bio import SeqIO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DISEASE = Path("/Users/azrakaraman/Desktop/disease")
TF_DIR  = DISEASE / "Danish_Transmission" / "src" / "network_construction"
sys.path.insert(0, str(TF_DIR))
import transmission_functions as tf  # type: ignore

REF     = Path("/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/phastSim/example/MN908947.3.fasta")
OUT_CSV = DISEASE / "generation_time_sweep_results.csv"
OUT_PLOT= DISEASE / "generation_time_sweep.png"
TMP_DIR = DISEASE / "gen_time_sweep_tmp"

# Fixed parameters
SEED         = 4
TARGET_CASES = 10_000
R0           = 2.0
K_DISP       = 0.3
GT_VAR       = 1.98          # variance held fixed
MU           = 0.091         # substitution rate (fixed in inference)
GENOME_LEN   = 29903
SCALE        = MU / GENOME_LEN
SR_INFERENCE = 0.091
TIME_WIN     = 14
MAX_HAMM     = 2

# Generation time means to sweep (days)
GT_MEANS = [2.0, 3.0, 4.0, 4.87, 6.0, 8.0, 10.0, 12.0]

# ── Precompute inference lookup table (fixed) ──────────────────────────────────
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

# ── Step 1: Epidemic simulation ────────────────────────────────────────────────
def simulate_epidemic(gt_mean):
    gt_shape = gt_mean**2 / GT_VAR
    gt_scale = GT_VAR / gt_mean
    nb_p     = K_DISP / (K_DISP + R0)
    rng      = np.random.default_rng(SEED)

    heap    = [(0.0, 0, -1)]
    next_id = 1
    records = []

    while heap and len(records) < TARGET_CASES:
        inf_time, case_id, parent_id = heapq.heappop(heap)
        records.append({"case_id": case_id, "parent_id": parent_id,
                        "infection_time": round(inf_time, 4)})
        n_off = int(rng.negative_binomial(K_DISP, nb_p))
        for _ in range(n_off):
            gt  = float(rng.gamma(gt_shape, gt_scale))
            heapq.heappush(heap, (inf_time + gt, next_id, case_id))
            next_id += 1

    df = pd.DataFrame(records)
    depth = {-1: -1}
    for _, row in df.sort_values("infection_time").iterrows():
        depth[row.case_id] = depth[row.parent_id] + 1
    df["generation"] = df.case_id.map(depth)
    df["sample_time"] = df["infection_time"]   # no incubation (matches without_inc.py)
    return df

# ── Step 2: Build Newick tree ──────────────────────────────────────────────────
def csv_to_newick(df):
    children = {-1: []}
    for cid in df.case_id:
        children[int(cid)] = []
    for _, row in df.iterrows():
        children[int(row.parent_id)].append(int(row.case_id))
    times   = dict(zip(df.case_id.astype(int), df.infection_time))
    times[-1] = 0.0
    parents = dict(zip(df.case_id.astype(int), df.parent_id.astype(int)))

    def newick(cid):
        kids   = children[cid]
        parent = parents.get(cid, -1)
        bl     = max(times[cid] - times[parent], 1e-8)
        if not kids:
            return f"case_{cid}:{bl:.6f}"
        child_strs = ",".join(newick(k) for k in kids)
        self_leaf  = f"case_{cid}:0.000001"
        return f"({child_strs},{self_leaf}):{bl:.6f}"

    roots = children[-1]
    return (newick(roots[0]) + ";" if len(roots) == 1
            else "(" + ",".join(newick(r) for r in roots) + "):0.0;")

# ── Step 3: Run inference ──────────────────────────────────────────────────────
def clean_seq(seq):
    return regex.sub(r"[^ACTG]", "-", str(seq).upper())

def run_inference(fasta_path, gt):
    true_parent = {f"case_{int(row.case_id)}": f"case_{int(row.parent_id)}"
                   for _, row in gt.iterrows()}
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
        h  = np.sum(seqs_var[jm] != seqs_var[i], axis=1)
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
    return p, r, f, int(var_mask.sum())

# ── Main sweep ─────────────────────────────────────────────────────────────────
TMP_DIR.mkdir(exist_ok=True)
results = []

for gt_mean in GT_MEANS:
    gt_shape = gt_mean**2 / GT_VAR
    gt_scale = GT_VAR / gt_mean
    print(f"\n{'─'*60}")
    print(f"  GT mean={gt_mean:.2f} days  (shape={gt_shape:.2f}, scale={gt_scale:.3f})")

    # 1. Simulate epidemic
    print("  Simulating epidemic ...")
    gt = simulate_epidemic(gt_mean)
    print(f"    {len(gt)} cases | {gt.infection_time.max():.1f} days | "
          f"max generation {gt.generation.max()}")

    # 2. Build + save Newick
    print("  Building Newick tree ...")
    nwk_path = TMP_DIR / "tree.nwk"
    nwk_path.write_text(csv_to_newick(gt))

    # 3. Run phastSim
    run_dir = TMP_DIR / "phast"
    if run_dir.exists(): shutil.rmtree(run_dir)
    run_dir.mkdir()
    print("  Running phastSim ...")
    res = subprocess.run(
        ["phastSim", "--outpath", str(run_dir) + "/", "--reference", str(REF),
         "--treeFile", str(nwk_path), "--scale", str(SCALE),
         "--seed", str(SEED), "--createFasta"],
        capture_output=True, text=True)
    if res.returncode != 0:
        print("  phastSim FAILED:", res.stderr[:200]); continue
    fasta = next(run_dir.glob("*.fasta"))

    # 4. Run inference
    print("  Running inference ...")
    prec, rec, f1, nv = run_inference(fasta, gt)
    shutil.rmtree(run_dir)
    print(f"  var_sites={nv:,}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")

    results.append({"gt_mean": gt_mean, "gt_shape": gt_shape, "gt_scale": gt_scale,
                    "n_variable_sites": nv, "precision": prec, "recall": rec, "f1": f1})
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)

shutil.rmtree(TMP_DIR)
print(f"\nSweep complete. Saved: {OUT_CSV}")

# ── Print table ────────────────────────────────────────────────────────────────
df = pd.read_csv(OUT_CSV)
print(df[["gt_mean", "n_variable_sites", "precision", "recall", "f1"]].to_string(index=False))

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Model performance vs true generation time mean\n"
             "(inference model parameters fixed at Denmark paper values)",
             fontsize=12, fontweight="bold")

TRUE_GT = 4.87
for ax, metric, color, label in [
    (axes[0], "recall", "steelblue",  "Recall"),
    (axes[1], "f1",     "darkorange", "F1 score"),
]:
    ax.plot(df.gt_mean, df[metric], marker="o", linewidth=2, markersize=7, color=color)
    ax.axvline(TRUE_GT, color="grey", linestyle="--", linewidth=1.2,
               label=f"True GT (4.87 days)")
    peak_idx = df[metric].idxmax()
    ax.axvline(df.loc[peak_idx, "gt_mean"], color="crimson", linestyle=":",
               linewidth=1.2, label=f"Peak = {df.loc[peak_idx, 'gt_mean']} days")
    ax.set_xlabel("True generation time mean (days)", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {OUT_PLOT}")
