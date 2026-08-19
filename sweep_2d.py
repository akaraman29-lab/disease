"""
sweep_2d.py
===========
2D sweep of simulated mutation rate × generation time mean.
Inference model parameters are fixed throughout.

Grid:
  mu      : [0.025, 0.05, 0.091, 0.2, 0.5]  subs/genome/day
  gt_mean : [3.0, 4.87, 6.0, 8.0, 10.0]     days

25 combinations total. Results saved after each cell.

Outputs:
  - sweep_2d_results.csv
  - sweep_2d_heatmap.png    F1 heatmap with contours of constant mu × gt_mean
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
OUT_CSV = DISEASE / "sweep_2d_results.csv"
OUT_PLOT= DISEASE / "sweep_2d_heatmap.png"
TMP_DIR = DISEASE / "sweep_2d_tmp"

# Fixed parameters
SEED         = 4
TARGET_CASES = 10_000
R0           = 2.0
K_DISP       = 0.3
GT_VAR       = 1.98
GENOME_LEN   = 29903
SR_INFERENCE = 0.091
TIME_WIN     = 14
MAX_HAMM     = 2

# Grid
MU_VALUES  = [0.025, 0.05, 0.091, 0.2, 0.5]
GT_MEANS   = [3.0, 4.87, 6.0, 8.0, 10.0]

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

# ── Epidemic simulation ────────────────────────────────────────────────────────
def simulate_epidemic(gt_mean):
    gt_shape = gt_mean**2 / GT_VAR
    gt_scale = GT_VAR / gt_mean
    nb_p     = K_DISP / (K_DISP + R0)
    rng      = np.random.default_rng(SEED)
    heap = [(0.0, 0, -1)]; next_id = 1; records = []
    while heap and len(records) < TARGET_CASES:
        inf_time, case_id, parent_id = heapq.heappop(heap)
        records.append({"case_id": case_id, "parent_id": parent_id,
                        "infection_time": round(inf_time, 4)})
        for _ in range(int(rng.negative_binomial(K_DISP, nb_p))):
            heapq.heappush(heap, (inf_time + float(rng.gamma(gt_shape, gt_scale)),
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
                         np.clip(dd, 0, ndays_max-1).astype(int)]
        infectors[sid] = {strains[j]: float(p) for j, p in zip(jm, probs)}
    tp = fp = fn = 0; n = len(eval_cases)
    for sid in eval_cases:
        tp_ = true_parent[sid]; c = infectors.get(sid)
        if c:
            if max(c, key=lambda k: c[k]) == tp_: tp += 1
            else: fp += 1
        else: fn += 1
    p = tp/(tp+fp) if (tp+fp)>0 else 0.0
    r = tp/n       if n>0       else 0.0
    f = 2*p*r/(p+r) if (p+r)>0 else 0.0
    return p, r, f

# ── Load existing results to skip completed cells ──────────────────────────────
if OUT_CSV.exists():
    done_df = pd.read_csv(OUT_CSV)
    done = set(zip(done_df.mu_sim, done_df.gt_mean))
    results = done_df.to_dict("records")
    print(f"Resuming — {len(done)} cells already done.")
else:
    done = set(); results = []

TMP_DIR.mkdir(exist_ok=True)
total = len(MU_VALUES) * len(GT_MEANS)
cell  = len(done)

# ── Main sweep ─────────────────────────────────────────────────────────────────
for mu in MU_VALUES:
    scale = mu / GENOME_LEN
    for gt_mean in GT_MEANS:
        if (mu, gt_mean) in done:
            print(f"  skip mu={mu} gt={gt_mean} (done)")
            continue

        cell += 1
        print(f"\n[{cell}/{total}] mu={mu}  gt_mean={gt_mean} days")

        # 1. Epidemic
        gt = simulate_epidemic(gt_mean)

        # 2. Newick
        nwk_path = TMP_DIR / "tree.nwk"
        nwk_path.write_text(csv_to_newick(gt))

        # 3. phastSim
        run_dir = TMP_DIR / "phast"
        if run_dir.exists(): shutil.rmtree(run_dir)
        run_dir.mkdir()
        res = subprocess.run(
            ["phastSim","--outpath",str(run_dir)+"/","--reference",str(REF),
             "--treeFile",str(nwk_path),"--scale",str(scale),
             "--seed",str(SEED),"--createFasta"],
            capture_output=True, text=True)
        if res.returncode != 0:
            print("  phastSim FAILED:", res.stderr[:200]); continue
        fasta = next(run_dir.glob("*.fasta"))

        # 4. Inference
        prec, rec, f1 = run_inference(fasta, gt)
        shutil.rmtree(run_dir)
        print(f"  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")

        results.append({"mu_sim": mu, "gt_mean": gt_mean, "subs_per_transmission": mu * gt_mean,
                        "precision": prec, "recall": rec, "f1": f1})
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)

shutil.rmtree(TMP_DIR, ignore_errors=True)
print(f"\nSweep complete. Saved: {OUT_CSV}")

# ── Plot heatmap ───────────────────────────────────────────────────────────────
df  = pd.read_csv(OUT_CSV)
mu_vals = sorted(df.mu_sim.unique())
gt_vals = sorted(df.gt_mean.unique())

# Pivot to matrix (rows=gt_mean, cols=mu_sim)
f1_mat    = df.pivot(index="gt_mean", columns="mu_sim", values="f1").values
rec_mat   = df.pivot(index="gt_mean", columns="mu_sim", values="recall").values
prec_mat  = df.pivot(index="gt_mean", columns="mu_sim", values="precision").values

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("2D sweep: simulated mutation rate × generation time\n"
             "(inference model fixed at Denmark paper values)",
             fontsize=12, fontweight="bold")

for ax, mat, title in [
    (axes[0], f1_mat,   "F1 score"),
    (axes[1], rec_mat,  "Recall"),
    (axes[2], prec_mat, "Precision"),
]:
    im = ax.imshow(mat, aspect="auto", origin="lower",
                   cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    # Annotate cells
    for i in range(len(gt_vals)):
        for j in range(len(mu_vals)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if mat[i,j] < 0.7 else "white")

    # Contours of constant mu × gt_mean (subs per transmission)
    mu_grid = np.array(mu_vals)
    gt_grid = np.array(gt_vals)
    MU_G, GT_G = np.meshgrid(np.linspace(0, len(mu_vals)-1, 200),
                              np.linspace(0, len(gt_vals)-1, 200))
    MU_V = np.interp(MU_G.flatten(), np.arange(len(mu_vals)), mu_vals).reshape(MU_G.shape)
    GT_V = np.interp(GT_G.flatten(), np.arange(len(gt_vals)), gt_vals).reshape(GT_G.shape)
    SPT  = MU_V * GT_V   # subs per transmission
    ax.contour(MU_G, GT_G, SPT, levels=[0.2, 0.5, 1.0, 2.0],
               colors="white", linewidths=0.8, linestyles="--", alpha=0.7)

    ax.set_xticks(range(len(mu_vals)))
    ax.set_xticklabels([str(m) for m in mu_vals], fontsize=9)
    ax.set_yticks(range(len(gt_vals)))
    ax.set_yticklabels([str(g) for g in gt_vals], fontsize=9)
    ax.set_xlabel("Mutation rate (subs/genome/day)", fontsize=10)
    ax.set_ylabel("Generation time mean (days)", fontsize=10)
    ax.set_title(title, fontsize=11)

    # Mark true SARS-CoV-2 values
    true_mu_idx = mu_vals.index(0.091) if 0.091 in mu_vals else None
    true_gt_idx = gt_vals.index(4.87)  if 4.87  in gt_vals else None
    if true_mu_idx is not None and true_gt_idx is not None:
        ax.plot(true_mu_idx, true_gt_idx, "w*", markersize=14,
                markeredgecolor="black", markeredgewidth=0.8, label="SARS-CoV-2")
        ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {OUT_PLOT}")
