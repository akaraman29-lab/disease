"""
Runs inference for mu=0.5 and mu=1.0 only, appends to existing CSV, then plots.
"""

import sys, subprocess, shutil, numpy as np, pandas as pd, regex
from Bio import SeqIO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DISEASE = Path("/Users/azrakaraman/Desktop/disease")
TF_DIR  = DISEASE / "Danish_Transmission" / "src" / "network_construction"
sys.path.insert(0, str(TF_DIR))
import transmission_functions as tf  # type: ignore

REF     = Path("/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/phastSim/example/MN908947.3.fasta")
NWK     = DISEASE / "transmission_tree.nwk"
GT_PATH = DISEASE / "ground_truth.csv"
OUT_CSV = DISEASE / "mutation_rate_sweep_results.csv"
OUT_PLOT= DISEASE / "mutation_rate_sweep.png"

GENOME_LEN=29903; SR_INFERENCE=0.091; TIME_WIN=14; MAX_HAMM=2; SEED=4

# ── Ground truth ───────────────────────────────────────────────────────────────
gt = pd.read_csv(GT_PATH)
gt["strain"] = "case_" + gt["case_id"].astype(str)
true_parent = {row.strain: f"case_{int(row.parent_id)}" for _, row in gt.iterrows()}

# ── Precompute lookup table ────────────────────────────────────────────────────
ndays_max=21; nsubs_max=9
prob_mat = np.zeros((nsubs_max+1, ndays_max))
for s in range(nsubs_max+1):
    for d in range(ndays_max):
        prob_mat[s,d] = tf.scenario1_probability(d, s, SR_INFERENCE, shift=0)
valid_idxs = {(0,0)}
for d in range(ndays_max):
    cum = np.cumsum(tf.substitution_probability(np.arange(nsubs_max+1), d, SR_INFERENCE))
    valid_idxs |= {(int(j),d) for j in np.argwhere(cum<=0.95).flatten()}

def clean_seq(seq):
    return regex.sub(r"[^ACTG]", "-", str(seq).upper())

def run_inference(fasta_path):
    raw = {r.id: str(r.seq) for r in SeqIO.parse(fasta_path, "fasta")}
    gt_leaves = (gt[gt.strain.isin(set(raw.keys()))].copy()
                   .sort_values("sample_time").reset_index(drop=True))
    seqs_np  = np.stack([np.frombuffer(clean_seq(raw[s]).encode(), dtype=np.uint8)
                         for s in gt_leaves.strain])
    var_mask = np.any(seqs_np != seqs_np[0], axis=0)
    seqs_var = seqs_np[:, var_mask]
    strains  = gt_leaves.strain.values
    sdays    = np.round(gt_leaves.sample_time.values).astype(int)
    eval_cases = [s for s in strains if true_parent.get(s, "case_-1") in set(strains)]
    infectors = {}
    for i in range(len(strains)):
        sid=strains[i]; t_i=sdays[i]
        jm=np.where((sdays<t_i)&(sdays>=t_i-TIME_WIN))[0]
        if not len(jm): infectors[sid]=None; continue
        h=np.sum(seqs_var[jm]!=seqs_var[i],axis=1)
        jm=jm[h<=MAX_HAMM]; h=h[h<=MAX_HAMM].astype(int)
        if not len(jm): infectors[sid]=None; continue
        dd=(t_i-sdays[jm]).astype(int)
        vm=np.array([(int(hh),int(d)) in valid_idxs for hh,d in zip(h,dd)])
        if not vm.any(): infectors[sid]=None; continue
        jm=jm[vm]; h=h[vm]; dd=dd[vm]
        probs=prob_mat[np.clip(h,0,nsubs_max).astype(int), np.clip(dd,0,ndays_max-1).astype(int)]
        infectors[sid]={strains[j]:float(p) for j,p in zip(jm,probs)}
    tp=fp=fn=0; n=len(eval_cases)
    for sid in eval_cases:
        tp_=true_parent[sid]; c=infectors.get(sid)
        if c:
            if max(c, key=lambda k: c[k])==tp_: tp+=1
            else: fp+=1
        else: fn+=1
    p=tp/(tp+fp) if (tp+fp)>0 else 0.0
    r=tp/n       if n>0       else 0.0
    f=2*p*r/(p+r) if (p+r)>0 else 0.0
    return p, r, f, int(var_mask.sum())

# ── Run missing rates ──────────────────────────────────────────────────────────
existing = pd.read_csv(OUT_CSV)
results  = existing.to_dict("records")
done     = set(existing.mu_sim)

for mu in [0.125, 0.175, 0.2, 0.3, 0.6, 0.75]:
    if mu in done:
        print(f"mu={mu} already done, skipping.")
        continue
    scale = mu / GENOME_LEN
    print(f"\nmu={mu}  scale={scale:.3e}")
    run_dir = DISEASE / "sweep_tmp"
    if run_dir.exists(): shutil.rmtree(run_dir)
    run_dir.mkdir()
    print("  Running phastSim ...")
    res = subprocess.run(
        ["phastSim","--outpath",str(run_dir)+"/","--reference",str(REF),
         "--treeFile",str(NWK),"--scale",str(scale),"--seed",str(SEED),"--createFasta"],
        capture_output=True, text=True)
    if res.returncode != 0:
        print("  phastSim FAILED:", res.stderr[:300]); continue
    fasta = next(run_dir.glob("*.fasta"))
    print("  Running inference ...")
    prec, rec, f1, nv = run_inference(fasta)
    shutil.rmtree(run_dir)
    print(f"  variable sites={nv:,}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")
    results.append({"mu_sim":mu,"scale":scale,"n_variable_sites":nv,
                    "precision":prec,"recall":rec,"f1":f1})
    pd.DataFrame(results).sort_values("mu_sim").to_csv(OUT_CSV, index=False)

# ── Print full table ───────────────────────────────────────────────────────────
df = pd.read_csv(OUT_CSV).sort_values("mu_sim")
print("\nFull results:")
print(df[["mu_sim","n_variable_sites","precision","recall","f1"]].to_string(index=False))

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"Model performance vs simulated mutation rate\n"
    f"(inference SR fixed at {SR_INFERENCE} subs/genome/day)",
    fontsize=12, fontweight="bold")

for ax, metric, color, label in [
    (axes[0], "recall",    "steelblue",  "Recall"),
    (axes[1], "f1",        "darkorange", "F1 score"),
]:
    ax.plot(df.mu_sim, df[metric], marker="o", linewidth=2, markersize=7, color=color)
    ax.axvline(SR_INFERENCE, color="grey", linestyle="--", linewidth=1.2, label="True rate (0.091)")
    ax.set_xlabel("Simulated mutation rate (subs/genome/day)", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

ax2 = axes[0].twiny()
ax2.set_xlim(axes[0].get_xlim())
ax2.set_xscale("log")
ax2.set_xticks(df.mu_sim.values)
ax2.set_xticklabels([f"{int(r):,}" for r in df.n_variable_sites.values], fontsize=7, rotation=45)
ax2.set_xlabel("Variable sites in simulated sequences", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {OUT_PLOT}")
plt.show()
