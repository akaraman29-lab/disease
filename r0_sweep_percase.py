"""
r0_sweep_percase.py
===================
Re-runs the R0 sweep but saves per-case outcomes for each R0 value.

For each case records:
  - outcome       : "tp" (correct), "fp" (wrong guess), "fn" (no candidates)
  - n_candidates  : size of candidate set
  - rank_true_parent : rank of true parent (1-based, NaN if absent)

Outputs:
  - r0_percase_results.csv   (~60,000 rows)
  - r0_percase_plot.png
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
OUT_CSV = DISEASE / "r0_percase_results.csv"
OUT_PLOT= DISEASE / "r0_percase_plot.png"
TMP_DIR = DISEASE / "r0_percase_tmp"

SEED=4; TARGET_CASES=10_000; K_DISP=0.3
GT_MEAN=4.87; GT_VAR=1.98
GT_SHAPE=GT_MEAN**2/GT_VAR; GT_SCALE=GT_VAR/GT_MEAN
MU=0.091; GENOME_LEN=29903; SCALE=MU/GENOME_LEN
SR_INFERENCE=0.091; TIME_WIN=14; MAX_HAMM=2

R0_VALUES = [1.2, 1.5, 2.0, 2.5, 3.0, 4.0]

# ── Precompute inference table ─────────────────────────────────────────────────
print("Precomputing inference probability table ...")
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

# ── Epidemic simulation ────────────────────────────────────────────────────────
def simulate_epidemic(r0):
    nb_p = K_DISP / (K_DISP + r0)
    rng  = np.random.default_rng(SEED)
    heap = [(0.0, 0, -1)]; next_id=1; records=[]
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
    times = dict(zip(df.case_id.astype(int), df.infection_time)); times[-1]=0.0
    parents = dict(zip(df.case_id.astype(int), df.parent_id.astype(int)))
    def newick(cid):
        kids = children[cid]; bl = max(times[cid]-times[parents.get(cid,-1)], 1e-8)
        if not kids: return f"case_{cid}:{bl:.6f}"
        return f"({','.join(newick(k) for k in kids)},case_{cid}:0.000001):{bl:.6f}"
    roots = children[-1]
    return (newick(roots[0])+";" if len(roots)==1
            else "("+",".join(newick(r) for r in roots)+"):0.0;")

# ── Inference with per-case logging ───────────────────────────────────────────
def run_inference_percase(fasta_path, gt, r0):
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

    # Build infectors dict
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
        probs=prob_mat[np.clip(h,0,nsubs_max).astype(int),
                       np.clip(dd,0,ndays_max-1).astype(int)]
        infectors[sid]={strains[j]:float(p) for j,p in zip(jm,probs)}

    # Per-case evaluation
    rows = []
    for sid in eval_cases:
        tp_   = true_parent[sid]
        cands = infectors.get(sid)

        if cands:
            ranked = sorted(cands, key=lambda k: cands[k], reverse=True)
            inferred = ranked[0]
            outcome  = "tp" if inferred == tp_ else "fp"
            n_cands  = len(cands)
            rank     = (ranked.index(tp_) + 1) if tp_ in ranked else np.nan
        else:
            outcome = "fn"
            n_cands = 0
            rank    = np.nan

        rows.append({"r0": r0, "case_id": sid, "outcome": outcome,
                     "n_candidates": n_cands, "rank_true_parent": rank})

    return pd.DataFrame(rows)

# ── Main sweep ─────────────────────────────────────────────────────────────────
if OUT_CSV.exists():
    existing = pd.read_csv(OUT_CSV)
    done     = set(existing.r0.unique())
    all_rows = [existing]
    print(f"Resuming — R0 values done: {sorted(done)}")
else:
    done = set(); all_rows = []

TMP_DIR.mkdir(exist_ok=True)

for r0 in R0_VALUES:
    if r0 in done:
        print(f"  skip R0={r0}"); continue

    print(f"\n{'─'*55}")
    print(f"  R0={r0}")

    gt = simulate_epidemic(r0)
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
    df_case = run_inference_percase(fasta, gt, r0)
    shutil.rmtree(run_dir)

    counts = df_case.outcome.value_counts()
    print(f"  TP={counts.get('tp',0)}  FP={counts.get('fp',0)}  FN={counts.get('fn',0)}")

    all_rows.append(df_case)
    pd.concat(all_rows).to_csv(OUT_CSV, index=False)
    print(f"  Saved progress ({len(pd.concat(all_rows))} rows total)")

shutil.rmtree(TMP_DIR, ignore_errors=True)

# ── Load and verify ────────────────────────────────────────────────────────────
df = pd.read_csv(OUT_CSV)
print(f"\nTotal rows: {len(df)}")
print("\nOutcome counts per R0:")
print(df.groupby(["r0","outcome"]).size().unstack(fill_value=0).to_string())

# Cross-check against existing aggregate results
agg = pd.read_csv(DISEASE / "r0_sweep_results.csv")
print("\nCross-check vs r0_sweep_results.csv:")
for _, row in agg.iterrows():
    sub = df[df.r0 == row.r0].outcome.value_counts()
    print(f"  R0={row.r0}: sweep TP={int(row.tp)} FP={int(row.fp)} FN={int(row.fn)}  |  "
          f"percase TP={sub.get('tp',0)} FP={sub.get('fp',0)} FN={sub.get('fn',0)}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Per-case outcome distribution across R0 values", fontsize=12, fontweight="bold")

r0_vals = sorted(df.r0.unique())

# Panel A — stacked bar of TP/FP/FN proportions
ax = axes[0]
tp_pct = []; fp_pct = []; fn_pct = []
for r0 in r0_vals:
    sub = df[df.r0 == r0]
    n   = len(sub)
    tp_pct.append(100 * (sub.outcome=="tp").sum() / n)
    fp_pct.append(100 * (sub.outcome=="fp").sum() / n)
    fn_pct.append(100 * (sub.outcome=="fn").sum() / n)

x = np.arange(len(r0_vals))
w = 0.5
b1 = ax.bar(x, tp_pct, w, label="TP (correct)",   color="#2ca02c")
b2 = ax.bar(x, fp_pct, w, bottom=tp_pct,           label="FP (wrong guess)", color="#ff7f0e")
b3 = ax.bar(x, fn_pct, w, bottom=np.array(tp_pct)+np.array(fp_pct),
            label="FN (no candidates)", color="#d62728")

ax.set_xticks(x); ax.set_xticklabels([str(r) for r in r0_vals])
ax.set_xlabel("R0", fontsize=11); ax.set_ylabel("% of cases", fontsize=11)
ax.set_title("Outcome breakdown", fontsize=11)
ax.set_ylim(0, 100); ax.legend(fontsize=9)

# Annotate bars with percentages
for i, (tp, fp, fn) in enumerate(zip(tp_pct, fp_pct, fn_pct)):
    if tp > 3:  ax.text(i, tp/2,          f"{tp:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    if fp > 3:  ax.text(i, tp + fp/2,     f"{fp:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    if fn > 3:  ax.text(i, tp + fp + fn/2,f"{fn:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

# Panel B — box plot of candidate set size
ax = axes[1]
data = [df[(df.r0==r0) & (df.n_candidates>0)].n_candidates.values for r0 in r0_vals]
bp = ax.boxplot(data, labels=[str(r) for r in r0_vals], patch_artist=True,
                medianprops=dict(color="black", linewidth=2))
for patch in bp["boxes"]:
    patch.set_facecolor("steelblue"); patch.set_alpha(0.6)
ax.set_xlabel("R0", fontsize=11)
ax.set_ylabel("Candidate set size (cases with ≥1 candidate)", fontsize=10)
ax.set_title("Haystack size distribution", fontsize=11)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved: {OUT_PLOT}")
