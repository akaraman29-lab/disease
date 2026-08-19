"""
Step 4: Simulate genome sequences using phastSim.

 phastSim
  - Uses SARS-CoV-2 empirical mutation rates (not just a uniform rate)
  - Models hypermutable sites (APOBEC, ROS activity)
  - Properly handles the Wuhan reference genome as the ancestral sequence
  - Outputs real FASTA sequences we can feed to the inference model

What we give phastSim:
  - Our transmission tree in Newick format
    (branch lengths in days, scaled to subs/site via --scale)
  - The SARS-CoV-2 reference genome MN908947.3 (ships with phastSim)

Key parameter:
  --scale = mu_per_genome_per_day / genome_length
          = 0.091 / 29903
          = 3.04e-6

  This converts our day-unit branch lengths into substitutions per site,
  which is the unit phastSim expects.
"""

import pandas as pd
import subprocess
import os
import shutil
from pathlib import Path

OUT  = Path(__file__).parent
REF  = Path("/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/phastSim/example/MN908947.3.fasta")

# Substitution rate from Denmark paper (Curran-Sebastian et al. 2026)
# 0.64 substitutions per genome per week = 0.091 per day
MU_PER_DAY  = 0.091
GENOME_LEN  = 29903
SCALE       = MU_PER_DAY / GENOME_LEN   # = 3.04e-6  subs/site/day


# ── Step 1: Convert CSV tree to Newick ───────────────────────────────────────

def csv_to_newick(df):
    """
    Build a Newick string from the transmission tree DataFrame.
    Branch lengths are in days (phastSim will scale them via --scale).
    """

    # children[x] = list of case_ids whose parent is x
    children = {-1: []}
    for cid in df.case_id:
        children[int(cid)] = []
    for _, row in df.iterrows():
        children[int(row.parent_id)].append(int(row.case_id))

    # time lookup
    times = dict(zip(df.case_id.astype(int), df.infection_time))
    times[-1] = 0.0

    # parent lookup
    parents = dict(zip(df.case_id.astype(int), df.parent_id.astype(int)))

    def newick(cid):
        kids = children[cid]
        parent = parents.get(cid, -1)

        # branch length = time difference to parent (in days)
        bl = max(times[cid] - times[parent], 1e-8)

        if not kids:
            # Terminal case: leaf node
            return f"case_{cid}:{bl:.6f}"
        else:
            # Internal case (infected others): add a zero-length leaf so this
            # case also gets a sequence, representing its virus at infection time.
            child_strs = ",".join(newick(k) for k in kids)
            self_leaf  = f"case_{cid}:0.000001"
            return f"({child_strs},{self_leaf}):{bl:.6f}"

    roots = children[-1]
    if len(roots) == 1:
        return newick(roots[0]) + ";"
    else:
        return "(" + ",".join(newick(r) for r in roots) + "):0.0;"


print("Step 1/3  Building Newick tree ...")
df = pd.read_csv(OUT / "ground_truth.csv")
nwk = csv_to_newick(df)

nwk_path = OUT / "transmission_tree.nwk"
with open(nwk_path, "w") as f:
    f.write(nwk)

n_leaves = nwk.count("case_")
print(f"          {n_leaves} leaves, {len(df) - n_leaves} internal nodes")
print(f"          Saved: {nwk_path}")


# ── Step 2: Run phastSim ──────────────────────────────────────────────────────
#every time you run simulate_mutations.py it wipes the old output folder first, so you're always working with freshly generated sequences.     
phast_out = OUT / "phastSim_output"
if phast_out.exists():
    shutil.rmtree(phast_out)
phast_out.mkdir()

print("\nStep 2/3  Running phastSim ...")
print(f"          scale = {SCALE:.4e}  (0.091 subs/genome/day ÷ {GENOME_LEN} sites)")

cmd = [
    "phastSim",
    "--outpath",    str(phast_out) + "/",
    "--reference",  str(REF),
    "--treeFile",   str(nwk_path),
    "--scale",      str(SCALE),
    "--seed",       "4",
    "--createFasta",          # output one FASTA with all sequences
    "--createNewick",         # output annotated tree with mutations
]

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print("phastSim error:")
    print(result.stderr)
    raise RuntimeError("phastSim failed")

print("          Done.")

# Show output files
for f in sorted(phast_out.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"          {f.name}  ({size_kb:.0f} KB)")


# ── Step 3: Parse FASTA and merge with ground truth ───────────────────────────

print("\nStep 3/3  Parsing output sequences ...")

from Bio import SeqIO

fasta_path = next(phast_out.glob("*.fasta"))
sequences  = {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")}

print(f"          {len(sequences)} sequences in FASTA")
print(f"          Sequence length: {len(next(iter(sequences.values())))} bp")

# Attach sequence to ground truth
df["sequence_id"] = "case_" + df["case_id"].astype(str)
df["has_sequence"] = df["sequence_id"].isin(sequences)

n_leaves_with_seq = df.has_sequence.sum()
print(f"          Cases with sequences: {n_leaves_with_seq} "
      f"(leaves only — internal nodes don't get sequences)")

# Quick mutation count check
ref_seq = str(next(SeqIO.parse(REF, "fasta")).seq)
sample_ids = df[df.has_sequence].sample(5, random_state=1).sequence_id.tolist()
print("\n          Sample mutation counts from reference:")
for sid in sample_ids:
    seq = sequences[sid]
    muts = sum(a != b for a, b in zip(ref_seq, seq))
    t    = df[df.sequence_id == sid].infection_time.values[0]
    print(f"            {sid:12s}  day {t:5.1f}  {muts} mutations from ref")

# Save sequence index (case_id -> sequence_id mapping + FASTA stays separate)
seq_index = df[df.has_sequence][["case_id", "sequence_id", "infection_time", "generation"]]
seq_index.to_csv(OUT / "sequence_index.csv", index=False)
print(f"\n          Saved sequence_index.csv")
print(f"          Full sequences in: {fasta_path}")

