"""
Branching epidemic simulation
Generates 10,000 cases with a true transmission tree.

Each case has:
  - case_id         : unique person ID
  - parent_id       : who infected them (-1 = index case)
  - infection_time  : when the virus entered their body
  - generation      : how many links from the root
  - sample_time     : infection_time + generation time to next test
                      (no incubation period modelled at this stage)

Parameters from the Denmark paper (Curran-Sebastian et al. 2026):
  Generation time : Gamma(mean=4.87 days, var=1.98 days)
  Offspring       : NegativeBinomial(R0=2.0, k=0.3)

Output: ground_truth.csv saved next to this script.
"""

import numpy as np
import pandas as pd
import heapq
from pathlib import Path

# ── Output goes next to this script (works on any machine) ───────────────────
OUT = Path(__file__).parent

# ── Parameters ───────────────────────────────────────────────────────────────
SEED         = 4
TARGET_CASES = 10_000
R0           = 2.0
K_DISP       = 0.3        # overdispersion of offspring distribution

# Generation time: Gamma(mean, variance) -> converted to (shape, scale)
GT_MEAN, GT_VAR = 4.87, 1.98
GT_SHAPE = GT_MEAN**2 / GT_VAR   # = 11.98
GT_SCALE = GT_VAR    / GT_MEAN   # = 0.41

# Negative binomial probability parameter
NB_P = K_DISP / (K_DISP + R0)   # = 0.13

rng = np.random.default_rng(SEED)


# ── Build the transmission tree ───────────────────────────────────────────────
def build_tree(target=TARGET_CASES):

    # heap entries: (infection_time, case_id, parent_id)
    heap    = [(0.0, 0, -1)]   # one index case at time=0, no parent
    next_id = 1
    records = []

    while heap and len(records) < target:

        inf_time, case_id, parent_id = heapq.heappop(heap)

        records.append({
            "case_id"       : case_id,
            "parent_id"     : parent_id,
            "infection_time": round(inf_time, 4),
        })

        # How many people does this case infect?
        n_offspring = int(rng.negative_binomial(K_DISP, NB_P))

        # When does each offspring get infected?
        for _ in range(n_offspring):
            generation_time = float(rng.gamma(GT_SHAPE, GT_SCALE))
            child_inf_time  = inf_time + generation_time
            heapq.heappush(heap, (child_inf_time, next_id, case_id))
            next_id += 1

    df = pd.DataFrame(records)

    # Generation depth: 0 = index case, 1 = their direct infectees, etc.
    depth = {-1: -1}
    for _, row in df.sort_values("infection_time").iterrows():
        depth[row.case_id] = depth[row.parent_id] + 1
    df["generation"] = df.case_id.map(depth)

    # sample_time = infection_time (no incubation)
    df["sample_time"] = df["infection_time"]

    return df


# ── Run ───────────────────────────────────────────────────────────────────────
print("Building transmission tree ...")
df = build_tree()
print(f"  {len(df)} cases | "
      f"{df.infection_time.max():.1f} days | "
      f"max generation {df.generation.max()}")

# Save
csv_path = OUT / "ground_truth.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# Summary
print("\nFirst 10 cases:")
print(df.head(10).to_string(index=False))

print("\nCases per generation:")
print(df.groupby("generation").size().to_string())