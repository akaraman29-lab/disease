# Transmission Inference Validation — SARS-CoV-2

A simulation framework for benchmarking a genomic transmission inference model (Jacob's model, from the Danish Transmission project) against a synthetic outbreak where the ground truth is known.

---

## What this does

Since ground truth is unknowable in real outbreak data, we simulate a synthetic epidemic where we control everything:

1. **Simulate the epidemic** — 10,000 cases with known true transmission pairs (who infected whom, when)
2. **Simulate genome sequences** — using phastSim with SARS-CoV-2 parameters, so each case has a realistic viral genome
3. **Run Jacob's model** — as if we didn't know the true pairs; let the model infer them
4. **Measure performance** — compare inferred pairs against ground truth using precision, recall, F1, and several other metrics

**Core finding:** SARS-CoV-2 sits in a difficult regime. At ~0.44 substitutions per transmission combined with R0=2, many cases look genomically identical within any 14-day window. The model finds the right person ~87% of the time in the candidate set (recall) but can only pick them out from ~130 equally plausible candidates (precision ~32%).

---

## Parameters (Denmark paper — Curran-Sebastian et al. 2026)

| Parameter | Value |
|---|---|
| R0 | 2.0 |
| Overdispersion k | 0.3 |
| Generation time | Gamma(mean=4.87 days, var=1.98 days) |
| Substitution rate | 0.091 subs/genome/day |
| Genome length | 29,903 bp (SARS-CoV-2) |
| Reference | MN908947.3 (Wuhan) |
| Inference time window | ±14 days |
| Hamming threshold | ≤ 2 |

---

## Core Pipeline

Run these four scripts in order.

### 1. `without_inc.py` — Epidemic simulation

Simulates a branching epidemic of 10,000 cases using a Gillespie-style priority queue (heap). Each case gets a `case_id`, `parent_id` (who infected them), `infection_time`, `generation` depth, and `sample_time`. Offspring drawn from NegativeBinomial(R0=2, k=0.3); generation times from Gamma(mean=4.87, var=1.98 days).

**Output:** `ground_truth.csv` — the answer key for the entire validation.

```
python without_inc.py
```

### 2. `simulate_mutations.py` — Genome sequence simulation

Takes `ground_truth.csv`, converts the transmission tree to Newick format with branch lengths in days, and runs phastSim with `--scale 3.04e-6` (= 0.091/29903) to simulate SARS-CoV-2 genome sequences along the tree using the Wuhan reference MN908947.3. Internal nodes (cases who infected others) normally get no sequence from phastSim, so a zero-length fake leaf `case_{cid}:0.000001` is added for every internal node, giving every case a sequence.

**Output:** `phastSim_output/sars-cov-2_simulation_output.fasta` — 10,000 sequences of 29,903 bp.

```
python simulate_mutations.py
```

### 3. `run_inference.py` — Transmission inference

Adapter that runs Jacob's model on the simulated data. Loads the FASTA and ground truth, cleans sequences, compresses to variable sites only (~3,646 out of 29,903), precomputes Jacob's `scenario1_probability` scores into a 10×21 lookup table (reducing runtime from ~75 minutes to ~2 minutes), then for each case finds candidates within 14 days and Hamming ≤ 2, scores them, and saves results.

**Output:** `infectors_dict.pickle` (Jacob's exact format) and `inference_results.csv`.

```
python run_inference.py
```

### 4. `evaluate_inference.py` — Multi-strategy evaluation

Loads `infectors_dict.pickle` and `ground_truth.csv`, rebuilds ranked candidate lists, and evaluates using five strategies plus Strategy 6a.

| Strategy | Description |
|---|---|
| Top-1 | True parent has the single highest score (precision / recall / F1) |
| Top-k | True parent appears in top k=2, 3, 5 by score |
| Any candidate | True parent appears anywhere in the candidate set |
| Rank histogram | Distribution of true parent's rank |
| MRR | Mean reciprocal rank across all evaluable cases |
| 6a | Normalised probability mass assigned to the true parent |

**Output:** `evaluation_results.csv` and printed summary.

```
python evaluate_inference.py
```

---

## Sensitivity Analyses

These scripts ask how performance changes as epidemic biology changes. They each re-run the full simulation + inference pipeline internally.

### `mutation_rate_sweep.py`

Varies the simulated mutation rate from 0.005 to 1.0 subs/genome/day while keeping the inference model's assumed rate fixed at 0.091. Higher mutation rate → more genomic signal → better performance, until the Hamming ≤ 2 filter starts excluding true pairs.

**Output:** `mutation_rate_sweep_results.csv`, `mutation_rate_sweep.png`

### `generation_time_sweep.py`

Varies the true generation time mean (2–12 days) while keeping all inference model parameters fixed. Longer generation time → fewer candidates per case → better performance.

**Output:** `generation_time_sweep_results.csv`, `generation_time_sweep.png`

### `sweep_2d.py`

2D sweep of mutation rate × generation time mean (5 × 5 = 25 combinations). Confirms that the ratio μ × g (substitutions per transmission) is the key performance driver, though this relationship breaks down at high mutation rates due to the hard-coded Hamming filter.

**Output:** `sweep_2d_results.csv`, `sweep_2d_heatmap.png`

### `r0_sweep.py`

Varies R0 from 1.2 to 4.0. Higher R0 → more secondary cases → more candidates per case → more false positives → lower precision. Recall stays relatively stable.

**Output:** `r0_sweep_results.csv`, `r0_sweep.png`

### `r0_sweep_ci.py`

Repeats the R0 sweep with 7 different random seeds per R0 value (42 runs total) to get confidence intervals on precision, recall, and F1.

**Output:** `r0_sweep_ci_results.csv`, `r0_sweep_ci_plot.png`

---

## Supporting / Diagnostic Scripts

These are not part of the main pipeline but were used during analysis.

| Script | Purpose |
|---|---|
| `bootstrap_evaluate.py` | Bootstrap 95% CIs (2000 iterations) for all 6 evaluation strategies from `evaluation_results.csv` |
| `r0_sweep_percase.py` | R0 sweep that saves per-case outcomes (outcome, n_candidates, true parent rank) for detailed analysis |
| `subsampling_ci.py` | Tests how subsampling rate (fraction of cases sequenced) affects model performance |
| `subsampling_realistic_ci.py` | Harsher subsampling evaluation: predictions made when the true infector is absent count as false positives |
| `tie_and_indegree_analysis.py` | (1) How many candidates tie for the top score (scoring resolution problem); (2) candidate set size growing over epidemic time |
| `sweep_remaining.py` | Utility — appends results for high mutation rates (μ=0.5, 1.0) to `mutation_rate_sweep_results.csv` without re-running the full sweep |
| `tree simulation.py` | Empty placeholder |

---

## File outputs summary

| File | Produced by | Contents |
|---|---|---|
| `ground_truth.csv` | `without_inc.py` | True transmission tree — case_id, parent_id, infection_time, generation, sample_time |
| `transmission_tree.nwk` | `simulate_mutations.py` | Newick tree with day-unit branch lengths |
| `phastSim_output/*.fasta` | `simulate_mutations.py` | 10,000 SARS-CoV-2 genome sequences (29,903 bp) |
| `sequence_index.csv` | `simulate_mutations.py` | case_id → sequence_id mapping |
| `sequences_clean.fasta` | `run_inference.py` | Cleaned sequences (uppercase, non-ACTG → `-`) |
| `infectors_dict.pickle` | `run_inference.py` | Jacob's format: `{strain: {candidate: (prob, (hamming, datediff))}}` |
| `inference_results.csv` | `run_inference.py` | Per-case inferred vs true parent |
| `evaluation_results.csv` | `evaluate_inference.py` | Per-case rank, top-k flags, reciprocal rank, normalised probability |

---

## Dependencies

- Python 3.x
- `numpy`, `pandas`, `scipy`, `matplotlib`
- `biopython`
- `regex`
- `hammingdist`
- `tqdm`
- [`phastSim`](https://github.com/NicolaDM/phastSim) (must be on PATH)
- Jacob's `transmission_functions.py` — expected at `Danish_Transmission/src/network_construction/`

---

## Context

This validation framework was built to test the limits of genomic transmission inference for SARS-CoV-2. The model works well when sequences are genetically distinct (high mutation rate or long generation time) but struggles in the real SARS-CoV-2 regime where the virus accumulates only ~0.44 substitutions per transmission event. The sensitivity analyses map out exactly where the method succeeds and fails across epidemic parameter space.
