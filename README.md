# Conformal-DriftCE — Artifact

Artifact for the paper **“When Coverage Misleads: Designing Reliability Monitoring for
Cardinality Estimation Under Drift”** (PVLDB Volume 20 submission).

Conformal-DriftCE is a model-agnostic reliability layer that wraps an existing cardinality
estimator: a lightweight residual corrector, conformal prediction intervals, version-consistent
window monitoring, on-demand repair, and an independent audit on disjoint query identities.

## What is in this repository

```
.
├── README.md                 this file
├── DATA.md                   data sources, licences, and what is not redistributed
├── REPRODUCIBILITY.md        seeds, determinism, and how to check the numbers
├── requirements.txt          Python dependencies
├── config.py, data.py, drift.py, features.py     data loading, features, drift injection
├── conformal.py, model.py, metrics.py            calibration, correction, evaluation
├── pool_adapter.py, datasim.py, merge_pools.py   feedback pools and synthetic data
├── run_all.py                one-command reproduction of the drift experiments
├── experiments/              one script per experiment reported in the paper
├── scripts/                  helper utilities
├── data/                     derived tables and protocol files (≈16 MB)
├── figures/                  figure sources (PDF/PNG/SVG)
└── results_*.npy             cached result arrays behind the reported tables
```

## Requirements

```bash
python -m pip install -r requirements.txt
```

Experiments that replay SQL plans additionally need **PostgreSQL 16** with the STATS-CDB schema
loaded from `data/` (see `DATA.md`).

## Quick start

```bash
python run_all.py            # runs the drift experiments, writes results_repro/
```

`run_all.py` runs deterministically (fixed seed, single job) and writes every produced array to
`results_repro/`. Individual experiments can be run on their own, for example:

```bash
python experiments/experiment_correlation.py
python experiments/experiment_detection_ci.py
```

## Reproducing a specific number

- The scripts under `experiments/` are named after the experiment they implement
  (`experiment_correlation*.py`, `experiment_detection*.py`, `experiment_crossdataset*.py`,
  `experiment_*_ci.py` for the confidence-interval variants, …).
- Each script prints the statistics reported in the corresponding subsection and writes the
  underlying arrays; the cached arrays used in the submitted version are the `results_*.npy`
  files at the repository root.
- `REPRODUCIBILITY.md` lists the seeds, the deterministic settings, and the comparison protocol.

## Notes on the code

- The scripts were developed on a single machine. Paths are relative to the repository root;
  set the working directory to the repository root before running them.
- A few experiments read pre-computed feature caches and a PRICE checkpoint that are **not**
  redistributed here; see `DATA.md` for how to obtain them from the public PRICE release.
- Plotting scripts write PDF/PNG files into the current directory.

## Data

See `DATA.md`. The four workloads and their true cardinalities come from the public
STATS-CEB, JOB/IMDb, ErgastF1 and Genome benchmarks, in the processed form released with PRICE.
