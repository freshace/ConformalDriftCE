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
├── run_all.py                dependency-aware driver for the experiment suite
├── check_tables.py           recomputes the paper's table numbers from data/
├── LICENSE                   MIT
├── experiments/              one script per experiment reported in the paper
├── scripts/                  helper utilities
├── data/                     derived tables and protocol files (≈16 MB)
├── figures/                  figure sources (PDF/PNG/SVG)
├── supplement/               Appendices A–H of the paper (DOCX and PDF)
└── results_*.npy             cached result arrays behind the reported tables
```

## What you need before running anything

Two levels of use, with different requirements:

| Goal | What you need |
|---|---|
| **Verify the paper's numbers** | nothing beyond Python and this repository — `python check_tables.py` |
| **Re-run the experiment suite** | the query feature cache (`features40_s3a.pkl`, not shipped, 160–210 MB per workload) plus the PRICE artefacts |

Only the SQL replay needs more: **PostgreSQL 16** with both **STATS-CDB** and **JOB/IMDb** loaded
(the original runs used port 55432). Everything else is offline. See `DATA.md` for how to obtain
each input and which environment variables point at them.

## Requirements

```bash
python -m pip install -r requirements.txt
```

Experiments that replay SQL plans additionally need PostgreSQL 16 with the STATS-CDB and JOB/IMDb
schemas loaded (see `DATA.md`).

## Quick start

```bash
python check_tables.py       # verify the paper's tables against data/tables/  (50 checks)
python run_all.py            # metrics self-check + table check + runnable experiments
```

`check_tables.py` needs nothing beyond the standard library: it recomputes the coverage,
label-cost and runtime numbers of Table 3 and Table 5 and the figures quoted in the abstract
directly from the committed CSV tables, and prints `PASS`/`FAIL` for each of them.

`run_all.py` has two modes:

- **table-only** (default for a fresh clone): the external feature cache is absent, so only the
  self-checks and `check_tables.py` run. The experiment suite is reported as *skipped*, with the
  reason.
- **full**: if `data/features40_s3a.pkl` exists, or `DRIFTCE_DATA_PATH` points at it, the whole
  suite runs deterministically (fixed seed, single job).

Individual experiments can be run on their own, for example:

```bash
python experiments/experiment_correlation.py           # needs the external feature cache
```

## Phase sweep experiments

`experiments/period_sweep.py`, `phase30.py` and `seed_extension.py` produce the schedule-phase
results quoted in the abstract and in the recurring-drift subsection: periodic updates shifted by
0–8 windows (sweep A), a varying number of repairs (sweep B), the 30-seed extension, and the
30-window SQL replay of the same sweep.

Their outputs are committed, so the numbers can be checked without re-running anything:

| Output | Content |
|---|---|
| `data/period_sweep/summary.csv` | coverage and paired gain per database, sweep and phase setting |
| `data/period_sweep/seed_extension.csv` | the 20 additional seeds (130–149) for current / periodic / initial reference |
| `data/period_sweep_sql/sql_runs_phase*.csv` | per-phase summaries of the 30-window SQL replay, 5 splits each |
| `data/period_sweep_sql/sql_windows_phase*.csv` | per-window detail behind those summaries |

Re-running the sweep needs the archived provenance inputs (`provenance/2026-10-12/{code,inputs,raw}`)
and `lightgbm`; that tree is **not** part of this repository. `run_all.py` therefore reports the
three scripts as skipped unless it finds it. The sweep itself takes about 20 seconds once the
inputs are in place; the SQL replay additionally needs PostgreSQL.

## Reproducing a specific number

- `check_tables.py` covers the headline results; each of its checks names the CSV it reads.
- The scripts under `experiments/` are named after the experiment they implement
  (`experiment_correlation*.py`, `experiment_detection*.py`, `experiment_crossdataset*.py`,
  `experiment_*_ci.py` for the confidence-interval variants, …).
- Each script prints the statistics reported in the corresponding subsection and writes the
  underlying arrays; the cached arrays used in the submitted version are the `results_*.npy`
  files at the repository root.
- `REPRODUCIBILITY.md` lists the seeds, the deterministic settings, and the comparison protocol.

## Scope of this artifact

The repository ships the code, the tables and the cached arrays behind every reported number, so
the paper's tables can be **verified** out of the box (`python check_tables.py`). Re-running the
full experiment suite from scratch additionally requires the query feature cache and the PRICE
artefacts, which are not redistributed (see `DATA.md`); it is not claimed that a fresh clone can
regenerate every result end to end.

## Notes on the code

- The scripts were developed on a single machine. Paths are relative to the repository root;
  set the working directory to the repository root before running them.
- A few experiments read pre-computed feature caches and a PRICE checkpoint that are **not**
  redistributed here; see `DATA.md` for how to obtain them from the public PRICE release.
- Plotting scripts write PDF/PNG files into the current directory.

## Supplementary material

`supplement/` contains Appendices A–H of the paper — the experimental protocol and evidence ledger,
the coverage proof, the complete result tables, the continuous-maintenance protocol and results, and
the plan-utility audit — in both DOCX and PDF form. These are the appendices referred to from the
submitted manuscript.

## Data

See `DATA.md`. The four workloads and their true cardinalities come from the public
STATS-CEB, JOB/IMDb, ErgastF1 and Genome benchmarks, in the processed form released with PRICE.
