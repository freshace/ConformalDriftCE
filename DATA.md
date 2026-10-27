# Data

This repository ships code, protocol files, derived tables and figure sources (about 16 MB under
`data/`). The large binary caches used during development are **not** redistributed.

## What is included

| Path | Content |
|---|---|
| `data/tables/` | Derived result tables (offline windows, feedback, mechanisms) used to rebuild the paper tables |
| `data/continuous/` | Derived continuous-maintenance tables |
| `data/period_sweep/` | Phase sweep of the periodic maintenance schedule: per-run results (`period_runs.csv`), seed extension, summary tables and logs |
| `data/period_sweep_sql/` | 30-window SQL replay of the same phase sweep: per-phase run summaries (`sql_runs_phase{,-1,1,2,3}.csv`, 5 splits each) and per-window details (`sql_windows_phase*.csv`) |
| `data/fig8/` | Detector raw and summary CSVs behind Figure 8 |
| other `data/*` | Protocol and configuration files |

The `data/period_sweep/cache/` directory (about 36 MB of `initial_*.npz`) is **not committed**: it is an
intermediate prediction cache that `experiments/period_sweep.py` regenerates.

## External sources

- The four query workloads and their true cardinalities come from the public
  **STATS-CEB**, **JOB/IMDb**, **ErgastF1** and **Genome** benchmarks, in the processed form
  released with **PRICE**. Please cite and follow the terms of those releases.
- Replaying SQL plans requires those databases to be loaded into a local PostgreSQL instance.
  **No database dump, and no STATS CSV/DDL file, is shipped in this repository**; load them from
  the benchmark releases.

## What is not redistributed, and why

| Not included | Reason | How to obtain |
|---|---|---|
| Query feature/label cache (`features40_s3a.pkl`, ≈160–210 MB per workload) | An input encoding shared by the base estimators, not part of this paper's contribution; also far above the size a repository should carry | Regenerate with the feature-encoding pipeline used by the base estimators, or point `DRIFTCE_DATA_PATH` at an existing copy |
| PRICE checkpoint (`model_params.pth`, ≈41 MB) | Third-party model weights, needed only for the PRICE base model and the native fine-tuning baseline of Table 4 | PRICE public release |
| PRICE feature caches (≈540 MB) | Same as above | Regenerate with the PRICE code, or use the PRICE release |
| Raw feature/label pickles | Produced by an unpublished internal prototype; not a public benchmark | Regenerate from the scripts |
| Full PostgreSQL database dumps | Size and third-party licence terms | Load STATS-CDB / JOB from the original sources |

Because of these omissions, the artifact supports **table-level verification** out of the box: the
committed tables under `data/` are the same numbers that appear in the paper, and the read-only
analysis scripts reproduce them. Re-running the full experiment suite additionally needs the
feature cache and the PRICE artefacts above; `run_all.py` reports which experiments it had to skip
and why.

## Getting the missing inputs

The code reads its inputs from environment variables, so no file paths are hard-coded:

| Variable | Used by | Meaning |
|---|---|---|
| `DRIFTCE_DATA_PATH` | `config.py`, most experiments | Path to the query feature/label cache (`features40_s3a.pkl`) |
| `DRIFTCE_FEATURES_ROOT` | `experiment_crossdataset*.py` | Directory holding the per-database feature caches |
| `DRIFT_FIG_OUT` | `experiments/plot_fig12_planregret.py` | Output directory for regenerated figures |

Steps for a full re-run:

1. **Query feature cache.** Build it with the feature-encoding pipeline used by the base estimators
   on the four benchmark workloads, or copy an existing `features40_s3a.pkl` and point
   `DRIFTCE_DATA_PATH` at it. One cache covers one workload (≈160–210 MB); the cross-database
   experiments expect one per database under `DRIFTCE_FEATURES_ROOT`.
2. **PRICE artefacts.** Only needed to rebuild the PRICE arm: its pretrained checkpoint and the
   `features/` directory from the public PRICE release. The reliability layer and the other three
   base models (PG, ALECE, MSCN) do not depend on them, and no PRICE weights are redistributed here.
3. **Databases.** Load STATS-CEB and JOB/IMDb into PostgreSQL 16; the SQL replay experiments
   (`data/tables/sql_runs.csv`, `sql_windows.csv`) were produced against those schemas.

Runtime can be reduced without changing the reported settings: most confidence-interval
experiments expose their repetition count through an environment variable
(`CORR_R`, `DET_R`, `NEC_R`, `MA_R`, `BL_R`, `GC_R`, `DS_R`, `ABL_R`, `CD_R`, `VA_R`, defaulting to
the values used in the paper).

## What is this paper's, and what is reused

This paper contributes the reliability layer — the residual corrector and conformal interval
(`conformal.py`), the version-consistent monitoring and repair logic (`drift.py`), the label-role
protocol, the audit design, and the metrics (`metrics.py`) — together with the experiment drivers
that evaluate it across four databases, three drift types and four base models.

The base estimators (PG, ALECE, PRICE, MSCN), the benchmark workloads and the query feature
encoding are **inputs** to those experiments, not contributions. PRICE is one of the four base
models and doubles as the strongest native baseline (Table 4); nothing in the reliability layer is
specific to it.
