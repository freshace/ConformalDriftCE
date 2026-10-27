# Data

This repository ships code, protocol files, derived tables and figure sources (about 16 MB under
`data/`). The large binary caches used during development are **not** redistributed.

## What is included

| Path | Content |
|---|---|
| `data/tables/` | Derived result tables (offline windows, feedback, mechanisms) used to rebuild the paper tables |
| `data/continuous/` | Derived continuous-maintenance tables |
| `data/period_sweep/` | Phase sweep of the periodic maintenance schedule: per-run results (`period_runs.csv`), seed extension, summary tables and logs |
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
| Query feature/label cache (`features40_s3a.pkl`, ≈160–210 MB per workload) | Built by an internal prototype on top of the PRICE pipeline; far above the size a repository should carry | Regenerate with the PRICE feature pipeline, or point `DRIFTCE_DATA_PATH` at an existing copy |
| PRICE checkpoint (`model_params.pth`, ≈41 MB) | Third-party model weights, governed by the PRICE release terms | PRICE public release |
| PRICE feature caches (≈540 MB) | Same as above | Regenerate with the PRICE code, or use the PRICE release |
| Raw feature/label pickles | Produced by an unpublished internal prototype; not a public benchmark | Regenerate from the scripts |
| Full PostgreSQL database dumps | Size and third-party licence terms | Load STATS-CDB / JOB from the original sources |

Because of these omissions, the artifact supports **table-level verification** out of the box: the
committed tables under `data/` are the same numbers that appear in the paper, and the read-only
analysis scripts reproduce them. Re-running the full experiment suite additionally needs the
feature cache and the PRICE artefacts above; `run_all.py` reports which experiments it had to skip
and why.
