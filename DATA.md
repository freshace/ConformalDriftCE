# Data

This repository ships code, protocol files, derived tables and figure sources (about 16 MB under
`data/`). The large binary caches used during development are **not** redistributed.

## What is included

| Path | Content |
|---|---|
| `data/tables/` | Derived result tables (offline windows, feedback, mechanisms) used to rebuild the paper tables |
| `data/continuous/` | Derived continuous-maintenance tables |
| other `data/*` | Protocol and configuration files |

## External sources

- The four query workloads and their true cardinalities come from the public
  **STATS-CEB**, **JOB/IMDb**, **ErgastF1** and **Genome** benchmarks, in the processed form
  released with **PRICE**. Please cite and follow the terms of those releases.
- The nine STATS CSV/DDL files needed to replay SQL plans are included under
  `data/` and follow the benchmark's original licence.

## What is not redistributed, and why

| Not included | Reason | How to obtain |
|---|---|---|
| PRICE checkpoint (`model_params.pth`, ≈41 MB) | Third-party model weights, governed by the PRICE release terms | PRICE public release |
| PRICE feature caches (≈540 MB) | Same as above | Regenerate with the PRICE code, or use the PRICE release |
| Raw feature/label pickles | Produced by an unpublished internal prototype; not a public benchmark | Regenerate from the scripts |
| Full PostgreSQL database dumps | Size and third-party licence terms | Load STATS-CDB / JOB from the original sources |

Because the omitted files are either third-party artefacts or regenerable caches, the reported
numbers can be reproduced from the public benchmarks plus the code in this repository; the
scripts that need the omitted caches say so when they are run.
