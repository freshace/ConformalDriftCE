# Reproducibility

## Determinism

- All experiments fix the random seed (`SEED = 42` in `config.py`) and run single-threaded where
  the script supports it; `run_all.py` runs every experiment with one job to keep results
  deterministic.
- Splits, pool construction and drift injection are seeded from the same seed and are recorded in
  the printed log of each script.
- Confidence intervals are split-bootstrap or seed-bootstrap intervals; the number of resamples is
  set in each script and reported in the paper's captions.

## Environment

| Item | Version |
|---|---|
| Python | 3.10+ (see `requirements.txt`) |
| PostgreSQL | 16 (only for the SQL replay experiments) |
| OS | Linux or Windows; no GPU required |

## Running everything

```bash
python run_all.py          # writes results_repro/
```

The script is idempotent: it recreates `results_repro/` from scratch and does not touch the cached
`results_*.npy` files that accompany the submitted version.

## Checking a reported number

1. Run `python check_tables.py`. It recomputes every number of Table 3 and Table 5 and the figures
   quoted in the abstract from the committed CSVs, and prints `PASS`/`FAIL` per number together
   with the file it read. This needs only the standard library and a fresh clone.
2. For results outside those tables, run the script named after the experiment (for example
   `experiments/experiment_detection_ci.py` for the detector comparison with confidence intervals);
   these need the external feature cache described in `DATA.md`.
3. The cached arrays used in the submitted version are the `results_*.npy` files at the repository
   root; loading them reproduces the plotted curves without re-running the experiments.

## Known limitations

- Wall-clock timings in the SQL replay depend on the machine and the PostgreSQL build; the paper
  reports wall-clock only for the original run, and all quality metrics are independent of timing.
- A few experiments need the PRICE checkpoint and feature caches, which are not redistributed
  (see `DATA.md`); those scripts fail fast with a clear message when the files are missing.
