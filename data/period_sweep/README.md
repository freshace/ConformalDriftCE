# Period sweep for the periodic baseline (80-window recurring drift)

Question: how much of the reported advantage of on-demand monitoring over
**periodic updating** survives when the periodic baseline is allowed to choose
its own phase *post hoc*?

Engine: `provenance/2026-10-12/code/run_offline.py` with one optional argument
(`schedule`) added for this sweep; its default reproduces the archived runs
unchanged. Driver: `experiments/period_sweep.py`.

Protocol `2026-10-12-v2`, trajectory `cycles`, seeds 120-129, 80 windows,
20 monitoring + 50 audit labels per window, 200/200/400 repair labels,
8,000-label update cap (ten repairs). Only the `periodic` policy is re-run; the
`current_coverage` arm is read from the archived tables, so every comparison is
paired within seed.

## Settings

| Family | Definition | Repairs | Labels |
|---|---|---|---|
| `A_phase` | `linspace(6, 76, 10)` shifted by delta and clipped to the replay horizon | 10 | 9,600 |
| `A_wrap` | same ten points shifted circularly over `[6, 76]`; no clipping | 10 | 8,800-9,600 |
| `B_count` | `linspace(6, 76, n)` for n = 4, 6, 8, 10 | n | 1,600 + 800n |

`A_phase` delta = 0 is the schedule used in the paper; it reproduces the archived
coverage exactly (86.20 / 86.43 / 84.54 / 73.48 on stats / imdb / ergastf1 / genome).

## Result

Mean audited coverage over ten seeds, and the paired gain of current coverage
(`current - periodic`); negative gain means the periodic baseline wins.

| Database | phase 0 (paper) | best post-hoc phase | current coverage | gain at phase 0 | gain at best phase |
|---|---|---|---|---|---|
| stats | 86.20 | 88.54 (delta 4) | 88.5 | +2.32 | **-0.02** |
| imdb | 86.43 | 89.18 (delta 4) | 89.0 | +2.58 | **-0.16** |
| ergastf1 | 84.54 | 88.12 (delta 4) | 87.1 | +2.56 | **-1.03** |
| genome | 73.48 | 84.17 (delta 4) | 86.8 | +13.30 | **+2.61** |

Phase envelope (min-max over the five phases): stats 85.81-88.54, imdb 86.43-89.18,
ergastf1 84.54-88.12, genome 72.50-84.17.

Reducing the number of repairs never helps: the best `B_count` setting is eight
repairs, and it stays 1.4-10.3 pp below current coverage.

## Interpretation

The periodic schedule used in the paper sits at an unfavourable phase. Shifting
it by four windows raises the baseline by 2.3-10.7 pp, which is the same order as
the reported advantage. Phase +4 places repairs at windows 18, 33, 49 and 64,
i.e. one to two windows after the drift switches at 16, 32, 48 and 64; a
deployment cannot choose that alignment, because the switch times are not
observable in advance.

Two defensible readings follow, and both change the wording of the continuous
maintenance results:

1. The advantage over periodic updating is not a stable effect across phases; it
   is an advantage over a *particular*, unfavourably aligned schedule.
2. The comparison that does survive is about an unobservable hyperparameter:
   periodic updating needs its phase, and its outcome varies by up to 11.7 pp
   with that choice, whereas on-demand monitoring has no such parameter and
   still reaches the best post-hoc phase on genome and stays within about 1 pp of
   it elsewhere.

## Files

- `period_runs.csv` - one row per (database, seed, sweep, setting), with the
  matched `current_coverage` and the paired gain.
- `summary.csv` - per (database, sweep, setting): mean coverage, mean paired gain,
  95% seed-bootstrap CI, mean label spend.
- `summary.json` - best post-hoc setting and envelope per database and family.
- `cache/` - initial predictors for ergastf1 and genome, refit with the archived
  recipe because no stored initial model exists for those two databases. The
  refit reproduces the archived periodic coverage to within 0.01 pp. Regenerated
  with `lightgbm==4.6.0`, the version pinned in `requirements-repro.txt`; a run
  with 4.7.0 gives the same figures to four decimal places, so the sweep does not
  depend on the minor version.
