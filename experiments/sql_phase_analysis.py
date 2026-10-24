# -*- coding: utf-8 -*-
"""Paired analysis of the 30-window phase sweep (cloud run, 2026-10-24).

Periodic coverage comes from data/period_sweep_sql/sql_runs_phase*.csv
(the cloud run); current coverage comes from the archived local table
data/tables/sql_runs.csv.  Both use the same five splits (220-224), so every
comparison is paired within split.

    python experiments/sql_phase_analysis.py
"""
import csv
import glob
import io
import os
import re

import numpy as np

CUR = r"data/tables/sql_runs.csv"
PHASE_DIR = r"data/period_sweep_sql"
CONDS = ("workload_cycles", "correlation_control")
rng = np.random.default_rng(17)


def load_current():
    out = {}
    for r in csv.DictReader(io.open(CUR, encoding="utf-8")):
        if r["policy"] == "current_coverage":
            out[(r["condition"], int(r["seed"]))] = 100 * float(r["coverage"])
    return out


def load_phases():
    out = {}
    for f in glob.glob(os.path.join(PHASE_DIR, "sql_runs_phase*.csv")):
        m = re.search(r"phase(-?\d+)\.csv$", f)
        if not m:
            continue
        ph = int(m.group(1))
        for r in csv.DictReader(io.open(f, encoding="utf-8")):
            out.setdefault(ph, {})[(r["condition"], int(r["seed"]))] = 100 * float(r["coverage"])
    return out


def boot(vals, n=10000):
    v = np.asarray(vals)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


cur = load_current()
phases = load_phases()
seeds = sorted({s for (_, s) in cur})

print(f"splits: {seeds}")
print(f"{'condition':22s} {'phase':>5s} {'periodic':>9s} {'current':>8s} {'gain_pp':>8s} {'95% CI':>16s}")
for cond in CONDS:
    for ph in sorted(phases):
        keys = [(cond, s) for s in seeds if (cond, s) in phases[ph]]
        per = [phases[ph][k] for k in keys]
        cur_v = [cur[k] for k in keys]
        gain = np.array(cur_v) - np.array(per)
        lo, hi = boot(gain)
        print(f"{cond:22s} {ph:>5d} {np.mean(per):9.2f} {np.mean(cur_v):8.2f} "
              f"{gain.mean():8.2f}  [{lo:6.2f},{hi:6.2f}]")
    print()

print("phase envelope of periodic coverage:")
for cond in CONDS:
    vals = {ph: np.mean([phases[ph][(cond, s)] for s in seeds if (cond, s) in phases[ph]])
            for ph in phases}
    best = max(vals, key=vals.get)
    print(f"  {cond:22s} {min(vals.values()):.2f} - {max(vals.values()):.2f} "
          f"(span {max(vals.values())-min(vals.values()):.2f}pp), best phase {best:+d}")

