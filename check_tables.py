# -*- coding: utf-8 -*-
"""Table-level verification for the Conformal-DriftCE artifact.

Every number checked here is quoted in the paper (Table 3 and Table 5, and the
abstract). It is recomputed from the committed CSV tables under ``data/tables/``
with plain arithmetic, so the check needs neither the external feature cache nor
a database.

Usage::

    python check_tables.py            # prints a PASS/FAIL line per number
    python check_tables.py --quiet    # summary only
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "data" / "tables"


def load(name):
    with open(TABLES / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(values):
    return statistics.fmean(values)


CHECKS = []


def check(name, expected, actual, tol, note=""):
    CHECKS.append({
        "name": name,
        "expected": expected,
        "actual": actual,
        "tol": tol,
        "note": note,
        "ok": abs(actual - expected) <= tol,
    })


def check_le(name, limit, actual, note=""):
    """Upper-bound claim: ``actual`` must not exceed ``limit``."""
    CHECKS.append({
        "name": name,
        "expected": limit,
        "actual": actual,
        "tol": 0.0,
        "note": note,
        "ok": actual <= limit + 1e-9,
    })


# ---------------------------------------------------------------- Table 3
def check_table3():
    """Coverage before and after repair for the four base models."""
    paper = {
        "PG": (63.5, 90.2),
        "ALECE": (68.0, 90.3),
        "PRICE": (65.6, 89.9),
        "MSCN": (69.1, 90.3),
    }
    for model, (frozen, repair) in paper.items():
        rows = load(f"crossmodel_{model}.csv")
        by_stage = {}
        for r in rows:
            by_stage.setdefault(r["stage"], []).append(float(r["coverage"]))
        got = {k: mean(v) * 100.0 for k, v in by_stage.items()}
        check(f"Table 3  {model:5s} frozen", frozen, got["frozen"], 0.06, "crossmodel_%s.csv" % model)
        check(f"Table 3  {model:5s} repaired", repair, got["repair"], 0.06, "crossmodel_%s.csv" % model)


# ---------------------------------------------------------------- abstract
def check_abstract_range():
    """``repair restores coverage to 89.1-91.0%``."""
    rows = [r for r in load("cross_all.csv") if r["stage"] == "repair"]
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["condition"]), []).append(float(r["coverage"]))
    values = [mean(v) * 100.0 for v in groups.values()]
    check("Abstract  repaired coverage, min", 89.1, min(values), 0.06, "cross_all.csv")
    check("Abstract  repaired coverage, max", 91.0, max(values), 0.06, "cross_all.csv")


# ---------------------------------------------------------------- Table 5
TABLE5 = {
    "frozen": ("No update", 150, 9.34, 0.00, 0.005),
    "frozen_trigger": ("Initial ref.", 1150, 9.20, 29.20, 0.008),
    "current_coverage": ("Current cov.", 830, 9.30, 19.58, 0.007),
    "current_residual": ("Current resid.", 670, 9.35, 14.67, 0.007),
    "periodic": ("Periodic", 1150, 9.91, 32.13, 0.008),
    "random": ("Random", 1150, 9.45, 32.47, 0.008),
}


def check_table5():
    """Joint-change cost: labels, monitoring SQL, repair SQL, compute."""
    rows = [r for r in load("sql_runs.csv") if r["condition"] == "workload_cycles"]
    by_policy = {}
    for r in rows:
        by_policy.setdefault(r["policy"], []).append(r)
    for policy, (label, labels, mon, rep, comp) in TABLE5.items():
        rs = by_policy[policy]
        got_labels = mean(float(r["deployment_labels"]) for r in rs)
        got_mon = mean(float(r["monitor_wall_seconds"]) for r in rs)
        got_rep = mean(float(r["repair_wall_seconds"]) for r in rs)
        got_comp = mean(float(r["compute_seconds"]) for r in rs)
        check(f"Table 5  {label:13s} labels", labels, got_labels, 1.0, "sql_runs.csv")
        check(f"Table 5  {label:13s} monitor_sql", mon, got_mon, 0.06, "sql_runs.csv")
        check(f"Table 5  {label:13s} repair_sql", rep, got_rep, 0.06, "sql_runs.csv")
        check(f"Table 5  {label:13s} compute", comp, got_comp, 0.0006, "sql_runs.csv")


# ---------------------------------------------------------------- headline
def check_headline():
    """Numbers quoted in the abstract for the real 30-window replay."""
    rows = [r for r in load("sql_runs.csv") if r["condition"] == "workload_cycles"]
    by_policy = {}
    for r in rows:
        by_policy.setdefault(r["policy"], []).append(r)
    cur = by_policy["current_coverage"]
    per = by_policy["periodic"]
    cur_cov = mean(float(r["coverage"]) for r in cur) * 100.0
    per_cov = mean(float(r["coverage"]) for r in per) * 100.0
    cur_lab = mean(float(r["deployment_labels"]) for r in cur)
    per_lab = mean(float(r["deployment_labels"]) for r in per)
    check("Abstract  current coverage (92.7%)", 92.7, cur_cov, 0.06, "sql_runs.csv")
    check("Abstract  periodic coverage (88.9%)", 88.9, per_cov, 0.06, "sql_runs.csv")
    check("Abstract  current labels (830)", 830, cur_lab, 1.0, "sql_runs.csv")
    check("Abstract  periodic labels (1,150)", 1150, per_lab, 1.0, "sql_runs.csv")
    check("Abstract  label reduction (28%)", 28.0, (per_lab - cur_lab) / per_lab * 100.0, 0.6, "sql_runs.csv")


# ---------------------------------------------------------------- phase sweep
PHASE_RANGES = {
    "stats": (85.81, 88.54),
    "imdb": (86.43, 89.18),
    "ergastf1": (84.54, 88.12),
    "genome": (72.50, 84.17),
}


def check_phase_sweep():
    """Periodic schedule shifted by 0-8 windows over the 80-window replay."""
    path = ROOT / "data" / "period_sweep" / "summary.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["sweep"] == "A_phase"]
    groups = {}
    for r in rows:
        groups.setdefault(r["dataset"], []).append(float(r["mean_coverage"]) * 100.0)
    for dataset, (lo, hi) in PHASE_RANGES.items():
        values = groups[dataset]
        check(f"Phase sweep  {dataset:8s} min coverage", lo, min(values), 0.01, "period_sweep/summary.csv")
        check(f"Phase sweep  {dataset:8s} max coverage", hi, max(values), 0.01, "period_sweep/summary.csv")


def check_joint_phase():
    """Joint change: the phase shift moves periodic coverage by at most 2.7 pp."""
    base = ROOT / "data" / "period_sweep_sql"
    per_phase = {}
    for tag in ("-1", "0", "1", "2", "3"):
        with open(base / f"sql_runs_phase{tag}.csv", newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["condition"] == "workload_cycles"]
        per_phase[tag] = mean(float(r["coverage"]) for r in rows) * 100.0
    lo, hi = min(per_phase.values()), max(per_phase.values())
    check_le("Joint change  phase shift range (<= 2.7 pp)", 2.7, hi - lo, "period_sweep_sql/")
    check_le("Joint change  default within best (<= 0.7 pp)", 0.7, hi - per_phase["0"], "period_sweep_sql/")


def main():
    ap = argparse.ArgumentParser(description="Verify the paper's table numbers against data/tables/.")
    ap.add_argument("--quiet", action="store_true", help="print the summary only")
    args = ap.parse_args()

    if not TABLES.is_dir():
        print(f"ERROR: {TABLES} not found; run from the repository root", file=sys.stderr)
        return 2

    check_table3()
    check_abstract_range()
    check_table5()
    check_headline()
    check_phase_sweep()
    check_joint_phase()

    failed = [c for c in CHECKS if not c["ok"]]
    if not args.quiet:
        print(f"{'check':44s} {'paper':>9s} {'recomputed':>11s}  {'tol':>6s}  result")
        print("-" * 84)
        for c in CHECKS:
            flag = "PASS" if c["ok"] else "FAIL"
            print(f"{c['name']:44s} {c['expected']:9.3f} {c['actual']:11.3f}  {c['tol']:6.3f}  {flag}")
        print("-" * 84)

    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("FAILED:")
        for c in failed:
            print(f"  {c['name']}: paper {c['expected']}, recomputed {c['actual']:.3f}")
        return 1
    print("All reported numbers match the committed tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
