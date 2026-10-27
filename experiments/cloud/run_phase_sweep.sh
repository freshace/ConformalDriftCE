#!/usr/bin/env bash
# Phase sweep of the periodic policy in the 30-window real replay.
#
# Put this file and patch_run_sql.py in the same directory as `code/`, `protocol/`,
# `inputs/` (i.e. provenance/2026-10-12/ on the host that has the databases).
#
#   bash run_phase_sweep.sh              # baseline + phases -1 1 2 3
#   PHASES="-2 -1 1 2 3" bash run_phase_sweep.sh
#
# Only the periodic policy is run; everything else (roles, repairer, budget,
# audits) is unchanged.  Each phase takes roughly 8 minutes on the original host.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PHASES="${PHASES--1 1 2 3}"
OUT="$HERE/tables/phase"

mkdir -p "$HERE/tables" "$HERE/raw" "$HERE/qa" "$HERE/tmp" "$OUT"

python3 "$HERE/patch_run_sql.py" "$HERE/code/run_sql.py"

run_phase () {
  local p="$1"
  echo ">>> periodic, phase $p"
  CE_ONLY_PERIODIC=1 PERIOD_PHASE="$p" python3 "$HERE/code/run_sql.py"
  cp "$HERE/tables/sql_runs.csv"    "$OUT/sql_runs_phase${p}.csv"
  cp "$HERE/tables/sql_windows.csv" "$OUT/sql_windows_phase${p}.csv"
}

echo "=== baseline (phase 0, the schedule used in the paper) ==="
run_phase 0
python3 - "$OUT/sql_runs_phase0.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print("  expected on the paper's host: workload_cycles 88.9%, correlation_control 91.2%")
for c in sorted({r["condition"] for r in rows}):
    v = [float(r["coverage"]) for r in rows if r["condition"] == c]
    print(f"  {c:22s} periodic coverage = {100*sum(v)/len(v):6.2f}%")
PY

for p in $PHASES; do
  run_phase "$p"
done

echo "=== summary ==="
python3 - "$OUT" <<'PY'
import csv, glob, os, sys, statistics
base = sys.argv[1]
for f in sorted(glob.glob(os.path.join(base, "sql_runs_phase*.csv")),
                key=lambda x: int(x.split("phase")[-1].split(".")[0])):
    rows = list(csv.DictReader(open(f)))
    tag = os.path.basename(f).replace("sql_runs_phase", "").replace(".csv", "")
    per_cond = {c: statistics.mean(float(r["coverage"]) for r in rows if r["condition"] == c)
                for c in {r["condition"] for r in rows}}
    print(f"phase {tag:>3s}: " + "  ".join(f"{c}={100*v:.2f}%" for c, v in sorted(per_cond.items())))
PY
echo "csv files in $OUT"
