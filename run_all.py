# -*- coding: utf-8 -*-
"""Reproduction driver for the Conformal-DriftCE experiments.

Two modes, selected automatically by whether the external inputs are present:

1. Full mode — ``data/features40_s3a.pkl`` (or ``DRIFTCE_DATA_PATH``) exists.
   Every experiment in ``EXPS`` is executed with a fixed seed and one job, and
   each script's console output is written to ``results_repro/``.

2. Table-only mode — the external feature cache is absent (the default for a
   fresh clone, because the cache is far too large to redistribute).
   Only the experiments that read the committed tables under ``data/`` are run;
   everything else is reported as SKIPPED with the reason, not as a failure.

Usage::

    python run_all.py            # auto-detect
    python run_all.py --strict   # exit non-zero if any experiment is skipped
"""
import argparse
import os
import subprocess
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DIR, "results_repro")
os.makedirs(OUT, exist_ok=True)

# External feature/label cache; see DATA.md.
DATA_PATH = os.environ.get("DRIFTCE_DATA_PATH", os.path.join("data", "features40_s3a.pkl"))

# Experiments that only read the committed tables / cached arrays and therefore
# run in a fresh clone. Keep this list short and explicit.
TABLE_ONLY = [
    "experiment_planregret_analysis.py",
]

EXPS = [
    "experiment_correlation.py",
    "experiment_correlation_realdata.py",
    "experiment_scaledrift.py",
    "experiment_workloaddrift.py",
    "experiment_mixeddrift.py",
    "experiment_schemadrift.py",
    "experiment_baselines.py",
    "experiment_modelagnostic.py",
    "analyze_consistency.py",
    "experiment_ablation.py",
    "experiment_residual_necessity.py",
    "experiment_residual_sensitivity.py",
    "experiment_loop_decision.py",
    "experiment_crossdataset.py",
    "experiment_alphatradeoff.py",
    "experiment_qerror_sideeffect.py",
    "experiment_verify_audit.py",
    "experiment_detection.py",
    "experiment_detection_ci.py",
    "experiment_necessity_ci.py",
    "experiment_correlation_ci.py",
    "experiment_ablation_ci.py",
    "experiment_modelagnostic_ci.py",
    "experiment_driftspectrum_ci.py",
    "experiment_baselines_ci.py",
    "experiment_crossdataset_ci.py",
    "experiment_temporaldrift.py",
    "experiment_groupcoverage.py",
    "experiment_groupcoverage_ci.py",
    "experiment_selective.py",
    "experiment_temporalhysteresis.py",
    "experiment_aci_baseline.py",
    "experiment_hybrid_policy.py",
    "experiment_recal_sideeffects.py",
    "experiment_propA_bound.py",
    "experiment_qstar_report.py",
    "experiment_qstar_precondition.py",
    "experiment_qstar_bias_dose.py",
    "experiment_qstar_audit_design.py",
    "experiment_qstar_detector.py",
    "experiment_correlation_realdata_ci.py",
    "experiment_planregret_analysis.py",
]

TIMEOUTS = {
    "experiment_correlation_realdata_ci.py": 2400,
    "experiment_detection_ci.py": 2400,
    "experiment_necessity_ci.py": 1800,
    "experiment_groupcoverage_ci.py": 2400,
    "experiment_correlation_ci.py": 2400,
    "experiment_ablation_ci.py": 3600,
    "experiment_modelagnostic_ci.py": 3600,
    "experiment_driftspectrum_ci.py": 3600,
    "experiment_baselines_ci.py": 2400,
    "experiment_crossdataset_ci.py": 5400,
    "experiment_crossdataset.py": 1800,
    "experiment_temporaldrift.py": 1800,
    "experiment_residual_necessity.py": 3600,
    "experiment_residual_sensitivity.py": 1800,
    "experiment_loop_decision.py": 1800,
}
DEFAULT_TIMEOUT = 900


def preflight():
    """Return True when the external feature cache is available."""
    candidates = [
        DATA_PATH if os.path.isabs(DATA_PATH) else os.path.join(DIR, DATA_PATH),
        os.path.join(DIR, "data", "features40_s3a.pkl"),
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"[ok] external feature cache: {c}")
            return True
    print("[missing] external feature cache (features40_s3a.pkl)")
    print("          Most experiments need it; they will be skipped.")
    print("          See DATA.md for what it is and how to obtain it.")
    return False


def run_metrics_selfcheck():
    path = os.path.join(DIR, "metrics.py")
    print("\n########## metrics.py (metric definition self-check) ##########")
    try:
        r = subprocess.run([sys.executable, path], cwd=DIR, capture_output=True, text=True,
                           timeout=120, encoding="utf-8", errors="replace")
        print((r.stdout or "")[-1200:])
        if r.returncode != 0:
            print(f"[FAILED exit={r.returncode}]\n{(r.stderr or '')[-600:]}")
            return False
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] {str(e)[:160]}")
        return False
    return True


def run_one(exp, timeout):
    path = os.path.join(DIR, "experiments", exp)
    print(f"\n########## {exp} ##########")
    if not os.path.exists(path):
        print(f"[MISSING] experiments/{exp}")
        return "missing"
    try:
        r = subprocess.run([sys.executable, path], cwd=DIR, capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        log = os.path.join(OUT, exp.replace(".py", ".txt"))
        with open(log, "w", encoding="utf-8") as f:
            f.write(f"exit={r.returncode}\n" + (r.stdout or ""))
        print((r.stdout or "")[-1800:])
        if r.returncode != 0:
            print(f"[FAILED exit={r.returncode}]")
            return "failed"
        return "passed"
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT after {timeout}s]")
        return "timeout"
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] {str(e)[:160]}")
        return "error"


def main():
    ap = argparse.ArgumentParser(description="Run the Conformal-DriftCE experiments.")
    ap.add_argument("--strict", action="store_true",
                    help="return a non-zero exit code when experiments are skipped")
    args = ap.parse_args()

    print("=" * 64)
    print("Conformal-DriftCE reproduction driver (seed 42, single job)")
    print("=" * 64)

    ready = preflight()
    metrics_ok = run_metrics_selfcheck()

    if ready:
        todo, skipped = EXPS, []
    else:
        todo = [e for e in EXPS if e in TABLE_ONLY]
        skipped = [e for e in EXPS if e not in TABLE_ONLY]
        if skipped:
            print(f"\n[skip] {len(skipped)} experiments need the external feature cache")
            print("       run with DRIFTCE_DATA_PATH=<path to features40_s3a.pkl> to execute them")

    results = {"passed": 0, "failed": 0, "timeout": 0, "error": 0, "missing": 0}
    for exp in todo:
        status = run_one(exp, TIMEOUTS.get(exp, DEFAULT_TIMEOUT))
        results[status] += 1

    print("\n" + "=" * 64)
    print(f"metrics self-check: {'PASSED' if metrics_ok else 'FAILED'}")
    print(f"experiments: {results['passed']} passed, {results['failed']} failed, "
          f"{results['timeout']} timeout, {results['error']} error, {results['missing']} missing")
    print(f"skipped (missing external inputs): {len(skipped)}")
    print(f"logs: {OUT}")
    print("=" * 64)

    hard_fail = (results["failed"] or results["timeout"] or results["error"]
                 or results["missing"] or not metrics_ok)
    if hard_fail:
        return 1
    if args.strict and skipped:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
