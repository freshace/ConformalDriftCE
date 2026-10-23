"""Recompute paper figure statistics from preserved arrays; no training or DB calls."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plot_fig12_planregret import paired_tail_ci

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results_repro"
OUT = ROOT / "results_revision_20260930" / "figures_audit"
NEW = ROOT / "results_revision_20260930" / "figures_sensitivity"


def summary(values, scale=1):
    values = np.asarray(values, float) * scale
    mean = float(values.mean())
    half = float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "normal95_half": half, "lower": mean - half,
            "upper": mean + half, "repetitions": len(values)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"interval_definition": "mean across random splits +/- 1.96 split SD/sqrt(R); descriptive split uncertainty"}
    reps = np.load(OLD / "results_decouple_reps.npy")
    np.testing.assert_allclose(reps.mean(axis=0), np.load(OLD / "results_decouple_mean.npy"))
    columns = {"gmq_all": 1, "coverage_all": 2, "gmq_multi": 3, "coverage_multi": 4,
               "qerror95_multi": 5, "qerror_max_multi": 6}
    stats["necessity"] = {name: {"baseline": summary(reps[:, 0, col], 100 if "coverage" in name else 1),
                                 "7.39x": summary(reps[:, -1, col], 100 if "coverage" in name else 1)}
                          for name, col in columns.items()}
    stats["plan_tail"] = {}
    for tag in ("D1", "D7.39", "D16q"):
        regret = np.load(OLD / ("planregret_native_" + tag + ".npz"))["R"]
        stats["plan_tail"][tag] = {"n": len(regret), "point_percent": float(np.mean(regret[:, 0] > 1.5) * 100),
                                  "robust_percent": float(np.mean(regret[:, 1] > 1.5) * 100),
                                  "paired_tail_bootstrap95_pp": (np.array(paired_tail_ci(regret)) * 100).tolist(),
                                  "bootstrap": "paired queries; 2000 draws; RandomState seed 7"}
    raw = np.load(NEW / "residual_sensitivity_raw.npz")
    for i in range(len(raw["base_scores"])):
        ids = np.concatenate([raw["indices_" + role][i] for role in ("train", "calibration", "validation", "test")])
        assert len(ids) == len(np.unique(ids)), "split roles overlap"
    stats["sensitivity"] = {key: summary(raw[key], 100 if "coverage" in key else 1)
                            for key in raw.files if "width" in key or "coverage" in key}
    stats["sensitivity"]["width_ratio_base_over_residual"] = summary(raw["base_matched_width"] / raw["residual_width"])
    stats["sensitivity"]["test_coverage_gap_residual_minus_matchedbase_pp"] = summary(raw["residual_coverage"] - raw["base_matched_coverage"], 100)
    stats["sensitivity"]["signal_gap_residual_minus_base_pp"] = [summary(raw["base_cov"][:, i] - raw["res_cov"][:, i], 100) for i in range(raw["base_cov"].shape[1])]
    stats["sensitivity"]["signal_gap_residual_minus_matchedbase_pp"] = [summary(raw["match_cov"][:, i] - raw["res_cov"][:, i], 100) for i in range(raw["base_cov"].shape[1])]
    stats["sensitivity"]["point_qerror"] = {}
    for arm, key in (("base", "base_scores"), ("residual", "residual_scores")):
        qe = np.exp(raw[key])
        stats["sensitivity"]["point_qerror"][arm] = {"q" + str(q): summary(np.percentile(qe, q, axis=1)) for q in (50, 90, 95, 99)}
        stats["sensitivity"]["point_qerror"][arm]["gmq"] = summary(np.exp(raw[key].mean(axis=1)))
    # Repair the stored report's sign from raw arrays without rerunning fitted models.
    report_path = NEW / "experiment_residual_sensitivity.txt"
    text = report_path.read_text(encoding="utf-8")
    start = text.index("signal gap (Residual - Base, pp):")
    end = text.index("读法：", start)
    corrected = "signal gap (Residual - Base, pp):\n" + "\n".join(
        f"  {factor:6.2f}x  signal gap = {gap['mean']:+.1f}pp"
        for factor, gap in zip(raw["dose"][0], stats["sensitivity"]["signal_gap_residual_minus_base_pp"])) + "\n\n"
    report_path.write_text(text[:start] + corrected + text[end:], encoding="utf-8")
    baseline = np.load(OLD / "baselines_ci.npz")
    stats["oracle"] = {tag: {arm: summary(baseline[tag + "__" + arm], 100) for arm in ("frozen", "ours", "oracle")} for tag in ("corr", "work")}
    for tag in ("corr", "work"):
        stats["oracle"][tag]["ours_minus_oracle_pp"] = summary(baseline[tag + "__ours"] - baseline[tag + "__oracle"], 100)
    audit = np.load(OLD / "verify_audit.npz")
    stats["verify_audit"] = {}
    for arm in ("frozen", "repair", "recal_only", "garbage"):
        np.testing.assert_allclose(audit["width_" + arm], np.exp(audit["q_" + arm]))
        stats["verify_audit"][arm] = {"width": summary(audit["width_" + arm]),
                                     "qerror_median": summary(audit["medqe_" + arm])}
    stats["crossmodel_width"] = {}
    for filename in ("modelagnostic_ci.npz", "crossmodel_stats5.npz"):
        saved = np.load(OLD / filename)
        stats["crossmodel_width"][filename] = {key: summary(saved[key]) for key in saved.files
                                               if key.startswith(("w0__", "wq__"))}
    correlation = np.load(OLD / "correlation_ci.npz")
    stats["correlation_width_from_q"] = {key: summary(np.exp(correlation[key])) for key in correlation.files
                                         if key.startswith(("q_frozen__", "q_new__"))}
    stats["boundaries"] = ["Oracle arrays store only coverage: q-error/width cannot be reconstructed without refitting and new saved scores.",
                           "verify_audit stores median q-error and width/q, not per-query q-error: p90/p99 cannot be recovered.",
                           "Old crossmodel arrays record widths/coverage but no saved query-level residuals: corrected q-error tails cannot be recovered.",
                           "CI across overlapping random splits describes split sensitivity, not 50 independent workload samples."]
    stats["budgets"] = {"default_repair": {"train": 500, "calibration": 500, "total_new": 1000},
                        "oracle": {"train": 2000, "calibration": 500, "total_new": 2500},
                        "copula": {"train": 500, "calibration": 400, "total_new": 900},
                        "stats_crossmodel_current_source": {"train": 500, "calibration": 500, "total_new": 1000}}
    source_paths = [OLD / "results_decouple_reps.npy", OLD / "verify_audit.npz", OLD / "baselines_ci.npz", NEW / "residual_sensitivity_raw.npz"]
    stats["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    (OUT / "statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({key: stats[key] for key in ("sensitivity", "oracle", "plan_tail")}, indent=2))


if __name__ == "__main__":
    main()
