# -*- coding: utf-8 -*-
"""P0.5: residual corrector 的必要性。

比较三条清晰的臂：
  1) Base + Conformal：原始底模点预测 + split conformal；
  2) Base + Residual：残差修正后的点预测，不把 conformal 增益混入点精度；
  3) Base + Residual + Conformal：残差修正后重新校准区间。

所有臂共用同一 train/cal/frozen-eval/adaptation-pool 切分。漂移只作用于
多过滤列查询，adaptation pool 与最终 audit 集不重叠。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import data
import drift
import model
import conformal


def build_arm_estimate(base, correction):
    """Return the point estimate for Base or Base+Residual."""
    base = np.asarray(base, dtype=np.float64)
    if correction is None:
        return base.copy()
    return base * np.exp(np.clip(np.asarray(correction, dtype=np.float64), -30.0, 30.0))


def qerror_summary(true, estimate):
    true = np.asarray(true, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    ratio = np.maximum(true, estimate) / np.maximum(np.minimum(true, estimate), 1e-12)
    return {
        "q50": float(np.quantile(ratio, 0.50)),
        "q90": float(np.quantile(ratio, 0.90)),
        "gmq": float(np.exp(np.mean(np.abs(np.log(ratio))))),
    }


def summarize_interval(true, estimate, q):
    scores = np.abs(np.log(np.asarray(true)) - np.log(np.clip(np.asarray(estimate), 1.0, None)))
    return {
        "coverage": float(np.mean(scores <= q)),
        "width": float(np.exp(q)),
        "mean_score": float(np.mean(scores)),
    }


def split_adaptation_labels(n, seed, train_size, cal_size):
    if train_size + cal_size > n:
        raise ValueError("adaptation split exceeds pool size")
    order = np.random.RandomState(seed).permutation(n)
    return order[:train_size], order[train_size:train_size + cal_size], order[train_size + cal_size:]


def _fit_residual(feat, base, true):
    corrector = model.fit_corrector(feat, base, true, config.LGBM_PARAMS)
    return build_arm_estimate(base, model.predict_corrector(corrector, feat)), corrector


def _ci(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(values.mean()), 0.0
    return float(values.mean()), float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def _format_ci(values, scale=1.0, suffix=""):
    mean, half = _ci(np.asarray(values) * scale)
    return f"{mean:.3f}+/-{half:.3f}{suffix}"


def run_repetition(base, true, feat, nfc, repetition, budgets):
    rng = np.random.RandomState(config.SEED + repetition)
    (f_tr, b_tr, y_tr, _), (f_cal, b_cal, y_cal, _), (f_te, b_te, y_te, nfc_te) = data.split(
        feat, base, true, nfc, config.N_TRAIN, config.N_CAL, config.N_TEST, rng)

    # Keep the final audit separate from all adaptation labels.
    audit_size = config.N_TEST // 2
    audit_slice = slice(0, audit_size)
    pool_slice = slice(audit_size, None)
    f_audit, b_audit, y_audit = f_te[audit_slice], b_te[audit_slice], y_te[audit_slice]
    nfc_audit = nfc_te[audit_slice]
    f_pool, b_pool, y_pool = f_te[pool_slice], b_te[pool_slice], y_te[pool_slice]
    nfc_pool = nfc_te[pool_slice]
    y_audit_drift = drift.inject_correlation_drift(y_audit, nfc_audit, 2.0, rng)
    y_pool_drift = drift.inject_correlation_drift(y_pool, nfc_pool, 2.0, rng)

    base_old = build_arm_estimate(b_tr, None)
    residual_old, corrector_old = _fit_residual(f_tr, b_tr, y_tr)
    base_cal = build_arm_estimate(b_cal, None)
    residual_cal = build_arm_estimate(
        b_cal, model.predict_corrector(corrector_old, f_cal))
    q_base = conformal.calibrate_q(
        conformal.score(y_cal, b_cal, np.zeros(len(y_cal))), config.ALPHA)
    q_residual = conformal.calibrate_q(
        conformal.score(y_cal, b_cal, model.predict_corrector(corrector_old, f_cal)), config.ALPHA)

    base_audit = build_arm_estimate(b_audit, None)
    residual_audit = build_arm_estimate(
        b_audit, model.predict_corrector(corrector_old, f_audit))
    results = {
        "point_no_drift_base": qerror_summary(y_audit, base_audit),
        "point_no_drift_residual": qerror_summary(y_audit, residual_audit),
        "point_drift_base": qerror_summary(y_audit_drift, base_audit),
        "point_drift_residual": qerror_summary(y_audit_drift, residual_audit),
        "interval_no_drift_base": summarize_interval(y_audit, base_audit, q_base),
        "interval_no_drift_residual": summarize_interval(y_audit, residual_audit, q_residual),
        "interval_drift_base": summarize_interval(y_audit_drift, base_audit, q_base),
        "interval_drift_residual": summarize_interval(y_audit_drift, residual_audit, q_residual),
        "label_efficiency": {},
    }

    for budget in budgets:
        train_size = budget // 2
        cal_size = budget - train_size
        train_idx, cal_idx, audit_idx = split_adaptation_labels(
            len(y_pool_drift), config.SEED + repetition * 1000 + budget,
            train_size, cal_size)
        if len(audit_idx) == 0:
            raise ValueError("adaptation budget leaves no audit examples")

        # Base+Conformal: spend all N labels on a fresh q; no residual training.
        calibration_idx = np.concatenate([train_idx, cal_idx])
        q_base_new = conformal.calibrate_q(
            conformal.score(y_pool_drift[calibration_idx], b_pool[calibration_idx],
                            np.zeros(len(calibration_idx))),
            config.ALPHA)
        base_eff = summarize_interval(y_pool_drift[audit_idx], b_pool[audit_idx], q_base_new)

        # Base+Residual+Conformal: split the same budget between residual training and q.
        residual_new, corrector_new = _fit_residual(
            f_pool[train_idx], b_pool[train_idx], y_pool_drift[train_idx])
        residual_cal_est = build_arm_estimate(
            b_pool[cal_idx], model.predict_corrector(corrector_new, f_pool[cal_idx]))
        q_residual_new = conformal.calibrate_q(
            conformal.score(y_pool_drift[cal_idx], b_pool[cal_idx],
                            model.predict_corrector(corrector_new, f_pool[cal_idx])),
            config.ALPHA)
        residual_audit_est = build_arm_estimate(
            b_pool[audit_idx], model.predict_corrector(corrector_new, f_pool[audit_idx]))
        residual_eff = summarize_interval(
            y_pool_drift[audit_idx], residual_audit_est, q_residual_new)
        results["label_efficiency"][int(budget)] = {
            "base_conformal": base_eff,
            "residual_conformal": residual_eff,
            "base_width": float(base_eff["width"]),
            "residual_width": float(residual_eff["width"]),
            "residual_point": qerror_summary(y_pool_drift[audit_idx], residual_audit_est),
        }
    return results


def main():
    repetitions = int(os.environ.get("P0_RESIDUAL_R", "20"))
    budgets = [int(x) for x in os.environ.get("P0_RESIDUAL_BUDGETS", "50,100,200,500,1000").split(",")]
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    all_results = [run_repetition(base, true, feat, nfc, r, budgets)
                   for r in range(repetitions)]
    out = []

    def log(line):
        out.append(line)
        print(line, flush=True)

    log(f"loaded {len(true)} queries; R={repetitions}; budgets={budgets}")
    log("All arms use identical train/cal/audit/pool splits; drift is correlation 7.39x.")
    log("== Point accuracy (q-error; lower is better) ==")
    for key, label in [
        ("point_no_drift_base", "Base"),
        ("point_no_drift_residual", "Base+Residual"),
        ("point_drift_base", "Base under drift"),
        ("point_drift_residual", "Base+Residual under drift"),
    ]:
        for metric in ("q50", "q90", "gmq"):
            log(f"{label:28s} {metric:4s} {_format_ci([r[key][metric] for r in all_results])}")

    log("== Same-coverage interval comparison (target 90%; width is multiplicative half-width) ==")
    for key, label in [
        ("interval_no_drift_base", "Base+Conformal no drift"),
        ("interval_no_drift_residual", "Residual+Conformal no drift"),
        ("interval_drift_base", "Base+Conformal drift"),
        ("interval_drift_residual", "Residual+Conformal drift"),
    ]:
        log(f"{label:28s} coverage={_format_ci([r[key]['coverage'] for r in all_results], 100, '%')} "
            f"width={_format_ci([r[key]['width'] for r in all_results], 1, 'x')}")

    log("== Label efficiency on independent drift audit ==")
    for budget in budgets:
        base_cov = [r["label_efficiency"][budget]["base_conformal"]["coverage"] for r in all_results]
        residual_cov = [r["label_efficiency"][budget]["residual_conformal"]["coverage"] for r in all_results]
        base_width = [r["label_efficiency"][budget]["base_width"] for r in all_results]
        residual_width = [r["label_efficiency"][budget]["residual_conformal"]["width"] for r in all_results]
        log(f"N={budget:4d} Base+Conformal coverage={_format_ci(base_cov, 100, '%')} "
            f"width={_format_ci(base_width, 1, 'x')} "
            f"Residual+Conformal coverage={_format_ci(residual_cov, 100, '%')} "
            f"width={_format_ci(residual_width, 1, 'x')}")

    for arm in ("base_conformal", "residual_conformal"):
        first = next((b for b in budgets if np.mean([
            r["label_efficiency"][b][arm]["coverage"] for r in all_results]) >= 1.0 - config.ALPHA), None)
        log(f"first budget with mean coverage >= 90% ({arm}) = {first}")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_repro")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "experiment_residual_necessity.txt"), "w", encoding="utf-8") as handle:
        handle.write("exit=0\n" + "\n".join(out) + "\n")
    print("saved results_repro/experiment_residual_necessity.txt", flush=True)


if __name__ == "__main__":
    main()