# -*- coding: utf-8 -*-
"""P0.5b: Residual 的 coverage-matched 区间效率与可靠性监控灵敏度。

1) Coverage-matched interval comparison：逐重复给 Base+Conformal 单独扫 alpha，
   使独立验证集覆盖对齐 Residual+Conformal，再用最终测试集比较覆盖与 width。
   目的：证明 Residual 的窄区间不是"牺牲 coverage 换来的"。
2) Drift dose-response：相关因子 1.0x--7.39x 下两臂冻结 coverage 与
   signal = 90% - coverage。把 reliability-monitoring sensitivity 从单点观察
   变成剂量-响应曲线。不声称 detector TPR/FPR/delay 的改善。
"""
import os
import json
import hashlib
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import data
import drift
import model
import conformal

DOSES = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]  # log 空间；2.0 => 7.39x
ALPHA_GRID = np.round(np.arange(0.01, 0.60, 0.005), 4)


def coverage_at(scores, q):
    return float(np.mean(scores <= q))


def match_q(scores_cal, scores_validation, target):
    """Choose alpha on independent validation scores; never pass final test scores."""
    best = None
    for a in ALPHA_GRID:
        q = conformal.calibrate_q(scores_cal, float(a))
        cov = coverage_at(scores_validation, q)
        diff = abs(cov - target)
        if best is None or diff < best[0]:
            best = (diff, q, cov, float(a))
    return best[1], best[2], best[3]


def split_indices(n, n_train, n_cal, n_validation, n_test, rng):
    sizes = (n_train, n_cal, n_validation, n_test)
    if min(sizes) <= 0 or sum(sizes) > n:
        raise ValueError("four nonempty disjoint splits require enough queries")
    return tuple(np.split(rng.permutation(n)[:sum(sizes)], np.cumsum(sizes)[:-1]))


def run_repetition(base, true, feat, nfc, repetition):
    rng = np.random.RandomState(config.SEED + repetition)
    tr, cal, val, te = split_indices(len(true), config.N_TRAIN, config.N_CAL,
                                    int(os.environ.get("P0B_VALIDATION", "2000")), config.N_TEST, rng)
    f_tr, b_tr, y_tr = feat[tr], base[tr], true[tr]
    f_cal, b_cal, y_cal = feat[cal], base[cal], true[cal]
    f_te, b_te, y_te, nfc_te = feat[te], base[te], true[te], nfc[te]

    base_cal_scores = conformal.score(y_cal, b_cal, np.zeros(len(y_cal)))
    corrector = model.fit_corrector(f_tr, b_tr, y_tr, config.LGBM_PARAMS)
    corr_cal = model.predict_corrector(corrector, f_cal)
    corr_te = model.predict_corrector(corrector, f_te)
    residual_cal_scores = conformal.score(y_cal, b_cal, corr_cal)

    q_base = conformal.calibrate_q(base_cal_scores, config.ALPHA)
    q_residual = conformal.calibrate_q(residual_cal_scores, config.ALPHA)

    base_nd = conformal.score(y_te, b_te, np.zeros(len(y_te)))
    res_nd = conformal.score(y_te, b_te, corr_te)

    # ---- (1) 无漂移 coverage-matched 对比 ----
    base_val = conformal.score(true[val], base[val], np.zeros(len(val)))
    res_val = conformal.score(true[val], base[val], model.predict_corrector(corrector, feat[val]))
    cov_res_val = coverage_at(res_val, q_residual)
    q_match, cov_match_val, alpha_match = match_q(base_cal_scores, base_val, cov_res_val)
    cov_res_nd = coverage_at(res_nd, q_residual)
    cov_match = coverage_at(base_nd, q_match)
    matched = {
        "residual_coverage": cov_res_nd,
        "residual_width": float(np.exp(q_residual)),
        "base_matched_coverage": cov_match,
        "base_matched_width": float(np.exp(q_match)),
        "base_matched_alpha": alpha_match,
        "base_nominal_coverage": coverage_at(base_nd, q_base),
        "base_nominal_width": float(np.exp(q_base)),
        "residual_validation_coverage": cov_res_val,
        "base_matched_validation_coverage": cov_match_val,
    }

    # ---- (2) 剂量-响应 ----
    rep = {"matched": matched, "dose": [], "base_cov": [], "res_cov": [], "match_cov": [],
           "split_indices": {"train": tr, "calibration": cal, "validation": val, "test": te},
           "base_scores": base_nd, "residual_scores": res_nd}
    for mean in DOSES:
        y_drift = drift.inject_correlation_drift(y_te, nfc_te, mean, rng)
        base_d = conformal.score(y_drift, b_te, np.zeros(len(y_te)))
        res_d = conformal.score(y_drift, b_te, corr_te)
        rep["dose"].append(float(np.exp(mean)))
        rep["base_cov"].append(coverage_at(base_d, q_base))
        rep["res_cov"].append(coverage_at(res_d, q_residual))
        rep["match_cov"].append(coverage_at(base_d, q_match))
    return rep

def _fmt(vals, scale=1.0, prec=3):
    v = np.asarray(vals, float) * scale
    return f"{v.mean():.{prec}f}±{1.96 * v.std(ddof=1) / np.sqrt(len(v)):.{prec}f}"


def main():
    repetitions = int(os.environ.get("P0B_R", "10"))
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    reps = []
    for r in range(repetitions):
        reps.append(run_repetition(base, true, feat, nfc, r))
        print(f"replicate {r + 1}/{repetitions}", flush=True)
    out = []

    def log(line):
        out.append(line)
        print(line, flush=True)

    log(f"loaded {len(true)} queries; R={repetitions}; doses(log)={DOSES}; correlation drift (multi-col only).")
    log("")
    log("== (1) Coverage-matched interval comparison, no drift ==")
    log("(Base alpha chosen only on independent validation; all reported coverage uses final test)")
    log(f"Residual+Conformal (alpha=0.10): coverage={_fmt([r['matched']['residual_coverage'] for r in reps], 100, 2)}%  "
        f"width={_fmt([r['matched']['residual_width'] for r in reps], prec=2)}x")
    log(f"Base+Conformal (alpha matched):  coverage={_fmt([r['matched']['base_matched_coverage'] for r in reps], 100, 2)}%  "
        f"width={_fmt([r['matched']['base_matched_width'] for r in reps], prec=2)}x")
    log(f"Base+Conformal (alpha=0.10):     coverage={_fmt([r['matched']['base_nominal_coverage'] for r in reps], 100, 2)}%  "
        f"width={_fmt([r['matched']['base_nominal_width'] for r in reps], prec=2)}x")
    log(f"matched alpha (per rep): {[round(r['matched']['base_matched_alpha'], 3) for r in reps]}")
    log(f"=> coverage-matched width ratio Base/Residual = "
        f"{_fmt([r['matched']['base_matched_width'] / r['matched']['residual_width'] for r in reps], prec=1)}x")
    log("")
    log("== (2) Drift dose-response: frozen coverage & monitoring signal (90% - coverage) ==")
    log(f"{'factor':>8} | {'Base cov':>16} | {'Base signal':>17} | {'Residual cov':>16} | {'Residual signal':>17} | {'matchBase cov':>16}")
    log("-" * 104)
    rows = []
    for i in range(len(DOSES)):
        bc = [r["base_cov"][i] for r in reps]
        rc = [r["res_cov"][i] for r in reps]
        mc = [r["match_cov"][i] for r in reps]
        factor = reps[0]["dose"][i]
        rows.append((factor, float(np.mean(bc)), float(np.mean(rc))))
        log(f"{factor:7.2f}x | {_fmt(bc, 100, 1) + '%':>16} | {_fmt([0.9 - v for v in bc], 100, 1) + 'pp':>17} | "
            f"{_fmt(rc, 100, 1) + '%':>16} | {_fmt([0.9 - v for v in rc], 100, 1) + 'pp':>17} | "
            f"{_fmt(mc, 100, 1) + '%':>16}")
    log("")
    log("signal gap (Residual - Base, pp):")
    for factor, b, r_ in rows:
        log(f"  {factor:6.2f}x  signal gap = {(b - r_) * 100:+.1f}pp")
    log("")
    log("读法：signal = 90% - frozen coverage，是 coverage-based reliability monitor 的失效信号强度。")
    log("只报告监控信号的经验灵敏度，不声称 drift detection 的 TPR/FPR/delay 改善。")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results_revision_20260930", "figures_sensitivity")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "experiment_residual_sensitivity.txt"), "w", encoding="utf-8") as handle:
        handle.write("exit=0\n" + "\n".join(out) + "\n")
    np.save(os.path.join(output_dir, "results_residual_sensitivity.npy"),
            np.array([[f, b, r_] for f, b, r_ in rows]))
    raw = {key: np.array([r["matched"][key] for r in reps]) for key in reps[0]["matched"]}
    raw.update({key: np.array([r[key] for r in reps]) for key in
                ("dose", "base_cov", "res_cov", "match_cov", "base_scores", "residual_scores")})
    raw.update({"indices_" + key: np.array([r["split_indices"][key] for r in reps])
                for key in reps[0]["split_indices"]})
    np.savez_compressed(os.path.join(output_dir, "residual_sensitivity_raw.npz"), **raw)
    protocol = {"seed_start": config.SEED, "repetitions": repetitions,
                "split_counts": {k: len(v) for k, v in reps[0]["split_indices"].items()},
                "selection": "alpha grid matched to residual validation coverage; final test untouched",
                "alpha_grid": ALPHA_GRID.tolist(), "width": "multiplicative half-width exp(q)",
                "data_sha256": hashlib.sha256(open(config.DATA_PATH, "rb").read()).hexdigest()}
    with open(os.path.join(output_dir, "protocol.json"), "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    print("saved " + output_dir, flush=True)


if __name__ == "__main__":
    main()
