# -*- coding: utf-8 -*-
"""覆盖-宽度 trade-off：不同 alpha 下的 coverage vs width。
证明区间是校准的、可调宽。无漂移 + 相关性漂移 7.39x 两个场景。
输出到 results_repro/experiment_alphatradeoff.txt
"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, data, model, conformal

ALPHAS = [0.01, 0.05, 0.10, 0.20, 0.30]

def main():
    rng = np.random.RandomState(config.SEED)
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    (f_tr, b_tr, y_tr, _), (f_cal, b_cal, y_cal, _), (f_te, b_te, y_te, nfc_te) = \
        data.split(feat, base, true, nfc, config.N_TRAIN, config.N_CAL, config.N_TEST, rng)

    m = model.fit_corrector(f_tr, b_tr, y_tr, config.LGBM_PARAMS)
    corr_te = model.predict_corrector(m, f_te)
    scores_cal = conformal.score(y_cal, b_cal, model.predict_corrector(m, f_cal))
    scores_te_nodrift = conformal.score(y_te, b_te, corr_te)
    # 相关性漂移 7.39x（多列过滤查询乘 exp(2.0)）
    y_drift = y_te * np.where(nfc_te >= 2, np.exp(2.0), 1.0)
    scores_te_drift = conformal.score(y_drift, b_te, corr_te)

    out = []
    def log(s):
        out.append(s); print(s, flush=True)
    log("alpha | 无漂移覆盖 | 无漂移宽 | 漂移后覆盖 | 漂移后宽")
    log("-" * 52)
    for a in ALPHAS:
        q = conformal.calibrate_q(scores_cal, a)
        cov0 = conformal.coverage(scores_te_nodrift, q)
        cov1 = conformal.coverage(scores_te_drift, q)
        log(f"{a:.2f} | {cov0:6.1%} | {np.exp(q):7.1f}x | {cov1:6.1%} | {np.exp(q):7.1f}x")

    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_repro")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "experiment_alphatradeoff.txt"), "w", encoding="utf-8") as f:
        f.write("exit=0\n" + "\n".join(out) + "\n")
    print("\nsaved to results_repro/experiment_alphatradeoff.txt", flush=True)

if __name__ == "__main__":
    main()
