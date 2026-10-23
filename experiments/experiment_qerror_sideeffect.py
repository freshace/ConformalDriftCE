# -*- coding: utf-8 -*-
"""q-error 副作用实验：证明残差修正器在改善可靠性的同时，顺带改善点估计精度。
场景：相关性漂移 7.39x。对比三种点估计的 q-error（q50/q90/q99）：
  - PG 底模（无修正）
  - 冻结（旧修正器，漂移后不更新）
  - 我们（500 标签适应后）
输出到 results_repro/experiment_qerror_sideeffect.txt
"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, data, model, conformal

def qerr(est, true):
    e = np.clip(est, 1e-6, None); t = np.clip(true, 1e-6, None)
    return np.maximum(e/t, t/e)

def main():
    rng = np.random.RandomState(config.SEED)
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    (f_tr, b_tr, y_tr, _), (f_cal, b_cal, y_cal, _), (f_te, b_te, y_te, nfc_te) = \
        data.split(feat, base, true, nfc, config.N_TRAIN, config.N_CAL, config.N_TEST, rng)

    m_old = model.fit_corrector(f_tr, b_tr, y_tr, config.LGBM_PARAMS)
    # 相关性漂移 7.39x
    y_drift = y_te * np.where(nfc_te >= 2, np.exp(2.0), 1.0)

    # 1) PG 底模（无修正）点估计 q-error
    qe_base = qerr(b_te, y_drift)
    # 2) 冻结：旧修正器（漂移后不更新）
    est_frozen = b_te * np.exp(model.predict_corrector(m_old, f_te))
    qe_frozen = qerr(est_frozen, y_drift)
    # 3) 我们：500 标签适应
    perm = rng.permutation(len(f_te))
    a_tr = perm[:config.N_ADAPT_TRAIN]; a_cal = perm[config.N_ADAPT_TRAIN:config.N_ADAPT_TRAIN+config.N_ADAPT_CAL]; a_ev = perm[config.N_ADAPT_TRAIN+config.N_ADAPT_CAL:]
    m_new = model.fit_corrector(f_te[a_tr], b_te[a_tr], y_drift[a_tr], config.LGBM_PARAMS)
    est_ours = b_te[a_ev] * np.exp(model.predict_corrector(m_new, f_te[a_ev]))
    qe_ours = qerr(est_ours, y_drift[a_ev])

    out = []
    def log(s):
        out.append(s); print(s, flush=True)
    def qs(name, qe):
        log(f"{name:12s} | q50={np.percentile(qe,50):7.1f} | q90={np.percentile(qe,90):7.1f} | q99={np.percentile(qe,99):9.1f}")
    log("=== 相关性漂移 7.39x 下的点估计 q-error（副作用）===")
    qs("PG底模", qe_base)
    qs("冻结(旧修正)", qe_frozen)
    qs("我们(500适应)", qe_ours)

    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_repro")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "experiment_qerror_sideeffect.txt"), "w", encoding="utf-8") as f:
        f.write("exit=0\n" + "\n".join(out) + "\n")
    print("\nsaved to results_repro/experiment_qerror_sideeffect.txt", flush=True)

if __name__ == "__main__":
    main()
