# -*- coding: utf-8 -*-
"""相关性漂移实验：证明 ALECE(直方图更新=no-op)修不好、conformal+LGBM 修得好。

漂移 = 多列过滤查询的联合选择性偏移（单列过滤不变 => 边际直方图不变 => ALECE 更新无意义）。
输出额外 split 成"单列过滤 vs 多列过滤"，看相关性漂移到底打在哪、LGBM 修在哪。
"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, data, model, conformal, drift

def main():
    rng = np.random.RandomState(config.SEED)
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    print(f"加载 {len(true)} 条查询")

    (f_tr, b_tr, y_tr, _), (f_cal, b_cal, y_cal, _), (f_te, b_te, y_te, nfc_te) = \
        data.split(feat, base, true, nfc, config.N_TRAIN, config.N_CAL, config.N_TEST, rng)

    m_old = model.fit_corrector(f_tr, b_tr, y_tr, config.LGBM_PARAMS)
    q_old = conformal.calibrate_q(conformal.score(y_cal, b_cal, model.predict_corrector(m_old, f_cal)), config.ALPHA)
    cov_base = conformal.coverage(conformal.score(y_te, b_te, model.predict_corrector(m_old, f_te)), q_old)
    print(f"[无漂移] q={q_old:.3f} (区间 {np.exp(q_old):.1f}x) 覆盖={cov_base:.1%}\n")

    # 单列 vs 多列 mask
    single = nfc_te <= 1
    multi  = nfc_te >= 2
    print(f"测试集: 单列过滤 {single.sum()} 条, 多列过滤 {multi.sum()} 条\n")

    print(f"{'漂移(相关因子)':>14} | {'冻结模型整体':>12} | {'多列(冻结)':>10} | {'多列(适应)':>10} | {'单列(冻结)':>10}")
    print("-" * 72)
    rows = []
    for mean in config.DRIFT_MEANS:
        # 相关性漂移：只动多列过滤查询
        y_drift = drift.inject_correlation_drift(y_te, nfc_te, mean, rng)

        # 冻结模型（等价 ALECE：边际没变，直方图更新=no-op）
        cov_all = conformal.coverage(conformal.score(y_drift, b_te, model.predict_corrector(m_old, f_te)), q_old)
        cov_multi_frozen = conformal.coverage(conformal.score(y_drift[multi], b_te[multi], model.predict_corrector(m_old, f_te[multi])), q_old)
        cov_single_frozen = conformal.coverage(conformal.score(y_drift[single], b_te[single], model.predict_corrector(m_old, f_te[single])), q_old)

        # 自适应：重训 LGBM + 重校准
        perm = rng.permutation(len(f_te))
        a_tr = perm[:config.N_ADAPT_TRAIN]
        a_cal = perm[config.N_ADAPT_TRAIN : config.N_ADAPT_TRAIN + config.N_ADAPT_CAL]
        a_ev = perm[config.N_ADAPT_TRAIN + config.N_ADAPT_CAL :]
        m_new = model.fit_corrector(f_te[a_tr], b_te[a_tr], y_drift[a_tr], config.LGBM_PARAMS)
        q_new = conformal.calibrate_q(conformal.score(y_drift[a_cal], b_te[a_cal], model.predict_corrector(m_new, f_te[a_cal])), config.ALPHA)
        # 多列子集上的适应后覆盖
        a_ev_multi = a_ev[nfc_te[a_ev] >= 2]
        cov_multi_adapt = conformal.coverage(conformal.score(y_drift[a_ev_multi], b_te[a_ev_multi], model.predict_corrector(m_new, f_te[a_ev_multi])), q_new)

        g = float(np.exp(mean)); rows.append((g, cov_all, cov_multi_frozen, cov_multi_adapt, cov_single_frozen))
        print(f"{g:>13.2f}x | {cov_all:>12.1%} | {cov_multi_frozen:>10.1%} | {cov_multi_adapt:>10.1%} | {cov_single_frozen:>10.1%}")

    np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_corr.npy"), np.array(rows))
    print("\n已存 results_corr.npy (列: [相关因子, 整体冻结, 多列冻结, 多列适应, 单列冻结])")

if __name__ == "__main__":
    main()
