# -*- coding: utf-8 -*-
"""工作负载漂移：训练在简单(2-3表)查询上，测试在复杂(5+表)查询上（查询分布变复杂）。
用固定采样数，不依赖 config.N_TRAIN（那是全数据流的大值）。"""
import os, sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, features, model, conformal

N_TRAIN_WL, N_CAL_WL = 1500, 300   # 固定：简单查询上训练+校准
N_ADAPT, N_ADAPT_CAL = 200, 200    # 自适应

def load(pkl_path):
    n_ts, n_cs = features.load_table_col_space(pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    base_l, true_l, feat_l, ntab_l = [], [], [], []
    for ft, label, pgv, *_ , ex in data["features"]:
        true = float(np.expm1(label - 1)); pg = float(np.expm1(pgv - 1))
        if true <= 0 or pg <= 0: continue
        f = np.nan_to_num(np.concatenate([features.build_schema_feature(ft, ex, n_ts, n_cs), features.stats9(ft)]),
                          nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        f = np.hstack([f, np.log(np.clip(pg, 1, None))]).astype(np.float32)
        base_l.append(pg); true_l.append(true); feat_l.append(f)
        ntab_l.append(len(ex.get("table_indices", [])))
    return (np.array(base_l), np.array(true_l), np.array(feat_l, dtype=np.float32), np.array(ntab_l, dtype=np.int32))

def main():
    rng = np.random.RandomState(config.SEED)
    base, true, feat, ntab = load(config.DATA_PATH)
    simple = ntab <= 3; complex_ = ntab >= 5
    print(f"简单(2-3表): {simple.sum()}, 复杂(5+表): {complex_.sum()}\n")

    tr_pool = np.where(simple)[0]
    tr_sel = rng.choice(tr_pool, size=min(N_TRAIN_WL, len(tr_pool)), replace=False)
    rest = np.setdiff1d(tr_pool, tr_sel)
    cal_sel = rng.choice(rest, size=min(N_CAL_WL, len(rest)), replace=False)
    print(f"训练 {len(tr_sel)}, 校准 {len(cal_sel)}")

    m_old = model.fit_corrector(feat[tr_sel], base[tr_sel], true[tr_sel], config.LGBM_PARAMS)
    q_old = conformal.calibrate_q(conformal.score(true[cal_sel], base[cal_sel], model.predict_corrector(m_old, feat[cal_sel])), config.ALPHA)
    cov_simple = conformal.coverage(conformal.score(true[cal_sel], base[cal_sel], model.predict_corrector(m_old, feat[cal_sel])), q_old)
    print(f"[训练集(简单查询)校准] 覆盖={cov_simple:.1%} (区间 {np.exp(q_old):.1f}x)\n")

    te_pool = np.where(complex_)[0]
    te = rng.choice(te_pool, size=min(500, len(te_pool)), replace=False)
    cov_drift = conformal.coverage(conformal.score(true[te], base[te], model.predict_corrector(m_old, feat[te])), q_old)
    print(f"[工作负载漂移→复杂查询] 覆盖={cov_drift:.1%} (从 {cov_simple:.1%} 掉 {cov_simple-cov_drift:+.1%})")

    a_tr = te[:N_ADAPT]; a_cal = te[N_ADAPT:N_ADAPT+N_ADAPT_CAL]; a_ev = te[N_ADAPT+N_ADAPT_CAL:]
    m_new = model.fit_corrector(feat[a_tr], base[a_tr], true[a_tr], config.LGBM_PARAMS)
    q_new = conformal.calibrate_q(conformal.score(true[a_cal], base[a_cal], model.predict_corrector(m_new, feat[a_cal])), config.ALPHA)
    cov_adapt = conformal.coverage(conformal.score(true[a_ev], base[a_ev], model.predict_corrector(m_new, feat[a_ev])), q_new)
    print(f"[自适应后] 覆盖={cov_adapt:.1%} (恢复 {cov_adapt-cov_drift:+.1%})")

if __name__ == "__main__":
    main()
