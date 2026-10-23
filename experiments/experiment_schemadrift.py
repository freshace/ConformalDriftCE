# -*- coding: utf-8 -*-
"""unseen-table-set / fixed-feature-space schema shift 测试：
训练查询不含某张既有表、评估查询含该表；特征空间由全量语料预定义且维度不变。
这不是新增表/列的真实 schema evolution，后者仍未解决。
训练在不含某表的查询上，测试在含该表的查询上。"""
import os, sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, features, model, conformal

def load(pkl_path):
    n_ts, n_cs = features.load_table_col_space(pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    base_l, true_l, feat_l, tabsets = [], [], [], []
    for ft, label, pgv, *_ , ex in data["features"]:
        true = float(np.expm1(label-1)); pg = float(np.expm1(pgv-1))
        if true <= 0 or pg <= 0: continue
        f = np.nan_to_num(np.concatenate([features.build_schema_feature(ft, ex, n_ts, n_cs), features.stats9(ft)]),
                          nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        f = np.hstack([f, np.log(np.clip(pg,1,None))]).astype(np.float32)
        base_l.append(pg); true_l.append(true); feat_l.append(f)
        tabsets.append(set(int(t) for t in ex.get("table_indices", [])))
    return (np.array(base_l), np.array(true_l), np.array(feat_l, dtype=np.float32), tabsets)

def main():
    rng = np.random.RandomState(config.SEED)
    base, true, feat, tabsets = load(config.DATA_PATH)
    n = len(true)
    # 统计每个表出现在多少个查询
    from collections import Counter
    occ = Counter()
    for ts in tabsets:
        for t in ts: occ[t] += 1
    # 挑"有足够多含它/不含它"的表（平衡切分）
    best = None
    for t, c in occ.items():
        no_c = n - c
        if 3000 <= c <= n-3000:  # 含/不含都足够
            if best is None or abs(c - no_c) < abs(best[1] - (n-best[1])):
                best = (t, c)
    if best is None:
        print("无平衡切分的表"); return
    t = best[0]
    with_t = np.array([t in ts for ts in tabsets])
    print(f"选表 index={t}: 含它 {int(with_t.sum())}, 不含 {int((~with_t).sum())} 条\n")

    tr_pool = np.where(~with_t)[0]
    tr_sel = rng.choice(tr_pool, size=min(config.N_TRAIN, len(tr_pool)), replace=False)
    rest = np.setdiff1d(tr_pool, tr_sel)
    cal_sel = rng.choice(rest, size=min(config.N_CAL, len(rest)), replace=False)
    m_old = model.fit_corrector(feat[tr_sel], base[tr_sel], true[tr_sel], config.LGBM_PARAMS)
    q_old = conformal.calibrate_q(conformal.score(true[cal_sel], base[cal_sel], model.predict_corrector(m_old, feat[cal_sel])), config.ALPHA)
    cov_train = conformal.coverage(conformal.score(true[cal_sel], base[cal_sel], model.predict_corrector(m_old, feat[cal_sel])), q_old)
    print(f"[训练集(不含该表)校准] 覆盖={cov_train:.1%} (区间 {np.exp(q_old):.1f}x)")

    te_pool = np.where(with_t)[0]
    te = rng.choice(te_pool, size=min(1000, len(te_pool)), replace=False)
    cov_drift = conformal.coverage(conformal.score(true[te], base[te], model.predict_corrector(m_old, feat[te])), q_old)
    print(f"[schema漂移→含新表查询] 覆盖={cov_drift:.1%} (掉 {cov_train-cov_drift:+.1%})")

    a_tr=te[:200]; a_cal=te[200:400]; a_ev=te[400:]
    m_new = model.fit_corrector(feat[a_tr], base[a_tr], true[a_tr], config.LGBM_PARAMS)
    q_new = conformal.calibrate_q(conformal.score(true[a_cal], base[a_cal], model.predict_corrector(m_new, feat[a_cal])), config.ALPHA)
    cov_adapt = conformal.coverage(conformal.score(true[a_ev], base[a_ev], model.predict_corrector(m_new, feat[a_ev])), q_new)
    print(f"[自适应后] 覆盖={cov_adapt:.1%} (恢复 {cov_adapt-cov_drift:+.1%})")

if __name__ == "__main__":
    main()
