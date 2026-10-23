# -*- coding: utf-8 -*-
"""尺度漂移（改统计+重算特征）：表大小/真值/pg 一起按比例变，residual 不变 -> 应被修正器吸收。
对比 experiment_correlation.py（特征不变、联合真值变 -> 检测到）。
"""
import os, sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, features, model, conformal

def load_raw(pkl_path):
    """返回 per-query: (ft, label, pgv, extras, nfc) —— 保留原始 ft 便于改统计。"""
    n_ts, n_cs = features.load_table_col_space(pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    rows = []
    for ft, label, pgv, *_ , ex in data["features"]:
        true = float(np.expm1(label - 1)); pg = float(np.expm1(pgv - 1))
        if true <= 0 or pg <= 0: continue
        rows.append((ft, float(true), float(pg), ex, len(set(ex.get("filter_col_identifiers", [])))))
    return rows, n_ts, n_cs

def extract(ft, true, pg, extras, n_ts, n_cs):
    f = np.nan_to_num(np.concatenate([features.build_schema_feature(ft, extras, n_ts, n_cs), features.stats9(ft)]),
                      nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    f = np.hstack([f, np.log(np.clip(pg,1,None))]).astype(np.float32)
    return f

def scale_drift(ft, true, pg, X):
    """尺度漂移：表大小 tf[:,0] += logX，真值和 pg 同步乘以 X（residual 不变）。"""
    jh, ff, tf, ftf = ft
    tf = np.array(tf, dtype=np.float64).reshape(-1, 4).copy()
    tf[:,0] += np.log(X)
    return (jh, ff, tf.astype(np.float32), ftf), true * X, pg * X

def main():
    rng = np.random.RandomState(config.SEED)
    rows, n_ts, n_cs = load_raw(config.DATA_PATH)
    true = np.array([r[1] for r in rows]); pg = np.array([r[2] for r in rows])
    n = len(rows); print(f"加载 {n} 条查询, 特征空间 ({n_ts}, {n_cs})")

    def feats():
        return np.array([extract(r[0], r[1], r[2], r[3], n_ts, n_cs) for r in rows], dtype=np.float32)

    F0 = feats()
    idx = rng.permutation(n)
    tr = idx[:config.N_TRAIN]; cal = idx[config.N_TRAIN:config.N_TRAIN+config.N_CAL]; te = idx[config.N_TRAIN+config.N_CAL:config.N_TRAIN+config.N_CAL+config.N_TEST]

    # 旧模型（原始数据）
    resid = np.log(true[tr]) - np.log(np.clip(pg[tr],1,None))
    m_old = model.fit_corrector(F0[tr], pg[tr], true[tr], config.LGBM_PARAMS)
    q_old = conformal.calibrate_q(conformal.score(true[cal], pg[cal], model.predict_corrector(m_old, F0[cal])), config.ALPHA)
    cov0 = conformal.coverage(conformal.score(true[te], pg[te], model.predict_corrector(m_old, F0[te])), q_old)
    print(f"[原始] 区间 {np.exp(q_old):.1f}x  覆盖={cov0:.1%}\n")

    print(f"{'尺度漂移X':>10} | {'覆盖(尺度漂移后)':>16} | {'残差是否变':>10}")
    print("-" * 44)
    for X in [1.0, 2.0, 5.0, 10.0, 50.0]:
        # 对每一条：改统计 + 重算特征 + 同步真值/pg
        new_feat = []; new_true = []; new_pg = []
        for r in rows:
            ft2, t2, p2 = scale_drift(r[0], r[1], r[2], X)
            new_feat.append(extract(ft2, t2, p2, r[3], n_ts, n_cs))
            new_true.append(t2); new_pg.append(p2)
        Fx = np.array(new_feat, dtype=np.float32)
        true_x = np.array(new_true); pg_x = np.array(new_pg)
        # 残差变化
        resid_old = true[te]/pg[te]; resid_new = true_x[te]/pg_x[te]
        resid_chg = np.mean(np.abs(np.log(resid_new) - np.log(resid_old)))
        cov_x = conformal.coverage(conformal.score(true_x[te], pg_x[te], model.predict_corrector(m_old, Fx[te])), q_old)
        print(f"{X:>9.1f}x | {cov_x:>16.1%} | {resid_chg:>10.4f}")

if __name__ == "__main__":
    main()
