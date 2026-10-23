# -*- coding: utf-8 -*-
"""混合漂移：尺度(改统计) + 相关性(多列真值偏移) 同时发生（真实数据多因漂移）。"""
import os, sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, features, model, conformal

def load_raw(pkl_path):
    n_ts, n_cs = features.load_table_col_space(pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    rows = []
    for ft, label, pgv, *_ , ex in data["features"]:
        true = float(np.expm1(label-1)); pg = float(np.expm1(pgv-1))
        if true <= 0 or pg <= 0: continue
        rows.append((ft, true, pg, ex, len(set(ex.get("filter_col_identifiers", [])))))
    return rows, n_ts, n_cs

def extract(ft, pg, extras, n_ts, n_cs):
    f = np.nan_to_num(np.concatenate([features.build_schema_feature(ft, extras, n_ts, n_cs), features.stats9(ft)]),
                      nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    return np.hstack([f, np.log(np.clip(pg,1,None))]).astype(np.float32)

def mixed_drift(ft, true, pg, nfc, scale, corr):
    jh, ff, tf, ftf = ft
    tf = np.array(tf, dtype=np.float64).reshape(-1,4).copy(); tf[:,0] += np.log(scale)
    ft2 = (jh, ff, tf.astype(np.float32), ftf)
    factor = scale * (corr if nfc >= 2 else 1.0)
    return ft2, true*factor, pg*scale

def main():
    rng = np.random.RandomState(config.SEED)
    rows, n_ts, n_cs = load_raw(config.DATA_PATH)
    true = np.array([r[1] for r in rows]); pg = np.array([r[2] for r in rows])
    n = len(rows)
    base_feat = np.array([extract(r[0], r[2], r[3], n_ts, n_cs) for r in rows], dtype=np.float32)
    nfc = np.array([r[4] for r in rows], dtype=np.int32)

    idx = rng.permutation(n)
    tr = idx[:config.N_TRAIN]; cal = idx[config.N_TRAIN:config.N_TRAIN+config.N_CAL]; te = idx[config.N_TRAIN+config.N_CAL:config.N_TRAIN+config.N_CAL+config.N_TEST]

    m_old = model.fit_corrector(base_feat[tr], true[tr], true[tr], config.LGBM_PARAMS)  # NOTE: base=pg passed as 'base' but we use true for resid — fix below
    # 上面用错了，正确：修正器学 log(true/pg)
    m_old = model.fit_corrector(base_feat[tr], pg[tr], true[tr], config.LGBM_PARAMS)
    q_old = conformal.calibrate_q(conformal.score(true[cal], pg[cal], model.predict_corrector(m_old, base_feat[cal])), config.ALPHA)
    cov0 = conformal.coverage(conformal.score(true[te], pg[te], model.predict_corrector(m_old, base_feat[te])), q_old)
    print(f"[原始] 覆盖={cov0:.1%} (区间 {np.exp(q_old):.1f}x)\n")

    print(f"{'尺度×':>8}{'相关×':>8} | {'覆盖(混合漂移后)':>16} | {'恢复':>8}")
    print("-" * 48)
    for scale, corr in [(1.0,1.0),(2.0,3.0),(3.0,5.0)]:
        nt, true_d, pg_d = [], [], []
        for r in rows:
            ft2, t2, p2 = mixed_drift(r[0], r[1], r[2], r[4], scale, corr)
            nt.append(extract(ft2, p2, r[3], n_ts, n_cs)); true_d.append(t2); pg_d.append(p2)
        Fd = np.array(nt, dtype=np.float32); true_d = np.array(true_d); pg_d = np.array(pg_d)
        cov_d = conformal.coverage(conformal.score(true_d[te], pg_d[te], model.predict_corrector(m_old, Fd[te])), q_old)
        # 自适应
        perm = rng.permutation(len(te)); at=te[perm[:config.N_ADAPT_TRAIN]]; ac=te[perm[config.N_ADAPT_TRAIN:config.N_ADAPT_TRAIN+config.N_ADAPT_CAL]]; ae=te[perm[config.N_ADAPT_TRAIN+config.N_ADAPT_CAL:]]
        m_new = model.fit_corrector(Fd[at], pg_d[at], true_d[at], config.LGBM_PARAMS)
        q_new = conformal.calibrate_q(conformal.score(true_d[ac], pg_d[ac], model.predict_corrector(m_new, Fd[ac])), config.ALPHA)
        cov_a = conformal.coverage(conformal.score(true_d[ae], pg_d[ae], model.predict_corrector(m_new, Fd[ae])), q_new)
        print(f"{scale:>7.1f}x{corr:>7.1f}x | {cov_d:>16.1%} | {cov_a-cov_d:>+7.1%}")

if __name__ == "__main__":
    main()
