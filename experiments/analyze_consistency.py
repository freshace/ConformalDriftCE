# -*- coding: utf-8 -*-
"""一致性分析：相关性漂移后，按 join 宽度 / 过滤数 拆分 冻结 vs 适应 覆盖率。"""
import os, sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, features, model, conformal, drift

def load_all(pkl_path):
    n_ts, n_cs = features.load_table_col_space(pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    feats = data["features"]
    base_l, true_l, feat_l, nfc_l, ntab_l = [], [], [], [], []
    for ft, label, pgv, *_ , ex in feats:
        true = float(np.expm1(label - 1)); pg = float(np.expm1(pgv - 1))
        if true <= 0 or pg <= 0: continue
        f = np.nan_to_num(np.concatenate([features.build_schema_feature(ft, ex, n_ts, n_cs), features.stats9(ft)]),
                          nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        f = np.hstack([f, np.log(np.clip(pg, 1.0, None))]).astype(np.float32)
        base_l.append(pg); true_l.append(true); feat_l.append(f)
        nfc_l.append(len(set(ex.get("filter_col_identifiers", []))))
        ntab_l.append(len(ex.get("table_indices", [])))
    return (np.array(base_l), np.array(true_l), np.array(feat_l, dtype=np.float32),
            np.array(nfc_l, dtype=np.int32), np.array(ntab_l, dtype=np.int32))

def main():
    rng = np.random.RandomState(config.SEED)
    base, true, feat, nfc, ntab = load_all(config.DATA_PATH)
    n = len(true)
    idx = rng.permutation(n)
    tr = idx[:config.N_TRAIN]
    cal = idx[config.N_TRAIN:config.N_TRAIN+config.N_CAL]
    te = idx[config.N_TRAIN+config.N_CAL : config.N_TRAIN+config.N_CAL+config.N_TEST]

    m_old = model.fit_corrector(feat[tr], base[tr], true[tr], config.LGBM_PARAMS)
    q_old = conformal.calibrate_q(conformal.score(true[cal], base[cal], model.predict_corrector(m_old, feat[cal])), config.ALPHA)

    # 7.39x 相关性漂移
    y_drift = drift.inject_correlation_drift(true[te], nfc[te], 2.0, rng)
    corr_old = model.predict_corrector(m_old, feat[te])
    s_frozen = conformal.score(y_drift, base[te], corr_old)

    # 自适应
    perm = rng.permutation(len(te))
    a_tr = perm[:config.N_ADAPT_TRAIN]
    a_cal = perm[config.N_ADAPT_TRAIN:config.N_ADAPT_TRAIN+config.N_ADAPT_CAL]
    a_ev = perm[config.N_ADAPT_TRAIN+config.N_ADAPT_CAL:]
    m_new = model.fit_corrector(feat[te][a_tr], base[te][a_tr], y_drift[a_tr], config.LGBM_PARAMS)
    q_new = conformal.calibrate_q(conformal.score(y_drift[a_cal], base[te][a_cal], model.predict_corrector(m_new, feat[te][a_cal])), config.ALPHA)
    corr_new = model.predict_corrector(m_new, feat[te][a_ev])
    s_adapt = conformal.score(y_drift[a_ev], base[te][a_ev], corr_new)

    # 冻结：全测试集；适应：a_ev 子集
    ntab_ev = ntab[te][a_ev]; nfc_ev = nfc[te][a_ev]

    def block(title, key_fn, keys, label_fn):
        print(f"\n=== {title} (7.39x 相关性漂移) ===")
        print(f"{'分组':>8} | {'n':>5} | {'冻结覆盖':>10} | {'适应覆盖':>10} | {'恢复':>8}")
        for k in keys:
            m_all = key_fn(ntab[te], nfc[te]) == k
            m_ev  = key_fn(ntab_ev, nfc_ev) == k
            if m_ev.sum() < 20:
                continue
            cf = conformal.coverage(s_frozen[m_all], q_old)
            ca = conformal.coverage(s_adapt[m_ev], q_new)
            print(f"{label_fn(k):>8} | {m_ev.sum():>5} | {cf:>10.1%} | {ca:>10.1%} | {ca-cf:>+7.1%}")

    block("join 宽度 (表数)", lambda t, f: t, sorted(set(ntab_ev)), lambda k: f"{k}表")
    block("过滤列数", lambda t, f: f, sorted(set(nfc_ev)), lambda k: f"{k}列")

if __name__ == "__main__":
    main()
