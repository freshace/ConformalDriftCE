# -*- coding: utf-8 -*-
"""消融实验（A 校正器有无 / B 特征维度 / C 自适应标签量 / D 树配置）。
在 stats 工作负载、相关性漂移 7.39x 下进行，固定种子 42。
输出到 results_repro/experiment_ablation.txt 并打印。可复现：python drift_ce/experiments/ablation.py
"""
import os, sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, features, model, conformal


def build(extractor, rows, pg, n_ts, n_cs):
    return np.array([extractor(r[0], p, r[3], n_ts, n_cs) for r, p in zip(rows, pg)], dtype=np.float32)


def run_all(feat, tr, cal, te, pg, true, nfc, params, use_corrector, adapt_N, rng):
    """feat=已按切分索引进好的特征矩阵。返回 (frozen, adapt, width_adapt, width_base)。"""
    X_tr, X_cal, X_te = feat[tr], feat[cal], feat[te]
    pe_tr, pe_cal, pe_te = pg[tr], pg[cal], pg[te]
    y_cal, y_tr = true[cal], true[tr]
    # 相关性漂移：只动多列过滤查询（去重过滤列数>=2），乘 exp(2.0)=7.39x
    y_drift = true[te] * np.where(nfc[te] >= 2, np.exp(2.0), 1.0)

    def cov(scores, q):
        return float(np.mean(scores <= q))

    q_old = None
    if use_corrector:
        m_old = model.fit_corrector(X_tr, pe_tr, y_tr, params)
        s_cal = conformal.score(y_cal, pe_cal, model.predict_corrector(m_old, X_cal))
        q_old = conformal.calibrate_q(s_cal, config.ALPHA)
        s_te = conformal.score(y_drift, pe_te, model.predict_corrector(m_old, X_te))
    else:
        s_cal = conformal.score(y_cal, pe_cal, np.zeros(len(pe_cal)))
        q_old = conformal.calibrate_q(s_cal, config.ALPHA)
        s_te = conformal.score(y_drift, pe_te, np.zeros(len(pe_te)))
    frozen = cov(s_te, q_old)
    width_base = float(np.exp(q_old))

    perm = rng.permutation(len(X_te))
    a_tr = perm[:adapt_N]; a_cal = perm[adapt_N:adapt_N+config.N_ADAPT_CAL]; a_ev = perm[adapt_N+config.N_ADAPT_CAL:]
    if use_corrector:
        m_new = model.fit_corrector(X_te[a_tr], pe_te[a_tr], y_drift[a_tr], params)
        s_ae = conformal.score(y_drift[a_ev], pe_te[a_ev], model.predict_corrector(m_new, X_te[a_ev]))
        q_new = conformal.calibrate_q(conformal.score(y_drift[a_cal], pe_te[a_cal], model.predict_corrector(m_new, X_te[a_cal])), config.ALPHA)
    else:
        s_ae = conformal.score(y_drift[a_ev], pe_te[a_ev], np.zeros(len(a_ev)))
        q_new = conformal.calibrate_q(conformal.score(y_drift[a_cal], pe_te[a_cal], np.zeros(len(a_cal))), config.ALPHA)
    adapt = cov(s_ae, q_new)
    return frozen, adapt, float(np.exp(q_new)), width_base


def main():
    n_ts, n_cs = features.load_table_col_space(config.DATA_PATH)
    with open(config.DATA_PATH, "rb") as f:
        data = pickle.load(f)
    rows = []
    for ft, label, pgv, *_ , ex in data["features"]:
        t = float(np.expm1(label-1)); p = float(np.expm1(pgv-1))
        if t <= 0 or p <= 0: continue
        rows.append((ft, ex, len(set(ex.get("filter_col_identifiers", [])))))
    true = np.array([float(np.expm1(l-1)) for ft, l, p, *_ , ex in data["features"]])
    pg = np.array([float(np.expm1(p-1)) for ft, l, p, *_ , ex in data["features"]])
    # 仅保留有效行（t>0 且 p>0）
    mask = np.array([(float(np.expm1(l-1))>0 and float(np.expm1(p-1))>0) for ft, l, p, *_ , ex in data["features"]])
    true = true[mask]; pg = pg[mask]
    idx_r = np.arange(len(rows))[mask]
    rows = [rows[i] for i in idx_r]
    nfc = np.array([r[2] for r in rows], dtype=np.int32)

    F = np.array([extract_full(r[0], p, r[1], n_ts, n_cs) for r, p in zip(rows, pg)], dtype=np.float32)
    G = np.array([extract_s9(r[0], p, r[1], n_ts, n_cs) for r, p in zip(rows, pg)], dtype=np.float32)
    print(f"loaded {len(true)} queries, F dim {F.shape[1]}, G dim {G.shape[1]}", flush=True)

    rng = np.random.RandomState(config.SEED)
    n = len(true)
    idx = rng.permutation(n)
    tr = idx[:config.N_TRAIN]; cal = idx[config.N_TRAIN:config.N_TRAIN+config.N_CAL]; te = idx[config.N_TRAIN+config.N_CAL:config.N_TRAIN+config.N_CAL+config.N_TEST]
    P = config.LGBM_PARAMS

    out = []
    def log(s):
        out.append(s); print(s, flush=True)

    log("=== 消融 A: 校正器有无 (7.39x) ===")
    for uc, label in [(False, "base-only conformal"), (True, "+LGBM corrector")]:
        fr, ad, wq, w0 = run_all(F, tr, cal, te, pg, true, nfc, P, uc, config.N_ADAPT_TRAIN, rng)
        log(f"{label:22s} | frozen {fr:6.1%} | adapt {ad:6.1%} | width_adapt {wq:6.1f}x | width_base {w0:6.1f}x")

    log("=== 消融 B: 特征维度 (7.39x) ===")
    for Xg_, label in [(G, "stats9-only (10-dim)"), (F, "full (622-dim)")]:
        fr, ad, wq, w0 = run_all(Xg_, tr, cal, te, pg, true, nfc, P, True, config.N_ADAPT_TRAIN, rng)
        log(f"{label:26s} | frozen {fr:6.1%} | adapt {ad:6.1%} | width_adapt {wq:6.1f}x")

    log("=== 消融 C: 自适应标签量 (7.39x, full) ===")
    for N in [100, 200, 500, 1000]:
        fr, ad, wq, w0 = run_all(F, tr, cal, te, pg, true, nfc, P, True, N, rng)
        log(f"N={N:5d} | frozen {fr:6.1%} | adapt {ad:6.1%} | width_adapt {wq:6.1f}x")

    log("=== 消融 D: LightGBM 树配置 (7.39x, full) ===")
    for name, prm in [("100 trees,63 leaf", dict(P, n_estimators=100)),
                      ("300 trees,63 leaf", dict(P)),
                      ("300 trees,127 leaf", dict(P, num_leaves=127)),
                      ("1000 trees,63 leaf", dict(P, n_estimators=1000))]:
        fr, ad, wq, w0 = run_all(F, tr, cal, te, pg, true, nfc, prm, True, config.N_ADAPT_TRAIN, rng)
        log(f"{name:20s} | frozen {fr:6.1%} | adapt {ad:6.1%} | width_adapt {wq:6.1f}x")

    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_repro")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ablation.txt"), "w", encoding="utf-8") as f:
        f.write("exit=0\n" + "\n".join(out) + "\n")
    print("\nsaved to results_repro/experiment_ablation.txt", flush=True)


def extract_full(ft, pg, ex, n_ts, n_cs):
    f = np.nan_to_num(np.concatenate([features.build_schema_feature(ft, ex, n_ts, n_cs), features.stats9(ft)]),
                      nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    return np.hstack([f, np.log(np.clip(pg, 1, None))]).astype(np.float32)


def extract_s9(ft, pg, ex, n_ts, n_cs):
    f = np.nan_to_num(features.stats9(ft), nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    return np.hstack([f, np.log(np.clip(pg, 1, None))]).astype(np.float32)


if __name__ == "__main__":
    main()
