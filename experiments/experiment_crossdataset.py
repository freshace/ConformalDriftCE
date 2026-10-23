# -*- coding: utf-8 -*-
"""跨数据集实验：相关性 / 工作负载 / schema 三类漂移在 stats/imdb/ergastf1/genome 上复现。
固定种子 42。输出到 results_repro/cross_dataset.txt 并打印。
可复现：python drift_ce/experiments/crossdataset.py
"""
import os, sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, features, model, conformal

BASE = os.environ.get("DRIFTCE_FEATURES_ROOT", "")


def load(ds):
    DATA = os.path.join(BASE, ds, "features40_s3a.pkl")
    n_ts, n_cs = features.load_table_col_space(DATA)
    with open(DATA, "rb") as f:
        data = pickle.load(f)
    bl, yl, sl, nfl, ntl = [], [], [], [], []
    for ft, label, pgv, *_ , ex in data["features"]:
        t = float(np.expm1(label-1)); p = float(np.expm1(pgv-1))
        if t <= 0 or p <= 0: continue
        f = np.nan_to_num(np.concatenate([features.build_schema_feature(ft, ex, n_ts, n_cs), features.stats9(ft)]),
                          nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        bl.append(p); yl.append(t); sl.append(f)
        nfl.append(len(set(ex.get("filter_col_identifiers", []))))
        ntl.append(len(ex.get("table_indices", [])))
    return (np.array(bl), np.array(yl), np.array(sl, dtype=np.float32),
            np.array(nfl, dtype=np.int32), np.array(ntl, dtype=np.int32))


def F_of(sf, base):
    return np.hstack([sf, np.log(np.clip(base, 1, None))[:, None]]).astype(np.float32)


def correlation(base, true, sf, nfc):
    rng = np.random.RandomState(config.SEED)
    F = F_of(sf, base)
    n = len(true)
    idx = rng.permutation(n)
    tr = idx[:config.N_TRAIN]; cal = idx[config.N_TRAIN:config.N_TRAIN+config.N_CAL]; te = idx[config.N_TRAIN+config.N_CAL:config.N_TRAIN+config.N_CAL+config.N_TEST]
    P = config.LGBM_PARAMS
    m_old = model.fit_corrector(F[tr], base[tr], true[tr], P)
    q_old = conformal.calibrate_q(conformal.score(true[cal], base[cal], model.predict_corrector(m_old, F[cal])), config.ALPHA)
    b0 = conformal.coverage(conformal.score(true[te], base[te], model.predict_corrector(m_old, F[te])), q_old)
    drift = true[te] * np.where(nfc[te] >= 2, np.exp(2.0), 1.0)
    frozen = conformal.coverage(conformal.score(drift, base[te], model.predict_corrector(m_old, F[te])), q_old)
    perm = rng.permutation(len(te))
    a_tr = perm[:config.N_ADAPT_TRAIN]; a_cal = perm[config.N_ADAPT_TRAIN:config.N_ADAPT_TRAIN+config.N_ADAPT_CAL]; a_ev = perm[config.N_ADAPT_TRAIN+config.N_ADAPT_CAL:]
    m_new = model.fit_corrector(F[te][a_tr], base[te][a_tr], drift[a_tr], P)
    q_new = conformal.calibrate_q(conformal.score(drift[a_cal], base[te][a_cal], model.predict_corrector(m_new, F[te][a_cal])), config.ALPHA)
    fix = conformal.coverage(conformal.score(drift[a_ev], base[te][a_ev], model.predict_corrector(m_new, F[te][a_ev])), q_new)
    return b0, frozen, fix


def workload(base, true, sf, ntab):
    rng = np.random.RandomState(config.SEED)
    F = F_of(sf, base)
    simple = ntab <= 3; comp = ntab >= 5
    if simple.sum() < 1600 or comp.sum() < 500: return None
    tr_pool = np.where(simple)[0]; tr_sel = rng.choice(tr_pool, size=min(1500, len(tr_pool)), replace=False)
    rest = np.setdiff1d(tr_pool, tr_sel); cal_sel = rng.choice(rest, size=min(300, len(rest)), replace=False)
    P = config.LGBM_PARAMS
    m_old = model.fit_corrector(F[tr_sel], base[tr_sel], true[tr_sel], P)
    q_old = conformal.calibrate_q(conformal.score(true[cal_sel], base[cal_sel], model.predict_corrector(m_old, F[cal_sel])), config.ALPHA)
    te_pool = np.where(comp)[0]; te = rng.choice(te_pool, size=min(500, len(te_pool)), replace=False)
    drift = conformal.coverage(conformal.score(true[te], base[te], model.predict_corrector(m_old, F[te])), q_old)
    a_tr = te[:200]; a_cal = te[200:400]; a_ev = te[400:]
    m_new = model.fit_corrector(F[a_tr], base[a_tr], true[a_tr], P)
    q_new = conformal.calibrate_q(conformal.score(true[a_cal], base[a_cal], model.predict_corrector(m_new, F[a_cal])), config.ALPHA)
    fix = conformal.coverage(conformal.score(true[a_ev], base[a_ev], model.predict_corrector(m_new, F[a_ev])), q_new)
    return drift, fix


def schema(base, true, sf, tabsets):
    rng = np.random.RandomState(config.SEED)
    F = F_of(sf, base)
    n = len(true)
    from collections import Counter
    occ = Counter()
    for ts in tabsets:
        for t in ts: occ[t] += 1
    best = None
    for t, c in occ.items():
        no = n - c
        if 3000 <= c <= n - 3000:
            if best is None or abs(c - no) < abs(best[1] - (n - best[1])): best = (t, c)
    if best is None: return None
    t = best[0]; with_t = np.array([t in ts for ts in tabsets])
    tr_pool = np.where(~with_t)[0]; tr_sel = rng.choice(tr_pool, size=min(config.N_TRAIN, len(tr_pool)), replace=False)
    rest = np.setdiff1d(tr_pool, tr_sel); cal_sel = rng.choice(rest, size=min(config.N_CAL, len(rest)), replace=False)
    P = config.LGBM_PARAMS
    m_old = model.fit_corrector(F[tr_sel], base[tr_sel], true[tr_sel], P)
    q_old = conformal.calibrate_q(conformal.score(true[cal_sel], base[cal_sel], model.predict_corrector(m_old, F[cal_sel])), config.ALPHA)
    te_pool = np.where(with_t)[0]; te = rng.choice(te_pool, size=min(1000, len(te_pool)), replace=False)
    drift = conformal.coverage(conformal.score(true[te], base[te], model.predict_corrector(m_old, F[te])), q_old)
    a_tr = te[:200]; a_cal = te[200:400]; a_ev = te[400:]
    m_new = model.fit_corrector(F[a_tr], base[a_tr], true[a_tr], P)
    q_new = conformal.calibrate_q(conformal.score(true[a_cal], base[a_cal], model.predict_corrector(m_new, F[a_cal])), config.ALPHA)
    fix = conformal.coverage(conformal.score(true[a_ev], base[a_ev], model.predict_corrector(m_new, F[a_ev])), q_new)
    return drift, fix


def main():
    out = []
    def log(s):
        out.append(s); print(s, flush=True)
    log("=== CROSS-DATASET: canonical correlation drift (7.39x) ===")
    log(f"{'dataset':10s} | {'n':>6s} {'ts':>3s} {'cs':>3s} | {'no-drift':>8s} {'frozen':>8s} {'adapt500':>8s} | {'det-fix':>8s}")
    corr = {}
    for ds in ["stats", "imdb", "ergastf1", "genome"]:
        base, true, sf, nfc, ntab = load(ds)
        n_ts, n_cs = features.load_table_col_space(os.path.join(BASE, ds, "features40_s3a.pkl"))
        b0, fr, fx = correlation(base, true, sf, nfc)
        corr[ds] = (b0, fr, fx)
        log(f"{ds:10s} | {len(true):6d} {n_ts:3d} {n_cs:3d} | {b0:8.1%} {fr:8.1%} {fx:8.1%} | {fx-fr:+8.1%}")

    log("")
    log("=== CROSS-DATASET: workload drift (frozen->adapt) ===")
    for ds in ["stats", "imdb", "ergastf1", "genome"]:
        base, true, sf, nfc, ntab = load(ds)
        wl = workload(base, true, sf, ntab)
        log(f"{ds:9s} | workload {wl[0]:.1%} -> {wl[1]:.1%}" if wl else f"{ds:9s} | workload n/a")

    log("")
    log("=== CROSS-DATASET: schema drift (unseen table, frozen->adapt) ===")
    for ds in ["stats", "imdb", "ergastf1", "genome"]:
        base, true, sf, nfc, ntab = load(ds)
        n_ts, n_cs = features.load_table_col_space(os.path.join(BASE, ds, "features40_s3a.pkl"))
        with open(os.path.join(BASE, ds, "features40_s3a.pkl"), "rb") as f:
            data = pickle.load(f)
        tabsets = [set(int(t) for t in ex.get("table_indices", [])) for ft, label, pgv, *_ , ex in data["features"]]
        sc = schema(base, true, sf, tabsets)
        log(f"{ds:9s} | schema {sc[0]:.1%} -> {sc[1]:.1%}" if sc else f"{ds:9s} | schema n/a")

    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_repro")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "cross_dataset.txt"), "w", encoding="utf-8") as f:
        f.write("exit=0\n" + "\n".join(out) + "\n")
    print("\nsaved to results_repro/cross_dataset.txt", flush=True)


if __name__ == "__main__":
    main()
