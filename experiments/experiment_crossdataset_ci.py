# -*- coding: utf-8 -*-
"""§4.9 跨数据集的重复切分版（P4 收尾）：给 stats/imdb/ergastf1/genome 上三类漂移补 95% CI。

【为什么需要】`experiment_crossdataset.py` 只有一次 `RandomState(config.SEED)`，
论文 §4.9 跨数据集那一组数字全是**单次读数**。

【复现门】三个函数内部各自新建 `RandomState(SEED)`，本脚本改为 `RandomState(SEED+r)`
并保持**完全相同的调用顺序**（choice/permutation 顺序一致），故 r=0 应逐位复现 `cross_dataset.txt`。

【效率改进】原脚本对每个数据集调用 3 次 `load()`（= 3 次特征重建）；本脚本**每数据集只 load 一次**
并缓存特征与 tabsets。

【产出】results_repro/experiment_crossdataset_ci.txt + crossdataset_ci.npz
"""
import os, sys, pickle, warnings
import numpy as np
from collections import Counter
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, features, model, conformal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("DRIFTCE_FEATURES_ROOT", "")
DATASETS = ["stats", "imdb", "ergastf1", "genome"]
R = int(os.environ.get("CD_R", "20"))


def load(ds):
    DATA = os.path.join(BASE, ds, "features40_s3a.pkl")
    n_ts, n_cs = features.load_table_col_space(DATA)
    with open(DATA, "rb") as f:
        data = pickle.load(f)
    bl, yl, sl, nfl, ntl, tsl = [], [], [], [], [], []
    for ft, label, pgv, *_, ex in data["features"]:
        t = float(np.expm1(label - 1)); p = float(np.expm1(pgv - 1))
        if t <= 0 or p <= 0:
            continue
        f = np.nan_to_num(np.concatenate([features.build_schema_feature(ft, ex, n_ts, n_cs),
                                          features.stats9(ft)]),
                          nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        bl.append(p); yl.append(t); sl.append(f)
        nfl.append(len(set(ex.get("filter_col_identifiers", []))))
        ntl.append(len(ex.get("table_indices", [])))
        tsl.append(set(int(x) for x in ex.get("table_indices", [])))
    return (np.array(bl), np.array(yl), np.array(sl, dtype=np.float32),
            np.array(nfl, dtype=np.int32), np.array(ntl, dtype=np.int32), tsl)


def F_of(sf, base):
    return np.hstack([sf, np.log(np.clip(base, 1, None))[:, None]]).astype(np.float32)


def correlation(base, true, sf, nfc, seed):
    rng = np.random.RandomState(seed)
    F = F_of(sf, base); n = len(true); P = config.LGBM_PARAMS
    idx = rng.permutation(n)
    tr = idx[:config.N_TRAIN]
    cal = idx[config.N_TRAIN:config.N_TRAIN + config.N_CAL]
    te = idx[config.N_TRAIN + config.N_CAL:config.N_TRAIN + config.N_CAL + config.N_TEST]
    m_old = model.fit_corrector(F[tr], base[tr], true[tr], P)
    q_old = conformal.calibrate_q(conformal.score(true[cal], base[cal],
                                                  model.predict_corrector(m_old, F[cal])), config.ALPHA)
    b0 = conformal.coverage(conformal.score(true[te], base[te], model.predict_corrector(m_old, F[te])), q_old)
    drift = true[te] * np.where(nfc[te] >= 2, np.exp(2.0), 1.0)
    frozen = conformal.coverage(conformal.score(drift, base[te], model.predict_corrector(m_old, F[te])), q_old)
    perm = rng.permutation(len(te))
    a_tr = perm[:config.N_ADAPT_TRAIN]
    a_cal = perm[config.N_ADAPT_TRAIN:config.N_ADAPT_TRAIN + config.N_ADAPT_CAL]
    a_ev = perm[config.N_ADAPT_TRAIN + config.N_ADAPT_CAL:]
    m_new = model.fit_corrector(F[te][a_tr], base[te][a_tr], drift[a_tr], P)
    q_new = conformal.calibrate_q(conformal.score(drift[a_cal], base[te][a_cal],
                                                  model.predict_corrector(m_new, F[te][a_cal])), config.ALPHA)
    fix = conformal.coverage(conformal.score(drift[a_ev], base[te][a_ev],
                                             model.predict_corrector(m_new, F[te][a_ev])), q_new)
    return b0, frozen, fix


def workload(base, true, sf, ntab, seed):
    rng = np.random.RandomState(seed)
    F = F_of(sf, base); P = config.LGBM_PARAMS
    simple = ntab <= 3; comp = ntab >= 5
    if simple.sum() < 1600 or comp.sum() < 500:
        return None
    tr_pool = np.where(simple)[0]
    tr_sel = rng.choice(tr_pool, size=min(1500, len(tr_pool)), replace=False)
    rest = np.setdiff1d(tr_pool, tr_sel)
    cal_sel = rng.choice(rest, size=min(300, len(rest)), replace=False)
    m_old = model.fit_corrector(F[tr_sel], base[tr_sel], true[tr_sel], P)
    q_old = conformal.calibrate_q(conformal.score(true[cal_sel], base[cal_sel],
                                                  model.predict_corrector(m_old, F[cal_sel])), config.ALPHA)
    te_pool = np.where(comp)[0]
    te = rng.choice(te_pool, size=min(500, len(te_pool)), replace=False)
    drift = conformal.coverage(conformal.score(true[te], base[te],
                                               model.predict_corrector(m_old, F[te])), q_old)
    a_tr, a_cal, a_ev = te[:200], te[200:400], te[400:]
    m_new = model.fit_corrector(F[a_tr], base[a_tr], true[a_tr], P)
    q_new = conformal.calibrate_q(conformal.score(true[a_cal], base[a_cal],
                                                 model.predict_corrector(m_new, F[a_cal])), config.ALPHA)
    fix = conformal.coverage(conformal.score(true[a_ev], base[a_ev],
                                             model.predict_corrector(m_new, F[a_ev])), q_new)
    return drift, fix


def schema(base, true, sf, tabsets, seed):
    rng = np.random.RandomState(seed)
    F = F_of(sf, base); n = len(true); P = config.LGBM_PARAMS
    occ = Counter()
    for ts in tabsets:
        for t in ts:
            occ[t] += 1
    best = None
    for t, c in occ.items():
        if 3000 <= c <= n - 3000:
            if best is None or abs(c - (n - c)) < abs(best[1] - (n - best[1])):
                best = (t, c)
    if best is None:
        return None
    t = best[0]
    with_t = np.array([t in ts for ts in tabsets])
    tr_pool = np.where(~with_t)[0]
    tr_sel = rng.choice(tr_pool, size=min(config.N_TRAIN, len(tr_pool)), replace=False)
    rest = np.setdiff1d(tr_pool, tr_sel)
    cal_sel = rng.choice(rest, size=min(config.N_CAL, len(rest)), replace=False)
    m_old = model.fit_corrector(F[tr_sel], base[tr_sel], true[tr_sel], P)
    q_old = conformal.calibrate_q(conformal.score(true[cal_sel], base[cal_sel],
                                                  model.predict_corrector(m_old, F[cal_sel])), config.ALPHA)
    te_pool = np.where(with_t)[0]
    te = rng.choice(te_pool, size=min(1000, len(te_pool)), replace=False)
    drift = conformal.coverage(conformal.score(true[te], base[te],
                                               model.predict_corrector(m_old, F[te])), q_old)
    a_tr, a_cal, a_ev = te[:200], te[200:400], te[400:]
    m_new = model.fit_corrector(F[a_tr], base[a_tr], true[a_tr], P)
    q_new = conformal.calibrate_q(conformal.score(true[a_cal], base[a_cal],
                                                 model.predict_corrector(m_new, F[a_cal])), config.ALPHA)
    fix = conformal.coverage(conformal.score(true[a_ev], base[a_ev],
                                             model.predict_corrector(m_new, F[a_ev])), q_new)
    return drift, fix


def ci95(a):
    a = np.asarray(a, dtype=np.float64); a = a[np.isfinite(a)]
    if len(a) < 2:
        return float(a.mean()), float("nan")
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def main():
    out = []
    def log(s):
        out.append(s); print(s, flush=True)

    log(f"== §4.9 跨数据集：R={R} 次独立重复划分（4 数据集 × 3 类漂移）==")
    log("口径与 experiment_crossdataset.py 同源；每数据集只 load 一次（原脚本 load 3 次）")
    CACHE = {}
    for ds in DATASETS:
        CACHE[ds] = load(ds)
        log(f"  已加载 {ds}: n={len(CACHE[ds][1])} 维={CACHE[ds][2].shape[1]}")

    acc = {}
    for r in range(R):
        seed = config.SEED + r
        for ds in DATASETS:
            base, true, sf, nfc, ntab, tabsets = CACHE[ds]
            b0, fr, fx = correlation(base, true, sf, nfc, seed)
            acc.setdefault((ds, "corr"), []).append((b0, fr, fx))
            wl = workload(base, true, sf, ntab, seed)
            if wl:
                acc.setdefault((ds, "work"), []).append(wl)
            sc = schema(base, true, sf, tabsets, seed)
            if sc:
                acc.setdefault((ds, "schema"), []).append(sc)
        if (r + 1) % 5 == 0 or r == R - 1:
            log(f"  ... replicate {r+1}/{R}")

    def cell(vals, pct=True):
        a = np.array(vals) * (100.0 if pct else 1.0); m, h = ci95(a)
        return f"{m:6.1f}±{h:4.1f}" if pct else f"{m:7.2f}±{h:5.2f}"

    log("")
    log("--- 相关性漂移（7.39x）---")
    log(f"{'dataset':<10} {'无漂移':>14} {'冻结':>14} {'适应后':>14} {'检测→修复':>14}")
    for ds in DATASETS:
        v = acc.get((ds, "corr"))
        if not v:
            continue
        b0 = [x[0] for x in v]; fr = [x[1] for x in v]; fx = [x[2] for x in v]
        gain = (np.array(fx) - np.array(fr)) * 100
        mg, hg = ci95(gain)
        log(f"{ds:<10} {cell(b0):>14} {cell(fr):>14} {cell(fx):>14} {mg:+6.1f}±{hg:4.1f}pp")
    log("")
    log("--- 工作负载漂移 / schema 漂移 ---")
    log(f"{'dataset':<10} {'work冻结':>12} {'work适应':>12} {'schema冻结':>12} {'schema适应':>12}")
    for ds in DATASETS:
        w = acc.get((ds, "work")); s = acc.get((ds, "schema"))
        fw = cell([x[0] for x in w]) if w else "n/a"
        aw = cell([x[1] for x in w]) if w else "n/a"
        fs = cell([x[0] for x in s]) if s else "n/a"
        as_ = cell([x[1] for x in s]) if s else "n/a"
        log(f"{ds:<10} {fw:>12} {aw:>12} {fs:>12} {as_:>12}")

    log("")
    log("== 复现门：第 1 次重复(seed=42) 应与 cross_dataset.txt 一致 ==")
    for ds in DATASETS:
        v = acc.get((ds, "corr"))
        if v:
            log(f"  {ds:<10} corr 无漂移 {v[0][0]*100:5.1f}% 冻结 {v[0][1]*100:5.1f}% 适应 {v[0][2]*100:5.1f}%")
        w = acc.get((ds, "work")); s = acc.get((ds, "schema"))
        if w:
            log(f"  {'':<10} work {w[0][0]*100:5.1f}% → {w[0][1]*100:5.1f}%")
        if s:
            log(f"  {'':<10} schema {s[0][0]*100:5.1f}% → {s[0][1]*100:5.1f}%")

    OUT = os.path.join(ROOT, "results_repro")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "experiment_crossdataset_ci.txt"), "w", encoding="utf-8") as f:
        f.write("exit=0\n" + "\n".join(out) + "\n")
    np.savez(os.path.join(OUT, "crossdataset_ci.npz"), R=R,
             **{f"{d}__{t}": np.array(v) for (d, t), v in acc.items()})
    print("\nsaved results_repro/experiment_crossdataset_ci.txt", flush=True)


if __name__ == "__main__":
    main()
