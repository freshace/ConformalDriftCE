# -*- coding: utf-8 -*-
"""§4.6 三类漂移（尺度 / 混合 / schema）的重复切分版（P4 收尾）。

【为什么需要】`experiment_scaledrift.py` / `experiment_mixeddrift.py` / `experiment_schemadrift.py`
都只有一次 `RandomState(config.SEED)` 划分，论文 `tab:drift-summary` 里这三行的数字都是**单次读数**。

【复现门】每族都**沿用原脚本完全相同的 RNG 消耗方式**，故第 1 次重复(seed=42)应逐位复现单次读数：
  scale  : RandomState(42).permutation(n)                → 89.4 / 89.0 / 88.0 / 86.7 / 83.2
  mixed  : permutation(n) 然后每个 (scale,corr) 各 permutation(|te|) → 89.4 / 82.6 / 72.3
  schema : choice(tr_pool) → choice(rest) → choice(te_pool)          → 90.0% / 69.0% / 91.5%

【额外补的一件事】`experiment_scaledrift.py` 原本**只测冻结**、没有测适应；论文表里"适应后 83.2%(被吸收)"
其实是**推断**。本脚本把它真的测出来，用数据确认"纯尺度漂移重训也救不动"。

【产出】results_repro/experiment_driftspectrum_ci.txt + driftspectrum_ci.npz
"""
import os, sys, pickle, warnings
import numpy as np
from collections import Counter
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, features, model, conformal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = int(os.environ.get("DS_R", "30"))
SCALE_X = [1.0, 2.0, 5.0, 10.0, 50.0]
MIXED = [(1.0, 1.0), (2.0, 3.0), (3.0, 5.0)]


def load_raw(pkl_path):
    n_ts, n_cs = features.load_table_col_space(pkl_path)
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    rows = []
    for ft, label, pgv, *_, ex in d["features"]:
        true = float(np.expm1(label - 1)); pg = float(np.expm1(pgv - 1))
        if true <= 0 or pg <= 0:
            continue
        rows.append((ft, true, pg, ex, len(set(ex.get("filter_col_identifiers", []))),
                     set(int(t) for t in ex.get("table_indices", []))))
    return rows, n_ts, n_cs


def extract(ft, pg, extras, n_ts, n_cs):
    f = np.nan_to_num(np.concatenate([features.build_schema_feature(ft, extras, n_ts, n_cs),
                                      features.stats9(ft)]),
                      nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    return np.hstack([f, np.log(np.clip(pg, 1, None))]).astype(np.float32)


def scale_ft(ft, X):
    jh, ff, tf, ftf = ft
    tf = np.array(tf, dtype=np.float64).reshape(-1, 4).copy()
    tf[:, 0] += np.log(X)
    return (jh, ff, tf.astype(np.float32), ftf)


def mixed_ft(ft, X, corr, nfc):
    """尺度(改统计) + 相关性(多列真值偏移) 同时发生；只改统计，真值/pg 的乘子在调用处给。"""
    return scale_ft(ft, X)


def ci95(a):
    a = np.asarray(a, dtype=np.float64); a = a[np.isfinite(a)]
    if len(a) < 2:
        return float(a.mean()), float("nan")
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def main():
    out = []
    def log(s):
        out.append(s); print(s, flush=True)

    rows, n_ts, n_cs = load_raw(config.DATA_PATH)
    n = len(rows)
    true = np.array([r[1] for r in rows]); pg = np.array([r[2] for r in rows])
    nfc = np.array([r[4] for r in rows], dtype=np.int32)
    log(f"加载 {n} 条查询；R={R} 次独立重复划分；三族各自沿用原脚本的 RNG 消耗方式")
    F0 = np.array([extract(r[0], r[2], r[3], n_ts, n_cs) for r in rows], dtype=np.float32)

    P = config.LGBM_PARAMS
    res = {}

    # ================= 族 1：尺度（并补测"适应"） =================
    log("")
    log("== 族 1：尺度漂移（重算统计特征；真值/pg 同步乘 X）==")
    Fx = {1.0: F0}
    for X in SCALE_X[1:]:
        log(f"  预计算尺度 X={X} 的特征 ...")
        Fx[X] = np.array([extract(scale_ft(r[0], X), r[2] * X, r[3], n_ts, n_cs) for r in rows],
                         dtype=np.float32)
    acc = {X: {"frozen": [], "adapt": [], "w0": [], "w1": [], "resid": []} for X in SCALE_X}
    for r in range(R):
        rng = np.random.RandomState(config.SEED + r)          # 与原脚本一致
        idx = rng.permutation(n)
        tr = idx[:config.N_TRAIN]; cal = idx[config.N_TRAIN:config.N_TRAIN + config.N_CAL]
        te = idx[config.N_TRAIN + config.N_CAL:config.N_TRAIN + config.N_CAL + config.N_TEST]
        m_old = model.fit_corrector(F0[tr], pg[tr], true[tr], P)
        q_old = conformal.calibrate_q(conformal.score(true[cal], pg[cal],
                                                      model.predict_corrector(m_old, F0[cal])), config.ALPHA)
        for X in SCALE_X:
            F = Fx[X]; t_ = true * X; p_ = pg * X
            acc[X]["frozen"].append(conformal.coverage(
                conformal.score(t_[te], p_[te], model.predict_corrector(m_old, F[te])), q_old))
            acc[X]["w0"].append(float(np.exp(q_old)))
            acc[X]["resid"].append(float(np.mean(np.abs(
                np.log(t_[te] / p_[te]) - np.log(true[te] / pg[te])))))
            perm = rng.permutation(len(te))
            at = te[perm[:config.N_ADAPT_TRAIN]]
            ac = te[perm[config.N_ADAPT_TRAIN:config.N_ADAPT_TRAIN + config.N_ADAPT_CAL]]
            ae = te[perm[config.N_ADAPT_TRAIN + config.N_ADAPT_CAL:]]
            m_new = model.fit_corrector(F[at], p_[at], t_[at], P)
            q_new = conformal.calibrate_q(conformal.score(t_[ac], p_[ac],
                                                          model.predict_corrector(m_new, F[ac])), config.ALPHA)
            acc[X]["adapt"].append(conformal.coverage(
                conformal.score(t_[ae], p_[ae], model.predict_corrector(m_new, F[ae])), q_new))
            acc[X]["w1"].append(float(np.exp(q_new)))
        if (r + 1) % 10 == 0 or r == R - 1:
            log(f"  ... 尺度 {r+1}/{R}")
    res["scale"] = acc
    del Fx

    # ================= 族 2：混合（尺度 × 相关） =================
    log("")
    log("== 族 2：混合漂移（尺度 × 相关性）==")
    Fm = {}
    for X, c in MIXED:
        log(f"  预计算 尺度{X}×相关{c} 的特征 ...")
        Fm[(X, c)] = F0 if X == 1.0 else np.array(
            [extract(mixed_ft(r[0], X, c, r[4]), r[2] * X, r[3], n_ts, n_cs) for r in rows],
            dtype=np.float32)
    dtrue = {}; dpg = {}
    for X, c in MIXED:
        fac = X * np.where(nfc >= 2, c, 1.0)
        dtrue[(X, c)] = true * fac; dpg[(X, c)] = pg * X
    accm = {(X, c): {"frozen": [], "adapt": [], "w0": [], "w1": []} for X, c in MIXED}
    for r in range(R):
        rng = np.random.RandomState(config.SEED + r)
        idx = rng.permutation(n)
        tr = idx[:config.N_TRAIN]; cal = idx[config.N_TRAIN:config.N_TRAIN + config.N_CAL]
        te = idx[config.N_TRAIN + config.N_CAL:config.N_TRAIN + config.N_CAL + config.N_TEST]
        m_old = model.fit_corrector(F0[tr], pg[tr], true[tr], P)
        q_old = conformal.calibrate_q(conformal.score(true[cal], pg[cal],
                                                      model.predict_corrector(m_old, F0[cal])), config.ALPHA)
        for X, c in MIXED:
            F = Fm[(X, c)]; t_ = dtrue[(X, c)]; p_ = dpg[(X, c)]
            accm[(X, c)]["frozen"].append(conformal.coverage(
                conformal.score(t_[te], p_[te], model.predict_corrector(m_old, F[te])), q_old))
            accm[(X, c)]["w0"].append(float(np.exp(q_old)))
            perm = rng.permutation(len(te))          # 与原脚本一致：每个组合各消耗一次
            at = te[perm[:config.N_ADAPT_TRAIN]]
            ac = te[perm[config.N_ADAPT_TRAIN:config.N_ADAPT_TRAIN + config.N_ADAPT_CAL]]
            ae = te[perm[config.N_ADAPT_TRAIN + config.N_ADAPT_CAL:]]
            m_new = model.fit_corrector(F[at], p_[at], t_[at], P)
            q_new = conformal.calibrate_q(conformal.score(t_[ac], p_[ac],
                                                          model.predict_corrector(m_new, F[ac])), config.ALPHA)
            accm[(X, c)]["adapt"].append(conformal.coverage(
                conformal.score(t_[ae], p_[ae], model.predict_corrector(m_new, F[ae])), q_new))
            accm[(X, c)]["w1"].append(float(np.exp(q_new)))
        if (r + 1) % 10 == 0 or r == R - 1:
            log(f"  ... 混合 {r+1}/{R}")
    res["mixed"] = accm
    del Fm, dtrue, dpg

    # ================= 族 3：schema（未见表） =================
    log("")
    log("== 族 3：schema 漂移（训练不含某表 / 评估含该表）==")
    tabsets = [r[5] for r in rows]
    occ = Counter()
    for ts in tabsets:
        for t in ts:
            occ[t] += 1
    best = None
    for t, c in occ.items():
        if 3000 <= c <= n - 3000:
            if best is None or abs(c - (n - c)) < abs(best[1] - (n - best[1])):
                best = (t, c)
    assert best is not None, "无平衡切分的表"
    tb = best[0]
    with_t = np.array([tb in ts for ts in tabsets])
    tr_pool = np.where(~with_t)[0]; te_pool = np.where(with_t)[0]
    log(f"  选表 index={tb}: 含它 {int(with_t.sum())} / 不含 {int((~with_t).sum())} 条")
    accs = {"frozen": [], "adapt": [], "cal": [], "w0": [], "w1": []}
    for r in range(R):
        rng = np.random.RandomState(config.SEED + r)     # 与原脚本一致的三次 choice
        tr_sel = rng.choice(tr_pool, size=min(config.N_TRAIN, len(tr_pool)), replace=False)
        rest = np.setdiff1d(tr_pool, tr_sel)
        cal_sel = rng.choice(rest, size=min(config.N_CAL, len(rest)), replace=False)
        te = rng.choice(te_pool, size=min(1000, len(te_pool)), replace=False)
        m_old = model.fit_corrector(F0[tr_sel], pg[tr_sel], true[tr_sel], P)
        q_old = conformal.calibrate_q(conformal.score(true[cal_sel], pg[cal_sel],
                                                      model.predict_corrector(m_old, F0[cal_sel])), config.ALPHA)
        accs["cal"].append(conformal.coverage(
            conformal.score(true[cal_sel], pg[cal_sel], model.predict_corrector(m_old, F0[cal_sel])), q_old))
        accs["frozen"].append(conformal.coverage(
            conformal.score(true[te], pg[te], model.predict_corrector(m_old, F0[te])), q_old))
        accs["w0"].append(float(np.exp(q_old)))
        a_tr, a_cal, a_ev = te[:200], te[200:400], te[400:]
        m_new = model.fit_corrector(F0[a_tr], pg[a_tr], true[a_tr], P)
        q_new = conformal.calibrate_q(conformal.score(true[a_cal], pg[a_cal],
                                                      model.predict_corrector(m_new, F0[a_cal])), config.ALPHA)
        accs["adapt"].append(conformal.coverage(
            conformal.score(true[a_ev], pg[a_ev], model.predict_corrector(m_new, F0[a_ev])), q_new))
        accs["w1"].append(float(np.exp(q_new)))
        if (r + 1) % 10 == 0 or r == R - 1:
            log(f"  ... schema {r+1}/{R}")
    res["schema"] = accs

    # ================= 汇总 =================
    log("")
    log(f"== 三类漂移（R={R} 次独立划分；mean ± 95%CI，方括号为单次读数范围）==")
    log("")
    log("--- 尺度 ---")
    log(f"{'X':>6} {'冻结覆盖':>22} {'适应后覆盖':>22} {'冻结宽':>14} {'适应宽':>14} {'残差变化':>10}")
    for X in SCALE_X:
        d = acc[X]
        def c(k, pct=True):
            a = np.array(d[k]) * (100.0 if pct else 1.0); m, h = ci95(a)
            return f"{m:6.1f}±{h:4.1f} [{a.min():.1f},{a.max():.1f}]" if pct else f"{m:7.2f}±{h:5.2f}"
        log(f"{X:>5.0f}x {c('frozen'):>22} {c('adapt'):>22} {c('w0',False):>14} {c('w1',False):>14} "
            f"{np.mean(d['resid']):>10.4f}")
    log("")
    log("--- 混合（尺度×相关）---")
    log(f"{'配置':>14} {'冻结覆盖':>22} {'适应后覆盖':>22} {'冻结宽':>14} {'适应宽':>14}")
    for X, cc in MIXED:
        d = accm[(X, cc)]
        def c2(k, pct=True):
            a = np.array(d[k]) * (100.0 if pct else 1.0); m, h = ci95(a)
            return f"{m:6.1f}±{h:4.1f} [{a.min():.1f},{a.max():.1f}]" if pct else f"{m:7.2f}±{h:5.2f}"
        log(f"{'%.0fx×%.0fx' % (X, cc):>14} {c2('frozen'):>22} {c2('adapt'):>22} "
            f"{c2('w0',False):>14} {c2('w1',False):>14}")
    log("")
    log("--- schema（未见表）---")
    for k, lab in (("cal", "校准集(不含该表)"), ("frozen", "含新表(冻结)"), ("adapt", "适应后")):
        a = np.array(accs[k]) * 100; m, h = ci95(a)
        log(f"  {lab:<20} {m:6.1f}% ± {h:4.1f}%  [{a.min():.1f}, {a.max():.1f}]")
    a0, h0 = ci95(accs["w0"]); a1, h1 = ci95(accs["w1"])
    log(f"  区间宽度              冻结 {a0:7.2f}±{h0:5.2f}x → 适应 {a1:7.2f}±{h1:5.2f}x")

    log("")
    log("== 复现门：第 1 次重复(seed=42) 应与各单次读数一致 ==")
    log("  尺度:  " + " / ".join(f"{acc[X]['frozen'][0]*100:.1f}%" for X in SCALE_X)
        + "   (原: 89.4/89.0/88.0/86.7/83.2)")
    log("  混合:  " + " / ".join(f"{accm[(X, c)]['frozen'][0]*100:.1f}%" for X, c in MIXED)
        + "   (原: 89.4/82.6/72.3)")
    log(f"  schema: 冻结 {accs['frozen'][0]*100:.1f}% | 适应 {accs['adapt'][0]*100:.1f}%"
        f"   (原: 69.0% / 91.5%)")

    OUT = os.path.join(ROOT, "results_repro")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "experiment_driftspectrum_ci.txt"), "w", encoding="utf-8") as f:
        f.write("exit=0\n" + "\n".join(out) + "\n")
    np.savez(os.path.join(OUT, "driftspectrum_ci.npz"), R=R,
             scale_X=np.array(SCALE_X),
             scale_frozen=np.array([acc[X]["frozen"] for X in SCALE_X]),
             scale_adapt=np.array([acc[X]["adapt"] for X in SCALE_X]),
             mixed_cfg=np.array(MIXED),
             mixed_frozen=np.array([accm[k]["frozen"] for k in MIXED]),
             mixed_adapt=np.array([accm[k]["adapt"] for k in MIXED]),
             schema_frozen=np.array(accs["frozen"]), schema_adapt=np.array(accs["adapt"]),
             schema_cal=np.array(accs["cal"]))
    print("\nsaved results_repro/experiment_driftspectrum_ci.txt", flush=True)


if __name__ == "__main__":
    main()
