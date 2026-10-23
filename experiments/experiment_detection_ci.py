# -*- coding: utf-8 -*-
"""检测器对比（诚实版 + 重复划分 CI）—— 修正 experiment_detection.py 的结构性缺陷。

【为什么必须重做】原 experiment_detection.py 的结果表（results_repro/experiment_detection.txt）
在方法学上是同义反复，不能作为"特征检测器无效"的证据：
  (1) 阈值取自"无漂移窗口"的 90 分位，而这些窗口**又是**被当作负例评估的同一批窗口
      -> df=1.0 行的 FPR **恒等于 10.0%**（构造出来的，不是测出来的）。
  (2) df!=1 时 `labels = np.ones(...)` 全为 1，没有负窗口
      -> 这些行的 FPR **结构性为 0.0%**（分母为空）。
  (3) 漂移只写进 `y_drift`，特征 `f_te` **一个字节都没动**
      -> 特征检测器分数在所有 df 下**逐位相同**，恰好复现无漂移时的 10.0%
      -> "feat_TPR ≡ 10%" 是**数学必然**，不是测量结果。

【本脚本怎么做】
  (a) 三个窗口集互不重叠：阈值只在**标定负窗口**上定，FPR 在**独立留出负窗口**上测，TPR 在**正窗口**上测。
      -> FPR 成为真正的测量量（而不是"90 分位"这个定义的复述）。
  (b) 两个漂移族的对照，构成"检测反转"：
      corr  : 相关性漂移（边际不变）-> 特征**逐位不变**（数学必然：PG 统计/特征空间是边际型）
              但对估计**有害**（覆盖崩塌）
      scale : 尺度漂移（改统计、重算特征）-> 特征**真的会动**，但残差不变、**基本无害**（被吸收）
  (c) **给无标签基线最强版本**：原脚本用的是"全 622 维 L2 距离"，被结构性 one-hot 块的采样噪声淹没，
      是稻草人。这里同时报两个特征监控器：
        featL2 : 朴素全维 L2（= 原脚本口径，作为对照）
        featZ  : 逐维标准化均值漂移 max_j |mean_win(F_j)-mean_tr(F_j)| / (std_tr(F_j)/sqrt(n))
                 -> 这是两样本均值漂移的标准统计量，是无标签监控器的**最强合理版本**
  (d) R 次独立重复（每次重抽 训练/校准划分 与 窗口角色分配）-> mean ± 95% CI。

【产出】results_repro/experiment_detection_ci.txt  +  detection_ci.npz
"""
import os, sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, features, model, conformal

N_WINDOW = 100          # 每窗口查询数
# 默认 R=50（论文口径）。冒烟测试可用环境变量 DET_R / DET_POOL 临时缩小，默认行为不变 -> 可复现。
R = int(os.environ.get("DET_R", "50"))
N_POOL = int(os.environ.get("DET_POOL", "20000"))   # 固定候选池（窗口从这里抽）
CORR_FACTORS = [1.0, 1.35, 2.0, 2.72, 4.48, 7.39]
SCALE_FACTORS = [1.0, 2.0, 5.0, 10.0, 50.0]
FRAC = 0.90             # 目标 FPR = 10%
STD_FLOOR = 1e-3        # 逐维标准差的数值下限，避免近常量维度放大 z


def load_raw(pkl_path):
    """per-query: (ft, true, pg, extras)。保留原始 ft，便于按尺度漂移改统计再重算特征。"""
    n_ts, n_cs = features.load_table_col_space(pkl_path)
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    rows = []
    for ft, label, pgv, *_, ex in d["features"]:
        true = float(np.expm1(label - 1)); pg = float(np.expm1(pgv - 1))
        if true <= 0 or pg <= 0:
            continue
        rows.append((ft, true, pg, ex))
    return rows, n_ts, n_cs


def extract(ft, extras, pg, n_ts, n_cs):
    f = np.nan_to_num(np.concatenate([features.build_schema_feature(ft, extras, n_ts, n_cs),
                                      features.stats9(ft)]),
                      nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    return np.hstack([f, np.log(np.clip(pg, 1, None))]).astype(np.float32)


def scale_drift(ft, X):
    """尺度漂移：表大小 tf[:,0] += log X（真值/pg 同步乘 X，故残差不变）。"""
    jh, ff, tf, ftf = ft
    tf = np.array(tf, dtype=np.float64).reshape(-1, 4).copy()
    tf[:, 0] += np.log(X)
    return (jh, ff, tf.astype(np.float32), ftf)


def ci95(a):
    a = np.asarray(a, dtype=np.float64)
    if len(a) < 2:
        return float(a.mean()), float("nan")
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


DETS = ("conf", "resid", "featL2", "featZ")


def main():
    out = []
    def log(s):
        out.append(s); print(s, flush=True)

    rows, n_ts, n_cs = load_raw(config.DATA_PATH)
    n_all = len(rows)
    assert n_all >= 30000 + N_POOL, f"查询数不足: {n_all}"

    N_WIN = N_POOL // N_WINDOW
    n_cal = max(1, int(0.30 * N_WIN))
    n_neg = max(1, int(0.30 * N_WIN))
    n_pos = N_WIN - n_cal - n_neg
    assert n_pos >= 1, f"窗口数太少: N_WIN={N_WIN}"

    log(f"加载 {n_all} 条查询, 特征空间 ({n_ts}, {n_cs})")
    log(f"候选池 {N_POOL} 条 -> {N_WIN} 窗口 x {N_WINDOW} 条"
        f"（标定负 {n_cal} / 测试负 {n_neg} / 正 {n_pos}）；R={R} 次重复，目标 FPR={1-FRAC:.0%}")

    true_all = np.array([r[1] for r in rows], dtype=np.float64)
    pg_all = np.array([r[2] for r in rows], dtype=np.float64)
    nfc_all = np.array([len(set(r[3].get("filter_col_identifiers", []))) for r in rows], dtype=np.int32)

    POOL = np.arange(n_all - N_POOL, n_all)
    SRC = np.arange(0, n_all - N_POOL)

    log("构建基准特征矩阵（训练池 + 候选池）...")
    F_src = np.array([extract(rows[i][0], rows[i][3], rows[i][2], n_ts, n_cs) for i in SRC],
                     dtype=np.float32)
    F0_pool = np.array([extract(rows[i][0], rows[i][3], rows[i][2], n_ts, n_cs) for i in POOL],
                       dtype=np.float32)
    Fx_pool = {1.0: F0_pool}
    for X in SCALE_FACTORS:
        if X == 1.0:
            continue
        log(f"  重算尺度漂移 X={X} 的特征 ...")
        Fx_pool[X] = np.array(
            [extract(scale_drift(rows[i][0], X), rows[i][3], rows[i][2] * X, n_ts, n_cs)
             for i in POOL], dtype=np.float32)

    # ---- 特征/残差位移量（检测反转的横轴）----
    log("")
    log("== 特征位移与残差位移（漂移后 vs 基准，候选池）==")
    log(f"{'family':>6} {'drift':>8} | {'max|dF|':>11} | {'变化维度占比':>12} | {'平均|d残差|':>12}")
    log("-" * 60)
    feat_shift, resid_shift = {}, {}
    for fam, factors in (("corr", CORR_FACTORS), ("scale", SCALE_FACTORS)):
        for df in factors:
            if fam == "corr":
                t_d = true_all[POOL] * np.where(nfc_all[POOL] >= 2, df, 1.0)
                F_d, pg_d = F0_pool, pg_all[POOL]
            else:
                t_d = true_all[POOL] * df
                F_d, pg_d = Fx_pool[df], pg_all[POOL] * df
            d = np.abs(F_d - F0_pool)
            feat_shift[(fam, df)] = float(d.max())
            frac_changed = float((d > 0).any(axis=0).mean())
            r0 = np.log(true_all[POOL]) - np.log(np.clip(pg_all[POOL], 1, None))
            r1 = np.log(t_d) - np.log(np.clip(pg_d, 1, None))
            rs = float(np.mean(np.abs(r1 - r0)))
            resid_shift[(fam, df)] = rs
            log(f"{fam:>6} {df:>7.2f}x | {d.max():>11.3e} | {frac_changed:>11.1%} | {rs:>12.4f}")

    # ---- 重复实验 ----
    recs = []
    for r in range(R):
        rs = np.random.RandomState(config.SEED + r)
        perm = rs.permutation(len(SRC))
        tr = SRC[perm[:config.N_TRAIN]]
        cal = SRC[perm[config.N_TRAIN:config.N_TRAIN + config.N_CAL]]
        F_tr = F_src[perm[:config.N_TRAIN]]
        F_cal = F_src[perm[config.N_TRAIN:config.N_TRAIN + config.N_CAL]]
        m = model.fit_corrector(F_tr, pg_all[tr], true_all[tr], config.LGBM_PARAMS)
        q90 = conformal.calibrate_q(
            conformal.score(true_all[cal], pg_all[cal], model.predict_corrector(m, F_cal)), config.ALPHA)
        f_mean = F_tr.mean(axis=0)
        f_std = np.maximum(F_tr.std(axis=0), STD_FLOOR) / np.sqrt(N_WINDOW)

        wr = np.random.RandomState(config.SEED + 5000 + r)
        wperm = wr.permutation(N_WIN)
        w_cal, w_neg, w_pos = wperm[:n_cal], wperm[n_cal:n_cal + n_neg], wperm[n_cal + n_neg:]

        corr_pool = model.predict_corrector(m, F0_pool)
        cache = {"corr": {1.0: corr_pool}, "scale": {1.0: corr_pool}}

        def _corr_for(fam, df):
            if df not in cache[fam]:
                cache[fam][df] = model.predict_corrector(m, F0_pool if fam == "corr" else Fx_pool[df])
            return cache[fam][df]

        sens_all = nfc_all[POOL] >= 2

        def scores(fam, df, wins):
            """窗口级 (覆盖缺口, 残差均值, 朴素L2, 标准化z, 敏感子集覆盖率)。"""
            if fam == "corr":
                F_d, pg_d = F0_pool, pg_all[POOL]
                t_d = true_all[POOL] * np.where(nfc_all[POOL] >= 2, df, 1.0)
            else:
                F_d, pg_d = Fx_pool[df], pg_all[POOL] * df
                t_d = true_all[POOL] * df
            corr_d = _corr_for(fam, df)
            cg, rm, l2, zz, sg = [], [], [], [], []
            for w in wins:
                sl = slice(w * N_WINDOW, (w + 1) * N_WINDOW)
                sc = conformal.score(t_d[sl], pg_d[sl], corr_d[sl])
                cg.append(1.0 - conformal.coverage(sc, q90))
                sg.append(conformal.coverage(sc[sens_all[sl]], q90) if sens_all[sl].sum() > 0
                          else np.nan)
                err = np.abs(np.log(t_d[sl]) - np.log(np.clip(pg_d[sl] * np.exp(corr_d[sl]), 1, None)))
                rm.append(float(err.mean()))
                wm = F_d[sl].mean(axis=0)
                l2.append(float(np.linalg.norm(wm - f_mean)))
                zz.append(float(np.max(np.abs(wm - f_mean) / f_std)))
            return (np.array(cg), np.array(rm), np.array(l2), np.array(zz), np.array(sg))

        for fam, factors in (("corr", CORR_FACTORS), ("scale", SCALE_FACTORS)):
            c0, r0s, l0, z0, _ = scores(fam, 1.0, w_cal)
            thr = [np.percentile(c0, FRAC * 100), np.percentile(r0s, FRAC * 100),
                   np.percentile(l0, FRAC * 100), np.percentile(z0, FRAC * 100)]
            cN, rN, lN, zN, _ = scores(fam, 1.0, w_neg)
            fpr = [(cN > thr[0]).mean(), (rN > thr[1]).mean(),
                   (lN > thr[2]).mean(), (zN > thr[3]).mean()]
            for df in factors:
                c, rr, ll, zzv, ss = scores(fam, df, w_pos)
                vals = (c, rr, ll, zzv)
                tpr = [(vals[k] > thr[k]).mean() for k in range(4)]
                cov_pos = float(1.0 - c.mean())
                cov_sens = float(np.nanmean(ss)) if np.isfinite(ss).any() else float("nan")
                for k, det in enumerate(DETS):
                    recs.append((fam, df, det, float(tpr[k]), float(fpr[k]), cov_pos, cov_sens))
        if (r + 1) % 10 == 0 or r == R - 1:
            log(f"  ... replicate {r+1}/{R}")

    # ---- 汇总 ----
    def cell(fam, df, det):
        sub = [x for x in recs if x[0] == fam and x[1] == df and x[2] == det]
        return ci95([x[3] for x in sub]), ci95([x[4] for x in sub])

    def fmt(t):
        return f"{t[0]*100:5.1f}±{t[1]*100:4.1f}"

    log("")
    log(f"== 表 A：有标签检测器（R={R} 次独立重复；mean ± 95%CI）==")
    log("   阈值只在标定负窗口上定；FPR 在独立留出负窗口上测；TPR 在正窗口上测（三者不重叠）")
    log("")
    h = (f"{'family':>6} {'drift':>8} {'整体覆盖':>9} {'敏感子集':>9} | "
         f"{'conf TPR':>16} {'conf FPR':>15} | {'resid TPR':>16} {'resid FPR':>15}")
    log(h); log("-" * len(h))
    for fam, factors in (("corr", CORR_FACTORS), ("scale", SCALE_FACTORS)):
        for df in factors:
            covp = float(np.mean([x[5] for x in recs if x[0] == fam and x[1] == df]))
            covs = float(np.nanmean([x[6] for x in recs if x[0] == fam and x[1] == df]))
            cc, cf = cell(fam, df, "conf"); rc, rf = cell(fam, df, "resid")
            log(f"{fam:>6} {df:>7.2f}x {covp*100:>8.1f}% {covs*100:>8.1f}% | "
                f"{fmt(cc):>16} {fmt(cf):>15} | {fmt(rc):>16} {fmt(rf):>15}")

    log("")
    log("== 表 B：无标签特征监控器（同一阈值口径；featZ 是给它的最强版本）==")
    log("")
    h2 = (f"{'family':>6} {'drift':>8} {'特征max|dF|':>12} {'变化维度':>9} | "
          f"{'featL2 TPR':>16} {'featL2 FPR':>16} | {'featZ TPR':>16} {'featZ FPR':>15}")
    log(h2); log("-" * len(h2))
    for fam, factors in (("corr", CORR_FACTORS), ("scale", SCALE_FACTORS)):
        for df in factors:
            d = np.abs((F0_pool if fam == "corr" else Fx_pool[df]) - F0_pool)
            frac_changed = float((d > 0).any(axis=0).mean())
            lc, lf = cell(fam, df, "featL2"); zc, zf = cell(fam, df, "featZ")
            log(f"{fam:>6} {df:>7.2f}x {feat_shift[(fam, df)]:>12.3e} {frac_changed*100:>8.1f}% | "
                f"{fmt(lc):>16} {fmt(lf):>16} | {fmt(zc):>16} {fmt(zf):>15}")

    # ---- 检测反转检验 ----
    log("")
    log("== 检测反转检验：'特征位移量' vs '真实危害（正窗口整体覆盖损失）' ==")
    xs, ys = [], []
    for fam, factors in (("corr", CORR_FACTORS), ("scale", SCALE_FACTORS)):
        for df in factors:
            covp = float(np.mean([x[5] for x in recs if x[0] == fam and x[1] == df]))
            xs.append(feat_shift[(fam, df)]); ys.append(1.0 - covp)
    xs, ys = np.array(xs), np.array(ys)
    ok = xs.std() > 0 and ys.std() > 0
    pear = float(np.corrcoef(xs, ys)[0, 1]) if ok else float("nan")
    rk = lambda a: np.argsort(np.argsort(a)).astype(np.float64)
    spear = float(np.corrcoef(rk(xs), rk(ys))[0, 1]) if ok else float("nan")
    log(f"  特征位移 max|dF| 与 覆盖损失 的 Pearson r = {pear:+.3f} ; Spearman rho = {spear:+.3f}")
    log("  解读：r<=0 意味着'无标签的特征监控器'报警强度与真实危害**方向相反/无关**——")
    log("        它在**无害**的尺度漂移上强烈报警，在**有害**的相关性漂移上完全失明。")

    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_repro")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "experiment_detection_ci.txt"), "w", encoding="utf-8") as f:
        f.write("exit=0\n" + "\n".join(out) + "\n")
    fam_i = {"corr": 0, "scale": 1}; det_i = {d: i for i, d in enumerate(DETS)}
    arr = np.array([[fam_i[x[0]], x[1], det_i[x[2]], x[3], x[4], x[5], x[6]] for x in recs],
                   dtype=np.float64)
    np.savez(os.path.join(OUT, "detection_ci.npz"), recs=arr, pearson=pear, spearman=spear,
             dets=np.array(DETS))
    print("\nsaved to results_repro/experiment_detection_ci.txt", flush=True)


if __name__ == "__main__":
    main()
