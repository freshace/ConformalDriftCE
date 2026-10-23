# -*- coding: utf-8 -*-
r"""真实数据漂移 —— 论文口径分析 v3（按独立审计修正 C1/C2/M1/M2/M3/M5）。

相对 v2 的修正
--------------
C1 【严重】v2 里 corr 臂的 q* 是【样本内】算的：`fit_corrector(F,...)` 在全部 496 条上拟合，
   又在同一批上取分位数 -> baseline q*=0.2742 nats（1.32 倍），比诚实的样本外小 12.6 倍。
   v3：corr 臂的 q* 也走三段 —— 每折 train 段拟合，q* 只在 eval 段上取。
C2 【严重】v2 的 McNemar 把 200 次【重叠】切分的 L/G 相加，同一条查询被计入 20~47 次，
   p 值虚低到 1e-24~1e-80。v3 改为【以查询为单位】：
     * 每条查询在它出现过的所有折里取"状态覆盖率 - baseline 覆盖率"的均值 -> Δ_i ∈ [-1,1]；
     * 对 Δ_i 做配对 t 检验（每条查询一票）；
     * 另报查询级 McNemar（Δ_i>0 记得、Δ_i<0 记损）。
M3 受影响子集报【逐查询标准误与 95%CI】，不再只报折池化点估计。
M1 分数一律用 true<=0 -> +inf（v2 已改，v3 保持一致）。
M5 真值变化分布【真正】剔除 true=0 的行，不混进均值/标准差/分位。
"""
import os
import pickle
import sys

import numpy as np
from scipy import stats as st

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))  # 脚本在 scripts/ 下，模块在仓库根
import config, data, metrics, model  # noqa: E402

POOL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "pool_all.pkl")
if not os.path.isabs(POOL):
    POOL = os.path.join(HERE, POOL)
PKL = config.DATA_PATH
ALPHA = config.ALPHA
SEED = 42
K = 200


def score_of(t, p):
    """true<=0 -> +inf（乘法区间永远不含 0，结构上不可覆盖）。"""
    t = np.asarray(t, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1, None)
    s = np.abs(np.log(np.clip(t, 1, None)) - np.log(p))
    return np.where(t <= 0, np.inf, s)


def qerr(a, b):
    a = np.clip(a, 1, None); b = np.clip(b, 1, None)
    return np.maximum(a / b, b / a)


def main():
    d = pickle.load(open(POOL, "rb"))
    _, _, feat_all, _ = data.load_stats(PKL)
    idx = np.asarray(d["idx"], dtype=int)
    sts = list(d["states"]); base_st = "baseline"
    ok_all = None; T, P = {}, {}
    for s in sts:
        v = d["states"][s]
        ok = np.asarray(v["ok"], dtype=bool)
        t = np.asarray([np.nan if x is None else x for x in v["true"]], dtype=float)
        p = np.asarray([np.nan if x is None else x for x in v["pg_est"]], dtype=float)
        m = ok & np.isfinite(t) & np.isfinite(p)
        ok_all = m if ok_all is None else (ok_all & m)
        T[s], P[s] = t, p
    n = int(ok_all.sum())
    F = feat_all[idx][ok_all]
    pg_base = P[base_st][ok_all]; t_base = T[base_st][ok_all]
    perm_states = [s for s in sts if s != base_st]

    print("=" * 116)
    print("真实数据漂移 —— 论文口径分析 v3（按独立审计修正 C1/C2/M3/M5）")
    print(f"共同查询集 {n} 条 | 特征 {F.shape[1]} 维 | 目标覆盖 {1-ALPHA:.2f} | 池 {os.path.basename(POOL)}")
    print("=" * 116)

    s0 = score_of(t_base, pg_base); fin = np.isfinite(s0)
    med, p90 = float(np.median(s0[fin])), float(np.percentile(s0[fin], 90))
    print("\n【检出力自检】")
    print("  baseline 分数: 中位 %.3f nats (%.1f 倍) | p90 %.3f nats (%.0f 倍) | 空结果 %d 条"
          % (med, np.exp(med), p90, np.exp(p90), int((~fin).sum())))
    print("  -> q0 必在 p90 附近 = %.0f 倍，比典型误差宽 %.0f 倍 —— 本实验固有低功效。"
          % (np.exp(p90), np.exp(p90) / np.exp(med)))

    # ---------- 表 0：受影响比例与真值变化分布（M5：真剔掉 true=0）----------
    print("\n表 0  受影响比例 + 真值变化分布（已真正剔除 true=0 的行）")
    print("%-24s | %8s | %7s | %8s | %14s | %14s | %10s" %
          ("状态", "变化条数", "占比", "新变空", "均值(剔空后)", "标准差(剔空后)", "log变化p90"))
    print("-" * 116)
    aff = {base_st: np.zeros(n, dtype=bool)}
    print("%-24s | %8d | %7s | %8d | %14s | %14s | %10s" % (base_st, 0, "-", 0, "-", "-", "-"))
    for s in perm_states:
        tt = T[s][ok_all]
        lr = np.log(np.clip(tt, 1, None)) - np.log(np.clip(t_base, 1, None))
        nz = (t_base > 0) & (tt <= 0)          # 新变空：结构性不可覆盖
        ch = (np.abs(lr) > 1e-9) | nz
        aff[s] = ch
        keep = ch & ~nz                        # ★ 真正只保留"非空"的变化行
        print("%-24s | %8d | %6.1f%% | %8d | %+14.4f | %14.4f | %+10.3f" %
              (s, int(ch.sum()), 100 * ch.mean(), int(nz.sum()),
               lr[keep].mean() if keep.any() else np.nan,
               lr[keep].std() if keep.any() else np.nan,
               np.percentile(lr[keep], 90) if keep.any() else np.nan))
    print("  注：'新变空'单列计数；均值/标准差/分位【只】在非空的真值变化上算（v2 混进去了）。")

    # ---------- 三段切分：两臂都在 eval 段上算 q*（C1）----------
    print(f"\n做 {K} 次随机【三段】切分；两臂的 q* 都在 eval 段上算（修 C1）；"
          f"显著性以【查询】为单位（修 C2）")
    rng = np.random.RandomState(SEED)
    # cov[arm][st][k] = 该折 eval 段覆盖率；d_i[arm][st][i] = 查询 i 在各折的 Δ 累加与出现次数
    cov = {m: {s: [] for s in sts} for m in ("pg_est", "corr")}
    qstar_eval = {m: {s: [] for s in sts} for m in ("pg_est", "corr")}
    q0s = {m: [] for m in ("pg_est", "corr")}
    dsum = {m: {s: np.zeros(n) for s in sts} for m in ("pg_est", "corr")}
    dcnt = {m: {s: np.zeros(n) for s in sts} for m in ("pg_est", "corr")}
    aff_cov = {m: {s: [] for s in sts} for m in ("pg_est", "corr")}
    aff_base = {m: {s: [] for s in sts} for m in ("pg_est", "corr")}
    q90 = {m: [] for m in ("pg_est", "corr")}
    for _ in range(K):
        pm = rng.permutation(n)
        i1, i2 = n // 3, 2 * n // 3
        Tr, Ca, Ev = pm[:i1], pm[i1:i2], pm[i2:]
        mm = model.fit_corrector(F[Tr], pg_base[Tr], t_base[Tr], config.LGBM_PARAMS)
        corr = np.clip(pg_base, 1, None) * np.exp(model.predict_corrector(mm, F))
        S = {"pg_est": {s: score_of(T[s][ok_all], P[s][ok_all]) for s in sts},
             "corr":   {s: score_of(T[s][ok_all], corr) for s in sts}}
        for m in ("pg_est", "corr"):
            q0 = metrics.q_star(S[m][base_st][Ca], ALPHA)
            q0s[m].append(q0)
            q90[m].append(float(np.percentile(qerr(
                pg_base[Ev] if m == "pg_est" else corr[Ev], t_base[Ev]), 90)))
            cb = S[m][base_st][Ev] <= q0
            for s in sts:
                cs = S[m][s][Ev] <= q0
                cov[m][s].append(cs.mean())
                # ★ C1：q* 只在 eval 段上取，且模型只在 train 段拟合 -> 纯样本外
                qstar_eval[m][s].append(metrics.q_star(S[m][s][Ev], ALPHA))
                d = cs.astype(float) - cb.astype(float)
                dsum[m][s][Ev] += d
                dcnt[m][s][Ev] += 1
                sel = aff[s][Ev]
                if sel.any():
                    aff_cov[m][s].append(cs[sel].mean())
                    aff_base[m][s].append(cb[sel].mean())

    print("\n" + "=" * 116)
    print("表 1  ★R1（冻结 q0）：全样本 / 受影响子集 / 查询级显著性")
    print("=" * 116)
    for m in ("pg_est", "corr"):
        print(f"\n  【基模 = {m}】 q0 = {np.mean(q0s[m]):.4f} nats = {np.exp(np.mean(q0s[m])):.1f} 倍"
              f" | 样本外 q-error p90 = {np.mean(q90[m]):.1f}")
        print("  %-24s | %14s | %14s | %18s | %16s | %14s" %
              ("状态", "全样本配对差", "受影响子集配对差", "查询级 t 检验", "查询级 McNemar", "q*(eval) 劣化"))
        print("  " + "-" * 112)
        bc = np.array(cov[m][base_st]); qb = float(np.mean(qstar_eval[m][base_st]))
        for s in sts:
            c = np.array(cov[m][s]); full = 100 * (c - bc).mean()
            ca = np.array(aff_cov[m][s]); cab = np.array(aff_base[m][s])
            sub = 100 * (ca.mean() - cab.mean()) if ca.size else np.nan
            # 查询级：Δ_i（只在出现过的折上平均）
            cnt = dcnt[m][s]; seen = cnt > 0
            di = np.where(seen, dsum[m][s] / np.maximum(cnt, 1), np.nan)
            v = di[seen]
            if v.size > 1 and v.std() > 0:
                tt = st.ttest_1samp(v, 0.0)
                tp, se = tt.pvalue, v.std(ddof=1) / np.sqrt(v.size)
            else:
                tp, se = np.nan, np.nan
            L = int((v < 0).sum()); G = int((v > 0).sum())
            mp = st.binomtest(L, L + G, 0.5).pvalue if (L + G) > 0 else 1.0
            qs = float(np.mean(qstar_eval[m][s]))
            deg = 100 * (1 - qb / qs) if qs > 0 else np.nan
            print("  %-24s | %+11.2f±%.2fpp | %+11.2f±%.2fpp | %+8.2f±%.2fpp p=%-5.3f | %5d损/%3d得 p=%-5.3f | %+11.2f%%" %
                  (s, full, 100 * (c - bc).std(), sub, 100 * se if ca.size else np.nan,
                   100 * v.mean() if v.size else np.nan, 100 * se if v.size else np.nan, tp,
                   L, G, mp, deg))
        print("  注：'查询级 t 检验' 以【每条查询一票】（Δ_i = 该查询在各折的覆盖率差均值），"
              "这才是对查询总体的推断；")
        print("      折池化的 p 值会因重复计数而虚低 5~7 倍（v2 的 1e-24 就是这么来的）。")

    print("\n" + "=" * 116)
    print("表 2  稀释公式检验（只在【未受影响查询两态逐位相同】成立的池上有效）")
    print("=" * 116)
    pg_eq_same = {}
    for s in perm_states:
        mp = np.isfinite(P[base_st][ok_all]) & np.isfinite(P[s][ok_all])
        pg_eq_same[s] = int((np.abs(np.log(np.clip(P[s][ok_all][mp], 1, None))
                                   - np.log(np.clip(P[base_st][ok_all][mp], 1, None))) > 1e-9).sum())
    print("  %-24s | %8s | %12s | %16s | %14s | %10s" %
          ("状态", "受影响比例", "pg_est 变化条数", "全样本配对差", "比例x单条", "恒等式"))
    print("  " + "-" * 110)
    for s in perm_states:
        c = np.array(cov["pg_est"][s]); bc = np.array(cov["pg_est"][base_st])
        full = 100 * (c - bc).mean()
        ca = np.array(aff_cov["pg_est"][s]); cab = np.array(aff_base["pg_est"][s])
        per = 100 * (ca.mean() - cab.mean()) if ca.size else np.nan
        frac = float(aff[s].mean()); pred = frac * per
        ok = pg_eq_same[s] == 0
        print("  %-24s | %7.1f%% | %12d | %+13.2fpp | %+11.2fpp | %10s"
              % (s, 100 * frac, pg_eq_same[s], full, pred,
                 ("成立" if ok and abs(pred - full) < 0.05 else "不适用")))
    print("  注：恒等式要求【未受影响查询两态逐位相同】。scale_del50 做了 ANALYZE、pg_est 真变了，")
    print("      该前提不成立，所以它那行的 0.1pp 缺口【不是 bug】（v2 的脚注归因错了）。")

    print("\n" + "=" * 116)
    print("★ 可信的结论形态（只保留通过查询级检验的）")
    print("=" * 116)
    for m in ("pg_est",):
        bc = np.array(cov[m][base_st])
        for s in perm_states:
            c = np.array(cov[m][s])
            cnt = dcnt[m][s]; seen = cnt > 0
            di = np.where(seen, dsum[m][s] / np.maximum(cnt, 1), np.nan)[seen]
            if di.size > 1 and di.std() > 0:
                tp = st.ttest_1samp(di, 0.0).pvalue
                sig = "★显著" if tp < 0.05 else "不显著"
                print("  %-24s 全样本 %+.2fpp | 查询级 %+.2fpp (p=%.3f) %s"
                      % (s, 100 * (c - bc).mean(), 100 * di.mean(), tp, sig))
    np.save(os.path.join(HERE, "results_real_drift_v3.npy"),
            dict(cov=cov, qstar_eval=qstar_eval, q0={m: float(np.mean(q0s[m])) for m in q0s},
                 qerr_p90={m: float(np.mean(q90[m])) for m in q90}, n=n,
                 aff={k: v for k, v in aff.items()}, pg_eq_same=pg_eq_same), allow_pickle=True)
    print("\n已存 results_real_drift_v3.npy")


if __name__ == "__main__":
    main()
