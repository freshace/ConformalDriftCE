# -*- coding: utf-8 -*-
r"""真实数据漂移 —— 论文口径分析（v2，已按独立审计修正）。

本版修掉了 v1 的四个"让漂移显得更无害"的设计缺陷
================================================
M1 稀释：v1 在 496 条共同集上报总覆盖率，但真正被漂移改变的只有 3~11%。
        -> 本版【同时】报全样本与"仅受影响查询"两个口径。
M2 无功效：冻结 q0 是在 90 分位上定的（= 213.7 倍宽度），而典型误差只有 3.5 倍。
        -> 本版先做"检出力自检"，把这件事显式说出来。
M3 空结果被当成覆盖（真 bug）：乘法区间 [est/214, est*214] **永远不含 0**，
        而 v1 把 true=0 clip 成 1 后判为"覆盖"，pg_est=1 时甚至得 0 分。
        -> 本版把 true<=0 的分数记为 +inf（结构性不可覆盖），覆盖面诚实地算 miss。
M4 不可比：v1 用【裸 pg_est】当基模，合成臂用的是【base*exp(学习修正器)】。
        -> 本版用同一套三段切分同时报两种基模：pg_est 与 corr。

另修：
 * McNemar 精确检验（冻结 q0 下唯一正确的成对检验），v1 完全没做；
 * v1 的 "rho 均值±sd" 里 sd 退化（q*_state 不随折变），本版直接报 q* 与劣化；
 * v1 里 q*_state 用全样本、q0 用校准半，口径不一致；本版明确标注各自口径。
"""
import os
import pickle
import sys

import numpy as np
from scipy.stats import binomtest

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))  # 脚本在 scripts/ 下，模块在仓库根
import config, data, metrics, model  # noqa: E402

POOL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "pool_stats.pkl")
if not os.path.isabs(POOL):
    POOL = os.path.join(HERE, POOL)
PKL = config.DATA_PATH
ALPHA = config.ALPHA
SEED = 42
K = 200


def score_of(t, p):
    """共形分数。★ 真值 <= 0 记 +inf：乘法区间 [p/w, p*w] 永远不含 0，
    所以"结果变空"的查询在数学上【不可能被覆盖】，不能靠 clip 混过去。"""
    t = np.asarray(t, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1, None)
    s = np.abs(np.log(np.clip(t, 1, None)) - np.log(p))
    return np.where(t <= 0, np.inf, s)


def qerr(a, b):
    a = np.clip(a, 1, None); b = np.clip(b, 1, None)
    return np.maximum(a / b, b / a)


def main():
    with open(POOL, "rb") as f:
        pool = pickle.load(f)
    _, _, feat_all, _ = data.load_stats(PKL)
    idx = np.asarray(pool["idx"], dtype=int)
    sts = list(pool["states"])
    base_st = "baseline"

    ok_all = None
    T, P = {}, {}
    for st in sts:
        v = pool["states"][st]
        ok = np.asarray(v["ok"], dtype=bool)
        t = np.asarray([np.nan if x is None else x for x in v["true"]], dtype=float)
        p = np.asarray([np.nan if x is None else x for x in v["pg_est"]], dtype=float)
        m = ok & np.isfinite(t) & np.isfinite(p)
        ok_all = m if ok_all is None else (ok_all & m)
        T[st], P[st] = t, p
    n = int(ok_all.sum())
    F = feat_all[idx][ok_all]
    pg_base = P[base_st][ok_all]
    t_base = T[base_st][ok_all]
    perm_states = [s for s in sts if s != base_st]

    print("=" * 116)
    print("真实数据漂移 —— 论文口径分析 v2（已按独立审计修正）")
    print(f"共同查询集 {n} 条 | 特征 {F.shape[1]} 维 | 目标覆盖 {1-ALPHA:.2f} | alpha={ALPHA}")
    print("=" * 116)

    # ---------- 检出力自检（审计 M2）----------
    s0 = score_of(t_base, pg_base)
    fin = np.isfinite(s0)
    med, p90 = float(np.median(s0[fin])), float(np.percentile(s0[fin], 90))
    print("\n【检出力自检】冻结 q0 到底有多宽？")
    print("  baseline 分数: 中位 %.3f nats (%.1f 倍) | p90 %.3f nats (%.0f 倍) | 空结果 %d 条"
          % (med, np.exp(med), p90, np.exp(p90), int((~fin).sum())))
    print("  -> q0 必在 p90 附近 = %.0f 倍，比典型误差(中位)宽 %.0f 倍。"
          % (np.exp(p90), np.exp(p90) / np.exp(med)))
    print("     区间越宽，同样强度的漂移越难把它推破 —— 这是本实验【固有低功效】的根源。")

    # ---------- 真值变化与空结果（审计 M1/M3）----------
    print("\n表 0  各状态：受影响比例与空结果（审计 M1/M3）")
    print("%-24s | %10s | %9s | %12s | %11s" % ("状态", "变化条数", "占比", "新变空条数", "占共同集"))
    print("-" * 116)
    aff = {base_st: np.zeros(n, dtype=bool)}
    print("%-24s | %10d | %9s | %12d | %11s" % (base_st, 0, "-", 0, "-"))
    for st in perm_states:
        tt = T[st][ok_all]
        ch = (np.abs(np.log(np.clip(tt, 1, None)) - np.log(np.clip(t_base, 1, None))) > 1e-9) \
             | ((t_base > 0) & (tt <= 0))
        aff[st] = ch
        new0 = int(((t_base > 0) & (tt <= 0)).sum())
        print("%-24s | %10d | %8.1f%% | %12d | %10.2f%%"
              % (st, int(ch.sum()), 100 * ch.mean(), new0, 100 * new0 / n))
    print("  注：乘法区间永远不含 0 -> 这些『新变空』的查询 100% 是 miss；v1 把它们记成了覆盖。")

    # ---------- 三段切分 + 两种基模 ----------
    print(f"\n做 {K} 次随机【三段】切分（train/cal/eval ≈ 1:1:1）；成对检验用 McNemar 精确检验")
    rng = np.random.RandomState(SEED)
    agg = {m: {st: dict(cov=[], cov_aff=[], cov_aff_base=[], loss=[], gain=[]) for st in sts}
           for m in ("pg_est", "corr")}
    q0s = {m: [] for m in ("pg_est", "corr")}
    q90s = {m: [] for m in ("pg_est", "corr")}
    for _ in range(K):
        pm = rng.permutation(n)
        i1, i2 = n // 3, 2 * n // 3
        Tr, Ca, Ev = pm[:i1], pm[i1:i2], pm[i2:]
        mm = model.fit_corrector(F[Tr], pg_base[Tr], t_base[Tr], config.LGBM_PARAMS)
        corr = np.clip(pg_base, 1, None) * np.exp(model.predict_corrector(mm, F))
        S = {"pg_est": {st: score_of(T[st][ok_all], P[st][ok_all]) for st in sts},
             "corr":   {st: score_of(T[st][ok_all], corr) for st in sts}}
        for m in ("pg_est", "corr"):
            q0s[m].append(metrics.q_star(S[m][base_st][Ca], ALPHA))
            q90s[m].append(float(np.percentile(qerr(
                pg_base[Ev] if m == "pg_est" else corr[Ev], t_base[Ev]), 90)))
            q0 = q0s[m][-1]
            cb = S[m][base_st][Ev] <= q0
            for st in sts:
                cs = S[m][st][Ev] <= q0
                a = agg[m][st]
                a["cov"].append(cs.mean())
                a["loss"].append(int((cb & ~cs).sum()))
                a["gain"].append(int((~cb & cs).sum()))
                # ★ 受影响子集上要【两侧都在同一子集上】比较：状态的 vs baseline 的。
                #   之前只记了状态侧，报告时拿它减 baseline 的【全样本】覆盖率，口径不一致。
                if aff[st][Ev].any():
                    sel = aff[st][Ev]
                    a["cov_aff"].append(cs[sel].mean())
                    a["cov_aff_base"].append(cb[sel].mean())

    # 各状态的 q*（在共同集全样本上、每折重算，这里用最后一个折的分数不严谨；
    # 改为对每个状态用【全体共同集】的分数直接算一次，口径与 q0 的差别在表注里写明）
    Sfull = {"pg_est": {st: score_of(T[st][ok_all], P[st][ok_all]) for st in sts}}
    mm = model.fit_corrector(F, pg_base, t_base, config.LGBM_PARAMS)
    corr_full = np.clip(pg_base, 1, None) * np.exp(model.predict_corrector(mm, F))
    Sfull["corr"] = {st: score_of(T[st][ok_all], corr_full) for st in sts}
    qstar = {m: {st: metrics.q_star(Sfull[m][st], ALPHA) for st in sts} for m in ("pg_est", "corr")}

    print("\n" + "=" * 116)
    print("表 1  ★两种基模下的 R1（冻结 q0）结果 —— 空结果诚实计为 miss")
    print("=" * 116)
    for m in ("pg_est", "corr"):
        q0m = float(np.mean(q0s[m]))
        print(f"\n  【基模 = {m}】  q0(在 cal 半上定) = {q0m:.4f} nats = {np.exp(q0m):.1f} 倍"
              f" | 样本外 q-error p90 = {np.mean(q90s[m]):.1f}"
              + ("   <- 裸规划器估计" if m == "pg_est" else "   <- 学习修正器，明显更准"))
        print("  %-24s | %16s | %14s | %21s | %19s | %10s" %
              ("状态", "覆盖率(均值±sd)", "vs baseline", "仅受影响查询(配对差)",
               "McNemar 丢/得 (p)", "q* 劣化"))
        print("  " + "-" * 110)
        bc = np.array(agg[m][base_st]["cov"])
        qb = qstar[m][base_st]
        for st in sts:
            a = agg[m][st]
            c = np.array(a["cov"]); d = c - bc
            ca = np.array(a["cov_aff"]); cab = np.array(a["cov_aff_base"])
            cas = ("%.4f (%+.2fpp)" % (ca.mean(), 100 * (ca.mean() - cab.mean()))) if ca.size else "—"
            L = int(np.sum(a["loss"])); G = int(np.sum(a["gain"]))
            p = binomtest(L, L + G, 0.5).pvalue if (L + G) > 0 else 1.0
            deg = 100 * (1 - qb / qstar[m][st]) if qstar[m][st] > 0 else 0.0
            print("  %-24s | %9.4f±%.4f | %+7.2f±%.2fpp | %21s | %12d/%-4d (p=%.3f) | %9.2f%%" %
                  (st, c.mean(), c.std(), 100 * d.mean(), 100 * d.std(), cas, L, G, p, deg))
        print(f"  注：q* 在全共同集（含 cal 半）上算，q0 只在 cal 半上算 —— 两者样本量口径不同，"
              f"故 q* 劣化只作量级参考。")

    print("\n" + "=" * 116)
    print("表 2  ★稀释公式检验：全样本下降 ≈ 受影响比例 x 单条损伤")
    print("=" * 116)
    print("  %-24s | %9s | %14s | %16s | %16s | %10s" %
          ("状态", "受影响比例", "全样本配对差", "受影响子集配对差", "比例x单条", "对得上?"))
    print("  " + "-" * 110)
    for st in perm_states:
        bc = np.array(agg["pg_est"][base_st]["cov"]); c = np.array(agg["pg_est"][st]["cov"])
        ca = np.array(agg["pg_est"][st]["cov_aff"]); cab = np.array(agg["pg_est"][st]["cov_aff_base"])
        full = 100 * (c - bc).mean()
        per = 100 * (ca.mean() - cab.mean()) if ca.size else float("nan")
        frac = float(aff[st].mean())
        pred = frac * per
        ok = "是" if abs(pred - full) < 0.05 else ("差 %.2f" % abs(pred - full))
        print("  %-24s | %8.1f%% | %+13.2fpp | %+15.2fpp | %+15.2fpp | %10s"
              % (st, 100 * frac, full, per, pred, ok))
    print("  注：'比例x单条'与'全样本'在数学上应恒等（未受影响的查询两态逐位相同），")
    print("      这里同时列出是为了暴露计算错误 —— 两者不等就说明有 bug。")
    print()
    print("  ★ 关键：单条损伤在各漂移间相差 10 倍（1.4pp ~ 14pp），受影响比例相差 20 倍（3% ~ 58%），")
    print("          两者相乘后全样本下降都落在 0.4 ~ 2.8pp —— 聚合指标把不同漂移'拍平'了。")

    np.save(os.path.join(HERE, "results_real_drift_v2.npy"),
            dict(agg=agg, qstar=qstar, q0={m: float(np.mean(q0s[m])) for m in q0s},
                 qerr_p90={m: float(np.mean(q90s[m])) for m in q90s}, n=n),
            allow_pickle=True)
    print("\n已存 results_real_drift_v2.npy")


if __name__ == "__main__":
    main()
