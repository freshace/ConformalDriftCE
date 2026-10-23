# -*- coding: utf-8 -*-
r"""真实数据漂移 —— 换用【更准的基模】后，漂移效应会不会显现？

动机
----
真实数据实验给出"漂移几乎不破坏覆盖"（覆盖率配对差 −0.43pp、rho 劣化 1.19%）。
但有一个必须排除的解释：**PG 的规划估算太差了**（q-error p50 3.5、p90 200、p99 22782，
log 误差 std 2.28）。分数分布的 90 分位完全由"本来就错得离谱"的查询决定，
少数查询的真值变化淹不进去 —— 于是"漂移无害"可能只是"基模太烂"的副产品。

做法
----
用本地 622 维特征训 LGBM 残差修正器（与合成实验同一套 `model.fit_corrector`）：
    corr = pg_est * exp( predictor(feat) )
关键：**保边际置换下特征逐位不变**（§6.2 g 已证），所以修正器对同一查询在所有
置换状态下给出**同一个** corr —— 它是一个"冻结的、更准的基模"。
于是可以干净地对比：同样的一批漂移，基模从 pg_est 换成 corr 后，覆盖/宽度劣化多少。

只在 **perm_\*** 三个状态上做（scale_del 的特征无效，无法套修正器）。

口径与 real_drift_analysis.py 保持一致：多次随机对半切分 + 同折配对比较。
"""
import os
import pickle
import sys

import warnings
import numpy as np

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))  # 脚本在 scripts/ 下，模块在仓库根
import config, data, metrics, model  # noqa: E402

POOL = os.path.join(HERE, "pool_stats.pkl")
PKL = config.DATA_PATH
ALPHA = config.ALPHA
SEED = 42
K = 200


def log_score(t, p):
    return np.abs(np.log(np.clip(t, 1, None)) - np.log(np.clip(p, 1, None)))


def qerr(a, b):
    a = np.clip(a, 1, None); b = np.clip(b, 1, None)
    return np.maximum(a / b, b / a)


def main():
    with open(POOL, "rb") as f:
        pool = pickle.load(f)
    base_all, true_all, feat_all, nfc_all = data.load_stats(PKL)
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
    print("=" * 104)
    print(f"真实数据：换更准基模 | 共同查询集 {n} 条 | 特征维数 {F.shape[1]} | 目标覆盖 {1-ALPHA:.2f}")
    print("=" * 104)

    # ---------- 先看修正器能把基模提升多少（用一半训、另一半评）----------
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    A, B = perm[:n // 2], perm[n // 2:]
    print("\n训练 LGBM 残差修正器（在 baseline-A 上，%d 条）" % len(A))
    m = model.fit_corrector(F[A], pg_base[A], t_base[A], config.LGBM_PARAMS)
    corr_all = np.clip(pg_base, 1, None) * np.exp(model.predict_corrector(m, F))
    print("\n表 0  基模质量对比（在留出的 baseline-B 上，%d 条）" % len(B))
    print("%-26s %8s %8s %9s %12s %12s" % ("基模", "q p50", "q p90", "q p99", "log误差std", "log误差p90"))
    for nm, pred in [("pg_est（规划的估算）", pg_base[B]), ("corr（+LGBM 修正器）", corr_all[B])]:
        q = qerr(pred, t_base[B])
        le = log_score(t_base[B], pred)
        print("%-26s %8.2f %8.1f %9.1f %12.4f %12.4f" %
              (nm, *np.percentile(q, [50, 90, 99]), le.std(), np.percentile(le, 90)))
    print("  -> 修正器把 log 误差 std 从 %.3f 降到 %.3f（%.1f 倍）"
          % (log_score(t_base[B], pg_base[B]).std(), log_score(t_base[B], corr_all[B]).std(),
             log_score(t_base[B], pg_base[B]).std() / max(log_score(t_base[B], corr_all[B]).std(), 1e-9)))

    perm_states = [s for s in sts if s.startswith("perm_")]

    # ---------- 多次【三段】切分 ----------
    # ★ 必须三段，不能两段：修正器是在 train 段上拟合的，它在 train 段上的误差是
    #   【样本内】的、远小于样本外。若用同一段既训模型又定 q0，q0 会被压低，
    #   到评估段覆盖率必然崩 —— 第一版两段切分实测 baseline 覆盖只有 0.265，就是这个 bug。
    #   合成实验用的也是三段（train 20000 / cal 3000 / stream 10000），这里保持一致。
    print(f"\n做 {K} 次随机【三段】切分（train/cal/eval ≈ 1:1:1），比较两种基模下的漂移劣化")
    rng = np.random.RandomState(SEED)
    res = {mdl: {st: dict(cov=[], rho=[]) for st in [base_st] + perm_states}
           for mdl in ("pg_est", "corr")}
    corr_q90, pg_q90 = [], []
    for k in range(K):
        pm = rng.permutation(n)
        i1, i2 = n // 3, 2 * n // 3
        T_, C_, E_ = pm[:i1], pm[i1:i2], pm[i2:]
        mm = model.fit_corrector(F[T_], pg_base[T_], t_base[T_], config.LGBM_PARAMS)
        c2 = np.clip(pg_base, 1, None) * np.exp(model.predict_corrector(mm, F))
        corr_q90.append(np.percentile(qerr(c2[E_], t_base[E_]), 90))
        pg_q90.append(np.percentile(qerr(pg_base[E_], t_base[E_]), 90))
        S_pg = {st: log_score(T[st][ok_all], P[st][ok_all]) for st in [base_st] + perm_states}
        S_co = {st: log_score(T[st][ok_all], c2) for st in [base_st] + perm_states}
        for mdl, S in (("pg_est", S_pg), ("corr", S_co)):
            q0 = metrics.q_star(S[base_st][C_], ALPHA)          # 在【校准段】定 q0
            for st in [base_st] + perm_states:
                res[mdl][st]["cov"].append(metrics.coverage_at_q(S[st][E_], q0))  # 在【评估段】测
                qs = metrics.q_star(S[st], ALPHA)
                res[mdl][st]["rho"].append(q0 / qs if qs > 0 else np.nan)
    print("  修正器【样本外】q-error p90 = %.1f   （pg_est = %.1f，提升 %.1f 倍）"
          % (float(np.mean(corr_q90)), float(np.mean(pg_q90)),
             float(np.mean(pg_q90)) / max(float(np.mean(corr_q90)), 1e-9)))

    print("\n表 1  ★两种基模下：R1（冻结 q0）的覆盖率与 rho 劣化")
    for mdl in ("pg_est", "corr"):
        print("\n  【基模 = %s】" % mdl)
        print("  %-24s | %16s | %14s | %10s | %14s | %12s" %
              ("状态", "覆盖率(均值±sd)", "vs baseline", "rho", "rho(均值±sd)", "rho劣化"))
        print("  " + "-" * 100)
        bc = np.array(res[mdl][base_st]["cov"])
        br = float(np.mean(res[mdl][base_st]["rho"]))
        for st in [base_st] + perm_states:
            c = np.array(res[mdl][st]["cov"]); r = float(np.mean(res[mdl][st]["rho"]))
            d = c - bc
            deg = (br - r) / br * 100 if br else np.nan
            print("  %-24s | %9.4f±%.4f | %+7.2f±%.2fpp | %9.4f±%.4f | %+10.2f%%" %
                  (st, c.mean(), c.std(), 100 * d.mean(), 100 * d.std(), r,
                   float(np.std(res[mdl][st]["rho"])), deg))

    print("\n" + "=" * 104)
    print("结论")
    print("=" * 104)
    for mdl in ("pg_est", "corr"):
        worst = max(perm_states, key=lambda s: (float(np.mean(res[mdl][s]["cov"])) -
                                                float(np.mean(res[mdl][base_st]["cov"]))))
        bc = float(np.mean(res[mdl][base_st]["cov"]))
        print("  基模=%-7s baseline覆盖 %.4f | 最差漂移状态 %-24s 覆盖 %.4f (配对差 %+.2fpp)"
              % (mdl, bc, worst, float(np.mean(res[mdl][worst]["cov"])),
                 100 * (float(np.mean(res[mdl][worst]["cov"])) - bc)))
    b1 = float(np.mean(res["pg_est"][base_st]["cov"]))
    w1 = min(100 * (float(np.mean(res["pg_est"][s]["cov"])) - b1) for s in perm_states)
    b2 = float(np.mean(res["corr"][base_st]["cov"]))
    w2 = min(100 * (float(np.mean(res["corr"][s]["cov"])) - b2) for s in perm_states)
    print("\n  最差配对差：pg_est 基模 %+.2fpp  ->  corr 基模 %+.2fpp" % (w1, w2))
    print("  => 基模变准后漂移破坏力放大 %.1f 倍" % (abs(w2) / max(abs(w1), 1e-9))
          if abs(w1) > 1e-9 else "  => pg_est 基模下配对差为 0，无法比较倍数")

    np.save(os.path.join(HERE, "results_better_base.npy"), res, allow_pickle=True)
    print("\n已存 results_better_base.npy")


if __name__ == "__main__":
    main()
