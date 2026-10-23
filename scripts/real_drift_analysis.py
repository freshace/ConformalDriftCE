# -*- coding: utf-8 -*-
"""真实数据漂移池 —— 按论文口径做最终分析（R1：冻结 q0，不重校准）。

要回答的问题
------------
合成套件里，"漂移 → 冻结区间覆盖跌破目标"是核心现象：
    FROZEN 覆盖率 62.7 / 79.9 / 81.1 / 85.5%（uniform/periodic/trend/long-tail）
    效率比 rho = q_reported / q* 低到 0.610
现在问：**真实数据漂移**下，这件事还成立吗？还是结论翻转？

口径
----
* 共同查询集 = 各状态 ok 的交集（跨状态比较【只能】用这个集合）。
* 分数 s = |log(clip(true,1)) - log(clip(pg_est,1))|（真值可能为 0，clip 到 1）。
* **严格 split**：把共同集按固定种子对半切成 A/B。
    在 baseline-A 上校准 q0 = q*(alpha)（与 conformal.calibrate_q 同口径）；
    在 baseline-B 与各状态-B 上测覆盖率。
  这样校准集与评估集不重叠，覆盖率不是同义反复。
* 同时报 q*_state（各状态自身的最小诚实宽度）与 rho = q0/q*_state：
    rho < 1 = 报告宽度不足（会静默失效）；rho ~ 1 = 诚实；rho > 1 = 用宽度买覆盖。
"""
import os
import pickle
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))  # 脚本在 scripts/ 下，模块在仓库根
import config, data, metrics  # noqa: E402

POOL = os.path.join(HERE, "pool_stats.pkl")
PKL = config.DATA_PATH
ALPHA = config.ALPHA          # 0.10 -> 目标覆盖 0.90
SEED = 42


def score_of(t, p):
    """与 conformal.score 同口径：对数尺度上的绝对误差，真值/pg_est 均 clip 到 >=1。"""
    return np.abs(np.log(np.clip(t, 1, None)) - np.log(np.clip(p, 1, None)))


def main():
    with open(POOL, "rb") as f:
        pool = pickle.load(f)
    base_all, true_all, feat_all, nfc_all = data.load_stats(PKL)
    idx = np.asarray(pool["idx"], dtype=int)
    sts = list(pool["states"])
    base_st = "baseline" if "baseline" in sts else sts[0]

    # 共同查询集
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
    n_all = int(ok_all.sum())

    print("=" * 104)
    print(f"真实数据漂移 —— 论文口径分析 | 共同查询集 {n_all} 条 | 目标覆盖 {1-ALPHA:.2f} | alpha={ALPHA}")
    print("=" * 104)

    S = {st: score_of(T[st][ok_all], P[st][ok_all]) for st in sts}

    # ---- 多次随机切分：单次对半切的覆盖率有 ±2-3pp 有限样本波动（见 baseline 自己就 87%），
    #      所以必须多次切分并做【配对】比较（同一折内 baseline 与漂移状态用同一个 q0）。
    K = 200
    print(f"\n做 {K} 次随机对半切分（每次在 baseline-校准半上定 q0，在评估半上测各状态覆盖率）")
    rng = np.random.RandomState(SEED)
    covs = {st: [] for st in sts}
    rhos = {st: [] for st in sts}
    q0s = []
    for k in range(K):
        perm = rng.permutation(n_all)
        A, B = perm[:n_all // 2], perm[n_all // 2:]
        q0 = metrics.q_star(S[base_st][A], ALPHA)
        q0s.append(q0)
        for st in sts:
            covs[st].append(metrics.coverage_at_q(S[st][B], q0))
            qs = metrics.q_star(S[st], ALPHA)
            rhos[st].append(q0 / qs if qs > 0 else np.nan)
    q0m = float(np.mean(q0s))
    print(f"  平均 q0 = {q0m:.4f}（exp = {np.exp(q0m):.1f}x），跨折标准差 {np.std(q0s):.4f}")

    print("\n表 I  ★R1（冻结 q0，不重校准）—— 真实漂移是否破坏覆盖保证？")
    print("  %d 折平均；vs基准 = 同一折内与 baseline 的配对差（消除折间波动）" % K)
    print("%-24s | %8s | %16s | %14s | %8s | %14s | %12s" %
          ("状态", "true变化", "覆盖率(均值±标准差)", "vs baseline", "q*_state", "rho(均值±sd)", "rho 相对劣化"))
    print("-" * 120)
    base_cov = np.array(covs[base_st])
    base_rho = float(np.mean(rhos[base_st]))
    rows = {}
    for st in sts:
        c = np.array(covs[st])
        d = c - base_cov                      # 配对差
        r = float(np.mean(rhos[st]))
        rs = float(np.std(rhos[st]))
        qs = metrics.q_star(S[st], ALPHA)
        mt = np.isfinite(T[base_st][ok_all]) & np.isfinite(T[st][ok_all])
        lrt = np.log(np.clip(T[st][ok_all][mt], 1, None)) - np.log(np.clip(T[base_st][ok_all][mt], 1, None))
        nch = int((np.abs(lrt) > 1e-9).sum())
        deg = (base_rho - r) / base_rho * 100 if base_rho else np.nan
        rows[st] = dict(cov=float(c.mean()), cov_sd=float(c.std()), dmean=float(d.mean()),
                        dsd=float(d.std()), qs=qs, rho=r, rho_sd=rs, deg=deg, nch=nch)
        print("%-24s | %8d | %9.4f±%.4f | %+7.2f±%.2fpp | %8.4f | %9.4f±%.4f | %+10.2f%%" %
              (st, nch, c.mean(), c.std(), 100 * d.mean(), 100 * d.std(), qs, r, rs, deg))

    print("\n表 II  与合成套件结论对照（合成来自 experiment_qstar_report / experiment_aci_baseline）")
    print("-" * 120)
    print("  合成 FROZEN 覆盖率 62.7 / 79.9 / 81.1 / 85.5%  (uniform/periodic/trend/long-tail)")
    print("  合成 FROZEN rho     0.610 / 0.783 / 0.792 / 0.891   -> 相对 none(0.979) 劣化 20~38%")
    print()
    print("  真实（本池）：")
    for st in sts:
        r = rows[st]
        print("    %-24s 覆盖 %.4f (%+.2fpp vs baseline)  rho %.4f (劣化 %+.2f%%)"
              % (st, r["cov"], 100 * r["dmean"], r["rho"], r["deg"]))
    print()
    worst = max((st for st in sts if st != base_st), key=lambda s: rows[s]["deg"])
    print("  真实最大 rho 劣化 = %.2f%% (%s)；合成最小劣化 = 20.4%% (uniform)"
          % (rows[worst]["deg"], worst))
    ratio = 20.4 / max(rows[worst]["deg"], 1e-9)
    print("  => 合成注入对 rho 的破坏力约为真实相关漂移的 **%.1f 倍**" % ratio)
    print("  => 覆盖率：合成掉 4.5~27.3pp；真实变化 %+.2f~%+.2fpp（基本不动）"
          % (100 * min(rows[s]["dmean"] for s in sts if s != base_st),
             100 * max(rows[s]["dmean"] for s in sts if s != base_st)))

    print("\n表 III  逐状态 q*(alpha) 与「为达标需要多宽」")
    print("%-24s | %11s | %11s | %11s | %s" % ("状态", "q*(alpha)", "exp(q*)", "相对基准", "相对 q0 需加宽"))
    print("-" * 120)
    qb = rows[base_st]["qs"]
    for st in sts:
        qs = rows[st]["qs"]
        print("%-24s | %11.4f | %10.2fx | %10.2fx | %10.2fx" %
              (st, qs, float(np.exp(qs)), float(np.exp(qs - qb)), float(np.exp(qs - q0m))))

    out = os.path.join(HERE, "results_real_drift_analysis.npy")
    np.save(out, dict(rows=rows, q0_mean=q0m, q0_sd=float(np.std(q0s)), K=K,
                      n_common=n_all), allow_pickle=True)
    print(f"\n已存 {out}")


if __name__ == "__main__":
    main()
