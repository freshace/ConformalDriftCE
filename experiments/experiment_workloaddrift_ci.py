# -*- coding: utf-8 -*-
r"""Phase 1：workload drift 的方法学补强（重复切分 + CI + 混淆拆分）。

为什么要重做
------------
现有两个脚本对**同一个现象**给出不同数字：
    experiment_workloaddrift.txt : 漂移后冻结覆盖 50.8%
    experiment_baselines.txt     : 工作负载 冻结覆盖 45.8%
根因（已定位）：前者 rng 从种子 42 首次使用就抽划分；后者在第 66 行已消耗 RNG 状态，
且复杂查询测试集取 4000 条（前者取 500）。**5pp 的差 = 抽样波动。**

本脚本回答三件事
----------------
Q1  冻结覆盖到底是多少？（R 次独立划分，给 mean±SE 与 95% CI，逐次一票）
Q2  测试集大小 500 vs 4000 对 CI 宽度的影响
Q3  ★混淆拆分：50.8% 里有多少是「模型不会外推」，多少是「校准不可交换」？
      做法：训练集里混入 x% 复杂查询，**校准集始终只用简单查询**
        x=0   → 现状（既外推不足，又不可交换）
        x=100 → 模型见过复杂查询，只剩「校准转移」这一个成分
      若 x=100 时冻结覆盖仍崩 → 纯粹是校准转移问题（正对 ICDE 2022 的 future work）
      若 x=100 时恢复到 ~90% → 是模型能力问题，本漂移要降级

用法
----
    python experiments/experiment_workloaddrift_ci.py --reps 100
    python experiments/experiment_workloaddrift_ci.py --reps 40 --test-size 4000
    python experiments/experiment_workloaddrift_ci.py --reps 40 --mix 100
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config  # noqa: E402
import conformal  # noqa: E402
import features  # noqa: E402
import model  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "results_repro")
CACHE = os.path.join(OUT, "_wl_cache.npz")


def load_cached():
    """载入并缓存 (base, true, feat, ntab)。缓存避免每次重读 204 MB pkl。"""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        print("  [cache] 命中 %s" % os.path.basename(CACHE))
        return z["base"], z["true"], z["feat"], z["ntab"]
    t0 = time.time()
    n_ts, n_cs = features.load_table_col_space(config.DATA_PATH)
    with open(config.DATA_PATH, "rb") as f:
        data = pickle.load(f)
    base_l, true_l, feat_l, ntab_l = [], [], [], []
    for ft, label, pgv, *_, ex in data["features"]:
        t = float(np.expm1(label - 1)); p = float(np.expm1(pgv - 1))
        if t <= 0 or p <= 0:
            continue
        f = np.nan_to_num(
            np.concatenate([features.build_schema_feature(ft, ex, n_ts, n_cs), features.stats9(ft)]),
            nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        f = np.hstack([f, np.log(np.clip(p, 1, None))]).astype(np.float32)
        base_l.append(p); true_l.append(t); feat_l.append(f)
        ntab_l.append(len(ex.get("table_indices", [])))
    out = (np.array(base_l), np.array(true_l), np.asarray(feat_l, np.float32),
           np.array(ntab_l, np.int32))
    np.savez(CACHE, base=out[0], true=out[1], feat=out[2], ntab=out[3])
    print("  [cache] 已建 %s（%.1fs）" % (os.path.basename(CACHE), time.time() - t0))
    return out


def one_split(base, true, feat, ntab, rng, n_train, n_cal, n_test, mix_pct):
    """一次独立划分；返回该次切分的各项覆盖率。"""
    simple = np.where(ntab <= 3)[0]
    cplx = np.where(ntab >= 5)[0]

    # ---- 训练集：可混入 mix_pct% 的复杂查询（占训练集的比例）----
    n_cplx_in_train = int(round(n_train * mix_pct / 100.0))
    n_simple_in_train = n_train - n_cplx_in_train
    tr_s = rng.choice(simple, size=n_simple_in_train, replace=False)
    tr_c = rng.choice(cplx, size=n_cplx_in_train, replace=False) if n_cplx_in_train else np.array([], int)
    tr = np.concatenate([tr_s, tr_c])

    # ---- 校准集：★始终只用简单查询（这是 workload drift 的定义）----
    rest_s = np.setdiff1d(simple, tr_s)
    cal = rng.choice(rest_s, size=min(n_cal, len(rest_s)), replace=False)

    # ---- 测试集：只测复杂查询；简单查询作为 none 对照 ----
    # ★ 必须把 tr_c 从复杂池里剔除：否则 mix>0 时训练集与测试集重叠，
    #   覆盖率会被"见过测试查询"虚高（mix=100 时预期重叠约 1500*500/18473 ≈ 4%）。
    cplx_pool = np.setdiff1d(cplx, tr_c) if len(tr_c) else cplx
    te = rng.choice(cplx_pool, size=min(n_test, len(cplx_pool)), replace=False)
    te_s = rng.choice(rest_s, size=min(n_test, len(rest_s)), replace=False)

    m_old = model.fit_corrector(feat[tr], base[tr], true[tr], config.LGBM_PARAMS)
    sc_cal = conformal.score(true[cal], base[cal], model.predict_corrector(m_old, feat[cal]))
    q_old = conformal.calibrate_q(sc_cal, config.ALPHA)

    r = {}
    r["cal_cov"] = conformal.coverage(sc_cal, q_old)                       # 应为 ~90%
    r["none_cov"] = conformal.coverage(                                    # 简单->简单 对照
        conformal.score(true[te_s], base[te_s], model.predict_corrector(m_old, feat[te_s])), q_old)
    sc_d = conformal.score(true[te], base[te], model.predict_corrector(m_old, feat[te]))
    r["frozen_cov"] = conformal.coverage(sc_d, q_old)
    r["frozen_width"] = float(np.exp(q_old))

    # ---- Oracle：★标签无限（复杂池里除测试集外的数据训练，上限 5000 条以控制耗时），
    #      但【校准集仍然只用简单查询】—— 与 frozen 只差训练数据，隔离"标签量" ----
    pool_o = np.setdiff1d(cplx, te)
    if len(pool_o) > 5000:
        pool_o = rng.choice(pool_o, size=5000, replace=False)
    m_or = model.fit_corrector(feat[pool_o], base[pool_o], true[pool_o], config.LGBM_PARAMS)
    q_or = conformal.calibrate_q(
        conformal.score(true[cal], base[cal], model.predict_corrector(m_or, feat[cal])), config.ALPHA)
    r["oracle_cov"] = conformal.coverage(
        conformal.score(true[te], base[te], model.predict_corrector(m_or, feat[te])), q_or)

    # ---- 适应：从复杂查询里再抽 200 训练 + 200 校准，其余评估 ----
    perm = rng.permutation(len(te))
    a_tr, a_cal, a_ev = te[perm[:200]], te[perm[200:400]], te[perm[400:]]
    if len(a_ev) < 20:
        r["adapt_cov"] = np.nan
        return r
    m_new = model.fit_corrector(feat[a_tr], base[a_tr], true[a_tr], config.LGBM_PARAMS)
    sc_ac = conformal.score(true[a_cal], base[a_cal], model.predict_corrector(m_new, feat[a_cal]))
    q_new = conformal.calibrate_q(sc_ac, config.ALPHA)
    r["adapt_cov"] = conformal.coverage(
        conformal.score(true[a_ev], base[a_ev], model.predict_corrector(m_new, feat[a_ev])), q_new)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--test-size", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-cal", type=int, default=300)
    ap.add_argument("--mix", type=int, default=0, help="训练集里复杂查询的百分比")
    a = ap.parse_args()

    print("=" * 100)
    print("Phase 1  workload drift 方法学：重复切分 + CI   reps=%d test=%d train=%d cal=%d mix=%d%%"
          % (a.reps, a.test_size, a.n_train, a.n_cal, a.mix))
    print("=" * 100)
    base, true, feat, ntab = load_cached()
    print("  样本 %d；简单(≤3表) %d；复杂(≥5表) %d"
          % (len(ntab), int((ntab <= 3).sum()), int((ntab >= 5).sum())))
    print()

    keys = ["cal_cov", "none_cov", "frozen_cov", "adapt_cov", "oracle_cov", "frozen_width"]
    acc = {k: [] for k in keys}
    t0 = time.time()
    for r in range(a.reps):
        rng = np.random.RandomState(10000 + r)
        res = one_split(base, true, feat, ntab, rng, a.n_train, a.n_cal, a.test_size, a.mix)
        for k in keys:
            acc[k].append(res.get(k, np.nan))
        if (r + 1) % max(1, a.reps // 10) == 0:
            print("    %d/%d  (%.0fs)" % (r + 1, a.reps, time.time() - t0))

    print("\n" + "=" * 100)
    print("Q1/Q2  各项覆盖率的分布（%d 次独立划分；逐次一票）" % a.reps)
    print("=" * 100)
    print("  %-16s %10s %10s %10s %22s" % ("指标", "均值", "标准差", "SE", "95% CI(正态近似)"))
    print("  " + "-" * 74)
    lam = {"cal_cov": "校准集(简单查询)", "none_cov": "简单→简单(对照)",
           "frozen_cov": "★简单→复杂(冻结)", "adapt_cov": "适应后(复杂)",
           "oracle_cov": "Oracle(复杂)"}
    for k in keys[:-1]:
        x = np.asarray(acc[k], float)
        x = x[np.isfinite(x)]
        se = x.std(ddof=1) / np.sqrt(len(x))
        print("  %-16s %9.1f%% %9.2fpp %9.2fpp   [%6.1f%%, %6.1f%%]"
              % (lam[k], 100 * x.mean(), 100 * x.std(ddof=1), 100 * se,
                 100 * (x.mean() - 1.96 * se), 100 * (x.mean() + 1.96 * se)))
    w = np.asarray(acc["frozen_width"], float)
    print("  %-16s %9.2f %9.2f %9.2f" % ("冻结区间宽度(倍)", w.mean(), w.std(ddof=1),
                                         w.std(ddof=1) / np.sqrt(len(w))))

    fz = np.asarray(acc["frozen_cov"], float)
    print("\n  ★ 冻结覆盖的单次读数范围: [%.1f%%, %.1f%%]  极差 %.1fpp"
          % (100 * np.nanmin(fz), 100 * np.nanmax(fz), 100 * (np.nanmax(fz) - np.nanmin(fz))))
    print("     -> 这就是 50.8%% 与 45.8%% 那 5pp 差异的来源：单次读数本身就有这个量级的抖动。")

    print("\n" + "=" * 100)
    print("Q3  混淆拆分：训练集混入 %d%% 复杂查询，校准集始终只用简单查询" % a.mix)
    print("=" * 100)
    n_cp = int(round(a.n_train * a.mix / 100.0))
    print("  训练集构成: 简单 %d + 复杂 %d" % (a.n_train - n_cp, n_cp))
    print("  冻结覆盖 = %.1f%% ± %.2f   ->  %s"
          % (100 * np.nanmean(fz), 100 * np.nanstd(fz, ddof=1) / np.sqrt(len(fz)),
             "仍崩（纯校准转移）" if np.nanmean(fz) < 0.80 else
             ("恢复正常（模型能力问题）" if np.nanmean(fz) > 0.87 else "部分恢复（两成分混合）")))

    os.makedirs(OUT, exist_ok=True)
    tag = "mix%d_test%d_rep%d" % (a.mix, a.test_size, a.reps)
    np.savez(os.path.join(OUT, "workloaddrift_ci_%s.npz" % tag), **acc)
    print("\n  已存 results_repro/workloaddrift_ci_%s.npz" % tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
