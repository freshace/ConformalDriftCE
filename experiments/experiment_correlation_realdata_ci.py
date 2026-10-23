# -*- coding: utf-8 -*-
r"""P0：数据级相关性漂移的【重复切分 + 置信区间】。

为什么要做
----------
`experiment_correlation_realdata.py` 是数据漂移线上唯一"真改数据、边际严格不变、
改完重算真值"的证据（敏感子集 90.9%->5.6%），但它**只有一次随机实现**：
  * 查询集固定用 SEED+1 生成一次
  * 划分固定用 SEED+2
  * 漂移数据固定用 SEED+3
  * 修复划分固定用 SEED+10+rho*100
同项目已实测：workload drift 的单次读数极差可达 28.6pp。所以本脚本做 R 次**独立重复**，
每次同时重抽【数据实现】与【查询/划分】，给出 mean±SE 与单次读数范围。

与主实验同一套外挂（model.py + conformal.py），参数与原脚本完全一致，仅把随机源换成 R 个种子。

闸门 G0：rho = 0.30（= 基础相关，即"无漂移"）时整体冻结覆盖必须落在 90% ± 抽样误差内。

用法：python experiments/experiment_correlation_realdata_ci.py [--reps 50]
输出：results_repro/experiment_correlation_realdata_ci.txt
      results_repro/correlation_realdata_ci.npz
"""
import argparse
import os
import sys
import time
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import conformal  # noqa: E402
import model  # noqa: E402

# ---- 与原脚本完全一致的参数 ----
N_ROWS = 200_000
D = 8
GROUPS = [[0, 1], [2, 3], [4, 5], [6, 7]]
RHO_BASE = 0.30
DRIFT_RHOS = [0.30, 0.45, 0.60, 0.75, 0.90]
N_QUERIES = 9000
N_TRAIN, N_CAL, N_TEST = 1500, 1500, 3000
N_ADAPT_TRAIN, N_ADAPT_CAL = 500, 400
ALPHA = 0.1
MIN_TRUE = 8
SAME_GROUP_BIAS = 0.5
LGBM_PARAMS = dict(n_estimators=400, num_leaves=63, learning_rate=0.06,
                   random_state=42, verbose=-1, n_jobs=1)
SEED_BASE = 42


def corr_block_matrix(D, groups, rho_in, rho_out=0.0):
    R = np.full((D, D), rho_out)
    for g in groups:
        R[np.ix_(g, g)] = rho_in
    np.fill_diagonal(R, 1.0)
    return R


def gaussian_copula(R, n, seed):
    rng = np.random.RandomState(seed)
    L = np.linalg.cholesky(R)
    z = (L @ rng.standard_normal((R.shape[0], n))).T
    u = np.empty_like(z)
    for j in range(D):
        order = np.argsort(z[:, j])
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1)
        u[:, j] = ranks / (n + 1)
    return u


def _group_of(col, groups):
    for gi, g in enumerate(groups):
        if col in g:
            return gi
    return -1


def gen_queries(n, D, groups, seed, min_width=0.03, max_width=0.4, max_pred=3, same_group_bias=0.5):
    rng = np.random.RandomState(seed)
    qs = []
    for _ in range(n):
        k = int(rng.randint(1, max_pred + 1))
        if k >= 2 and rng.rand() < same_group_bias:
            g = groups[int(rng.randint(0, len(groups)))]
            cols = rng.choice(g, size=min(k, len(g)), replace=False)
        else:
            cols = rng.choice(D, size=k, replace=False)
        w = rng.uniform(min_width, max_width, size=len(cols))
        lo = rng.uniform(0.0, 1.0 - w)
        qs.append((cols, lo, lo + w))
    return qs


def count_rows(u, cols, lo, hi):
    mask = np.ones(u.shape[0], dtype=bool)
    for i, c in enumerate(cols):
        mask &= (u[:, c] >= lo[i]) & (u[:, c] <= hi[i])
    return int(mask.sum())


def n_group_pairs(cols, groups):
    gs = [_group_of(int(c), groups) for c in cols]
    c = Counter(gs)
    return int(sum(v * (v - 1) // 2 for v in c.values()))


def query_feature(cols, lo, hi, D, N, pg, groups):
    onehot = np.zeros(D, dtype=np.float32)
    onehot[list(cols)] = 1.0
    onehot = onehot / max(len(cols), 1)
    logw = np.zeros(D, dtype=np.float32)
    mid = np.zeros(D, dtype=np.float32)
    for i, c in enumerate(cols):
        logw[c] = np.log(np.clip(hi[i] - lo[i], 1e-9, None))
        mid[c] = 0.5 * (lo[i] + hi[i])
    ncol = float(len(cols))
    return np.concatenate([onehot, logw, mid,
                           np.array([np.log(pg), ncol, float(n_group_pairs(cols, groups))],
                                    dtype=np.float32)])


def one_replicate(seed):
    """一次独立实现：重抽数据 + 重抽查询 + 重抽划分。返回 {rho: {...}}。"""
    u_base = gaussian_copula(corr_block_matrix(D, GROUPS, RHO_BASE), N_ROWS, seed)
    queries = gen_queries(N_QUERIES, D, GROUPS, seed + 1, same_group_bias=SAME_GROUP_BIAS)

    keep_idx, pg_l, true_l, feat_l = [], [], [], []
    for i, (cols, lo, hi) in enumerate(queries):
        w = hi - lo
        p = N_ROWS * float(np.prod(w))
        t = count_rows(u_base, cols, lo, hi)
        if t < MIN_TRUE or t > 0.5 * N_ROWS or p < MIN_TRUE:
            continue
        keep_idx.append(i)
        pg_l.append(p)
        true_l.append(float(t))
        feat_l.append(query_feature(cols, lo, hi, D, N_ROWS, p, GROUPS))
    pg = np.array(pg_l, dtype=np.float64)
    true_base = np.array(true_l, dtype=np.float64)
    feat = np.array(feat_l, dtype=np.float32)
    sens_full = np.array([n_group_pairs(queries[i][0], GROUPS) >= 1 for i in keep_idx], dtype=bool)

    idx = np.random.RandomState(seed + 2).permutation(len(true_base))
    tr = idx[:N_TRAIN]
    cal = idx[N_TRAIN:N_TRAIN + N_CAL]
    te = idx[N_TRAIN + N_CAL:N_TRAIN + N_CAL + N_TEST]

    m_old = model.fit_corrector(feat[tr], pg[tr], true_base[tr], LGBM_PARAMS)
    q_old = conformal.calibrate_q(
        conformal.score(true_base[cal], pg[cal], model.predict_corrector(m_old, feat[cal])), ALPHA)

    res = {"n_keep": len(true_base), "n_sens": int(sens_full.sum())}
    res["base_cov"] = conformal.coverage(
        conformal.score(true_base[te], pg[te], model.predict_corrector(m_old, feat[te])), q_old)
    res["base_cov_sens"] = conformal.coverage(
        conformal.score(true_base[te][sens_full[te]], pg[te][sens_full[te]],
                        model.predict_corrector(m_old, feat[te][sens_full[te]])), q_old)
    res["width_base"] = float(np.exp(q_old))

    for rho in DRIFT_RHOS:
        u_d = gaussian_copula(corr_block_matrix(D, GROUPS, rho), N_ROWS, seed + 3)
        true_d = np.array([max(count_rows(u_d, q[0], q[1], q[2]), 1.0) for q in queries],
                          dtype=np.float64)[keep_idx]
        sens_te = sens_full[te]
        cov_f = conformal.coverage(
            conformal.score(true_d[te], pg[te], model.predict_corrector(m_old, feat[te])), q_old)
        cov_f_s = conformal.coverage(
            conformal.score(true_d[te][sens_te], pg[te][sens_te],
                            model.predict_corrector(m_old, feat[te][sens_te])), q_old)

        perm = np.random.RandomState(seed + 10 + int(rho * 100)).permutation(len(te))
        a_tr = te[perm[:N_ADAPT_TRAIN]]
        a_cal = te[perm[N_ADAPT_TRAIN:N_ADAPT_TRAIN + N_ADAPT_CAL]]
        a_ev = te[perm[N_ADAPT_TRAIN + N_ADAPT_CAL:]]
        m_new = model.fit_corrector(feat[a_tr], pg[a_tr], true_d[a_tr], LGBM_PARAMS)
        q_new = conformal.calibrate_q(
            conformal.score(true_d[a_cal], pg[a_cal], model.predict_corrector(m_new, feat[a_cal])), ALPHA)
        cov_r = conformal.coverage(
            conformal.score(true_d[a_ev], pg[a_ev], model.predict_corrector(m_new, feat[a_ev])), q_new)
        ae_s = sens_full[a_ev]
        cov_r_s = conformal.coverage(
            conformal.score(true_d[a_ev][ae_s], pg[a_ev][ae_s],
                            model.predict_corrector(m_new, feat[a_ev][ae_s])), q_new)
        res[rho] = dict(cov_f=cov_f, cov_f_sens=cov_f_s, cov_r=cov_r, cov_r_sens=cov_r_s,
                        width_rep=float(np.exp(q_new)))
    return res


def ms(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return x.mean(), se, x.min(), x.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, "results_repro")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 104)
    print("P0  数据级相关性漂移：重复切分 + CI   reps=%d" % a.reps)
    print("=" * 104)
    print("  表 %dx%d，组内相关 rho_base=%.2f；每次重复【同时重抽数据实现与查询/划分】"
          % (N_ROWS, D, RHO_BASE))

    recs = []
    t0 = time.time()
    for r in range(a.reps):
        recs.append(one_replicate(SEED_BASE + 1000 * r))
        if (r + 1) % max(1, a.reps // 10) == 0:
            print("    %d/%d  (%.0fs)" % (r + 1, a.reps, time.time() - t0))

    n_keep = [x["n_keep"] for x in recs]
    print("\n  每次构造出的查询数：%.0f ± %.0f（敏感子集 %.0f ± %.0f）"
          % (np.mean(n_keep), np.std(n_keep, ddof=1), np.mean([x["n_sens"] for x in recs]),
             np.std([x["n_sens"] for x in recs], ddof=1)))

    lines = []
    def P(s):
        print(s); lines.append(s)

    P("\n" + "=" * 104)
    P("闸门 G0：rho=0.30（= 基础相关，即无漂移）时整体冻结覆盖须落在 90% ± 抽样误差内")
    P("=" * 104)
    m, se, lo, hi = ms([x[0.30]["cov_f"] for x in recs])
    P("  rho=0.30 整体冻结覆盖 = %.1f%% ± %.2fpp  95%%CI[%.1f%%, %.1f%%]  %s"
      % (100 * m, 100 * se, 100 * (m - 1.96 * se), 100 * (m + 1.96 * se),
         "✓ 通过" if lo > 0.82 and hi < 0.98 else "!! 未通过"))
    m0, se0, lo0, hi0 = ms([x["base_cov"] for x in recs])
    P("  （同分布基础覆盖   = %.1f%% ± %.2fpp）" % (100 * m0, 100 * se0))

    P("\n" + "=" * 104)
    P("表 1  覆盖率（%d 次独立重复；mean ± SE，方括号为单次读数范围）" % a.reps)
    P("=" * 104)
    P("  %-6s | %-24s | %-24s | %-24s" % ("rho", "整体 冻结", "整体 修复后", "敏感子集 冻结"))
    P("  " + "-" * 92)
    for rho in DRIFT_RHOS:
        cf = [x[rho]["cov_f"] for x in recs]
        cr = [x[rho]["cov_r"] for x in recs]
        cs = [x[rho]["cov_f_sens"] for x in recs]
        f = lambda v: "%.1f±%.1f [%.1f,%.1f]" % (100 * ms(v)[0], 100 * ms(v)[1],
                                                 100 * ms(v)[2], 100 * ms(v)[3])
        P("  %-6.2f | %-24s | %-24s | %-24s" % (rho, f(cf), f(cr), f(cs)))

    P("\n" + "=" * 104)
    P("表 2  敏感子集：修复效果 + 区间宽度")
    P("=" * 104)
    P("  %-6s | %-22s | %-22s | %-18s" % ("rho", "敏感 修复后", "敏感 修复−冻结", "半宽(倍) 基础→修复"))
    P("  " + "-" * 92)
    for rho in DRIFT_RHOS:
        cs = [x[rho]["cov_f_sens"] for x in recs]
        crs = [x[rho]["cov_r_sens"] for x in recs]
        d = np.array(crs) - np.array(cs)
        P("  %-6.2f | %-22s | %+21.1fpp | %.2f → %.2f"
          % (rho, "%.1f±%.1f" % (100 * np.mean(crs), 100 * ms(crs)[1]),
             100 * d.mean(), np.mean([x["width_base"] for x in recs]),
             np.mean([x[rho]["width_rep"] for x in recs])))

    fz = np.array([x[0.90]["cov_f"] for x in recs])
    P("\n" + "=" * 104)
    P("★ 不确定度（这是本次补 CI 的核心产出）")
    P("=" * 104)
    P("  rho=0.90 整体冻结覆盖的单次读数范围 : [%.1f%%, %.1f%%]  极差 %.1fpp"
      % (100 * fz.min(), 100 * fz.max(), 100 * (fz.max() - fz.min())))
    sz = np.array([x[0.90]["cov_f_sens"] for x in recs])
    P("  rho=0.90 敏感子集冻结的单次读数范围 : [%.1f%%, %.1f%%]  极差 %.1fpp"
      % (100 * sz.min(), 100 * sz.max(), 100 * (sz.max() - sz.min())))
    P("  ⇒ 原脚本报出的是这两条分布里的【一次抽样】，论文中必须标注为 mean±SE。")

    with open(os.path.join(out_dir, "experiment_correlation_realdata_ci.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    np.savez(os.path.join(out_dir, "correlation_realdata_ci.npz"),
             reps=np.array([K for K in DRIFT_RHOS]),
             cov_f=np.array([[x[rho]["cov_f"] for rho in DRIFT_RHOS] for x in recs]),
             cov_f_sens=np.array([[x[rho]["cov_f_sens"] for rho in DRIFT_RHOS] for x in recs]),
             cov_r=np.array([[x[rho]["cov_r"] for rho in DRIFT_RHOS] for x in recs]),
             cov_r_sens=np.array([[x[rho]["cov_r_sens"] for rho in DRIFT_RHOS] for x in recs]),
             base_cov=np.array([x["base_cov"] for x in recs]),
             n_keep=np.array(n_keep))
    print("\n  已存 results_repro/experiment_correlation_realdata_ci.txt 与 correlation_realdata_ci.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
