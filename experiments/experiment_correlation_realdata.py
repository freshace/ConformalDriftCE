# -*- coding: utf-8 -*-
"""L3 真实数据级相关性漂移实验 (experiment_correlation_realdata.py)

动机：
  现有 experiment_correlation.py 是【标签级合成注入】(inject_correlation_drift 把多列查询真基数乘因子)，
  快、只验机制，但审稿人会问"漂移是合成的，从没在真实漂移库上验证过"。
  本脚本做【数据级】：真的造一个"列间相关结构可控"的数据，漂移 = 真的改列间相关矩阵
  (边际不变、联合选择性变)，然后【在改动后的数据上重新执行查询、重算真基数】。
  与标签级互补：标签级验机制 + 本脚本给真实数据级的可信度，堵住"合成刷数字"质疑。

机制(和主实验同一套外挂 model.py + conformal.py)：
  - 底模 pg_est = N * prod(边际选择性)  (=Postgres/ALECE 的 1D 直方图/边际估计, 只存边际不存相关)
  - 相关性漂移：列间相关矩阵 R 变(组内 rho 变)，边际不变 => pg_est 不变；
    但联合选择性变 => 真基数在重算后变了 => LGBM 残差修正器(老)失效 => 覆盖掉。
  - 修复：用漂移后的新标签重训 LGBM + 重校准，覆盖回升。
  用 Gaussian copula 生成数据：边际 uniform(经验秩变换)，相关结构由 R 精确控制。

生成器要点：
  - GROUPS 把列分成"组内相关、跨组独立"的块；相关性漂移=改组内 rho。
  - 查询生成带 same_group_bias：一部分多列查询偏向"同组" (相关性敏感)，一部分随机(含跨组)，
    这样"相关性敏感子集"被充分填充，漂移效果可见。
  - 报告拆分"整体 + 相关性敏感子集(选中的列含>=1 同组对)"，敏感子集漂移掉得最狠。

用法：
  python drift_ce/experiments/experiment_correlation_realdata.py
输出：
  drift_ce/results_repro/experiment_correlation_realdata.txt
  drift_ce/results_corr_realdata.npy
  列: [rho, 基础覆盖, 漂移覆盖(冻结), 修复后覆盖, 漂移覆盖-敏感子集(冻结), 修复后覆盖-敏感子集]
"""
import os, sys
from collections import Counter
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model, conformal

# ------------------------------------------------------------------ 参数
SEED = 42
N_ROWS = 200_000            # 合成表行数
D = 8                       # 列数
GROUPS = [[0, 1], [2, 3], [4, 5], [6, 7]]   # 组内相关、跨组独立(相关性敏感结构)
RHO_BASE = 0.30             # 基础组内相关
DRIFT_RHOS = [0.30, 0.45, 0.60, 0.75, 0.90]  # 漂移后组内相关(驱动因子)
N_QUERIES = 9000            # 生成的候选查询数(构造完按条件筛选)
N_TRAIN, N_CAL, N_TEST = 1500, 1500, 3000   # 基础查询 train/cal/test
N_ADAPT_TRAIN, N_ADAPT_CAL = 500, 400        # 漂移后重训/重校准标签
ALPHA = 0.1                 # 目标覆盖 90%
MIN_TRUE = 8               # 真基数下限(保持 log 稳定)
SAME_GROUP_BIAS = 0.5       # 多列查询里"偏同组"的比例(相关性敏感子集)
LGBM_PARAMS = dict(n_estimators=400, num_leaves=63, learning_rate=0.06,
                   random_state=SEED, verbose=-1, n_jobs=1)

# ------------------------------------------------------------------ 数据生成
def corr_block_matrix(D, groups, rho_in, rho_out=0.0):
    """块相关矩阵：每组内 off-diag=rho_in，组间=rho_out，对角=1。(正相关块结构, PD)"""
    R = np.full((D, D), rho_out)
    for g in groups:
        R[np.ix_(g, g)] = rho_in
    np.fill_diagonal(R, 1.0)
    return R

def gaussian_copula(R, n, seed):
    """z~N(0,R) -> 各列做经验秩变换成 uniform[0,1] 边际。相关性由 R 支配(经验 copula)。"""
    rng = np.random.RandomState(seed)
    L = np.linalg.cholesky(R)
    z = (L @ rng.standard_normal((R.shape[0], n))).T          # (n, D)
    u = np.empty_like(z)
    for j in range(D):
        order = np.argsort(z[:, j])
        ranks = np.empty(n, dtype=np.float64); ranks[order] = np.arange(1, n + 1)
        u[:, j] = ranks / (n + 1)                              # uniform in (0,1)
    return u

def _group_of(col, groups):
    for gi, g in enumerate(groups):
        if col in g: return gi
    return -1

def gen_queries(n, D, groups, seed, min_width=0.03, max_width=0.4, max_pred=3, same_group_bias=0.5):
    """生成范围查询。多列查询有 same_group_bias 概率"偏同组"(相关性敏感)，否则随机选列(含跨组)。"""
    rng = np.random.RandomState(seed)
    qs = []
    for _ in range(n):
        k = int(rng.randint(1, max_pred + 1))
        if k >= 2 and rng.rand() < same_group_bias:
            # 从同一个组内取列(相关性敏感)
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
    onehot = np.zeros(D, dtype=np.float32); onehot[list(cols)] = 1.0
    onehot = onehot / max(len(cols), 1)
    logw = np.zeros(D, dtype=np.float32)
    mid  = np.zeros(D, dtype=np.float32)
    for i, c in enumerate(cols):
        logw[c] = np.log(np.clip(hi[i] - lo[i], 1e-9, None))
        mid[c]  = 0.5 * (lo[i] + hi[i])
    ncol = float(len(cols))
    return np.concatenate([onehot, logw, mid,
                           np.array([np.log(pg), ncol, float(n_group_pairs(cols, groups))], dtype=np.float32)])

# ------------------------------------------------------------------ 主流程
def main():
    R_base = corr_block_matrix(D, GROUPS, RHO_BASE)
    u_base = gaussian_copula(R_base, N_ROWS, SEED)
    queries = gen_queries(N_QUERIES, D, GROUPS, SEED + 1, same_group_bias=SAME_GROUP_BIAS)

    keep_idx, pg_l, true_l, feat_l = [], [], [], []
    for i, (cols, lo, hi) in enumerate(queries):
        w = hi - lo
        p = N_ROWS * float(np.prod(w))
        t = count_rows(u_base, cols, lo, hi)
        if t < MIN_TRUE or t > 0.5 * N_ROWS or p < MIN_TRUE:
            continue
        keep_idx.append(i)
        pg_l.append(p); true_l.append(float(t))
        feat_l.append(query_feature(cols, lo, hi, D, N_ROWS, p, GROUPS))
    pg   = np.array(pg_l, dtype=np.float64); true_base = np.array(true_l, dtype=np.float64)
    feat = np.array(feat_l, dtype=np.float32)
    sens_mask_full = np.array([n_group_pairs(queries[i][0], GROUPS) >= 1 for i in keep_idx], dtype=bool)
    print(f"已构造 {len(true_base)} 条查询 (表 {N_ROWS}x{D}，组内相关 rho={RHO_BASE}；相关性敏感子集 {sens_mask_full.sum()})")

    idx = np.random.RandomState(SEED + 2).permutation(len(true_base))
    tr, cal, te = idx[:N_TRAIN], idx[N_TRAIN:N_TRAIN + N_CAL], idx[N_TRAIN + N_CAL:N_TRAIN + N_CAL + N_TEST]

    m_old = model.fit_corrector(feat[tr], pg[tr], true_base[tr], LGBM_PARAMS)
    q_old = conformal.calibrate_q(conformal.score(true_base[cal], pg[cal], model.predict_corrector(m_old, feat[cal])), ALPHA)
    cov_base = conformal.coverage(conformal.score(true_base[te], pg[te], model.predict_corrector(m_old, feat[te])), q_old)
    cast_base_sens = conformal.coverage(conformal.score(true_base[te][sens_mask_full[te]], pg[te][sens_mask_full[te]],
                                                       model.predict_corrector(m_old, feat[te][sens_mask_full[te]])), q_old)
    print(f"[基础(同分布)] 整体覆盖={cov_base:.1%} (区间 {np.exp(q_old):.1f}x)；敏感子集={cast_base_sens:.1%}")

    sens_test = sens_mask_full[te]
    rows = []
    header = f"{'rho':>5} | {'基础':>7} | {'漂移(冻结)':>10} | {'修复后':>8} | {'敏感漂移(冻结)':>14} | {'敏感修复后':>10}"
    out_lines = [f"[L3 真实数据级相关性漂移] 表 {N_ROWS}x{D}，组内相关 rho_base={RHO_BASE}，漂移=组内 rho 变(边际不变)",
                 f"基础(同分布): 整体覆盖={cov_base:.1%} (区间 {np.exp(q_old):.1f}x)，相关性敏感子集覆盖={cast_base_sens:.1%}",
                 "", header, "-" * 80]
    print(f"\n{header}"); print("-" * 80)
    for rho in DRIFT_RHOS:
        R_d = corr_block_matrix(D, GROUPS, rho)
        u_d = gaussian_copula(R_d, N_ROWS, SEED + 3)
        true_d_all = np.array([max(count_rows(u_d, q[0], q[1], q[2]), 1.0) for q in queries], dtype=np.float64)
        true_d = true_d_all[keep_idx]
        assert len(true_d) == len(true_base)
        # 冻结：老修正器 + 老q
        cov_drift = conformal.coverage(conformal.score(true_d[te], pg[te], model.predict_corrector(m_old, feat[te])), q_old)
        cov_sens_frozen = conformal.coverage(conformal.score(true_d[te][sens_test], pg[te][sens_test],
                                                             model.predict_corrector(m_old, feat[te][sens_test])), q_old)
        # 修复：漂移后新标签重训 LGBM + 重校准
        perm = np.random.RandomState(SEED + 10 + int(rho * 100)).permutation(len(te))
        a_tr  = te[perm[:N_ADAPT_TRAIN]]
        a_cal = te[perm[N_ADAPT_TRAIN:N_ADAPT_TRAIN + N_ADAPT_CAL]]
        a_ev  = te[perm[N_ADAPT_TRAIN + N_ADAPT_CAL:]]
        m_new = model.fit_corrector(feat[a_tr], pg[a_tr], true_d[a_tr], LGBM_PARAMS)
        q_new = conformal.calibrate_q(conformal.score(true_d[a_cal], pg[a_cal], model.predict_corrector(m_new, feat[a_cal])), ALPHA)
        cov_repair = conformal.coverage(conformal.score(true_d[a_ev], pg[a_ev], model.predict_corrector(m_new, feat[a_ev])), q_new)
        a_ev_sens = a_ev[sens_mask_full[a_ev]]
        cov_repair_sens = conformal.coverage(conformal.score(true_d[a_ev_sens], pg[a_ev_sens],
                                                             model.predict_corrector(m_new, feat[a_ev_sens])), q_new)
        rows.append((rho, cov_base, cov_drift, cov_repair, cov_sens_frozen, cov_repair_sens))
        line = f"{rho:>5.2f} | {cov_base:>7.1%} | {cov_drift:>10.1%} | {cov_repair:>8.1%} | {cov_sens_frozen:>14.1%} | {cov_repair_sens:>10.1%}"
        print(line); out_lines.append(line)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_repro")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "experiment_correlation_realdata.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_corr_realdata.npy"), np.array(rows))
    print(f"\n[L3 真实数据级相关性漂移] 完成。")
    print("npy 列: [rho, 基础覆盖, 漂移覆盖(冻结), 修复后覆盖, 敏感漂移(冻结), 敏感修复后]")

if __name__ == "__main__":
    main()
