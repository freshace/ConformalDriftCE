# -*- coding: utf-8 -*-
"""pool_adapter —— 把云上带回来的【真实数据漂移池】对齐到本地 622 维特征，并出首轮诊断。

对齐依据（已本地验证）
    features40_s3a.pkl 的条目 i  ≡  finetune/stats/workloads.sql 的非空行 i
    （50000/50000 逐条比对，0 不一致）
池内 meta.idx_semantics 记录同一语义，故直接按 idx 取特征，【无需 SQL 匹配】。

用法
----
  python pool_adapter.py --pool pool_stats.pkl                 # 真实池
  python pool_adapter.py --selftest                            # 无需云端，验证适配器可用

输出（首轮诊断，决定后续策略实验怎么做）
  1. 各状态规模与 ok 数
  2. 真值变化分布 log(true_state/true_baseline)：均值/标准差/分位 —— 是否"非常数"
  3. 各状态的**最小诚实宽度 q*(alpha)**（用未修正分数 |log(true/pg_est)|）
     —— 直接回答"真漂移要把区间撑宽多少"
  4. 边际/基模不变性：pg_est 的变化幅度（保边际漂移下应≈0）
"""
import argparse, os, pickle, sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config, data, metrics  # noqa: E402

DEF_PKL = config.DATA_PATH


def load_aligned(pool_path, pkl_path=DEF_PKL, verbose=True):
    with open(pool_path, "rb") as f:
        pool = pickle.load(f)
    base_all, true_all, feat_all, nfc_all = data.load_stats(pkl_path)
    idx = np.asarray(pool["idx"], dtype=int)
    if idx.max() >= len(true_all):
        raise ValueError(f"池中 idx 最大 {idx.max()} 超过本地特征条数 {len(true_all)}")
    out = dict(meta=pool["meta"], idx=idx,
               feat=feat_all[idx], nfc=nfc_all[idx],
               true_local=true_all[idx], base_local=base_all[idx],
               states={})
    for st, v in pool["states"].items():
        ok = np.asarray(v["ok"], dtype=bool)
        out["states"][st] = dict(
            ok=ok,
            true=np.asarray([np.nan if x is None else x for x in v["true"]], dtype=float),
            pg=np.asarray([np.nan if x is None else x for x in v["pg_est"]], dtype=float),
            secs=v.get("secs"))
    if verbose:
        print(f"[pool] {pool_path}: {len(idx)} 条查询, {len(out['states'])} 个状态 "
              f"({', '.join(out['states'])})")
    return out


def read_workload_true(path, limit=None):
    """读 workload 文件第 2 列（原始环境上的【真值】）。

    文件格式：`sql || true || pg_est || tag`，非空行按 0-based 编号，
    与 features40_s3a.pkl 条目一一对应（已本地验证 50000/50000）。
    返回 (idx_list, true_list)，true 解析失败的记为 None。
    """
    out_i, out_t = [], []
    k = 0
    with open(os.path.expanduser(path), encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("--"):
                continue
            parts = line.split("\t") if "\t" in line else line.split("||")
            sql = parts[0].strip().rstrip(";").strip() if parts else ""
            if not sql:
                continue
            t = None
            if len(parts) >= 2:
                try:
                    t = float(parts[1].strip())
                except Exception:
                    t = None
            out_i.append(k)
            out_t.append(t)
            k += 1
            if limit and k >= limit:
                break
    return out_i, out_t


def verify_baseline(al, workload_path, max_show=5):
    """把池中 baseline 状态的真值与 workload 文件记录的真值逐条比对。

    为什么这是最关键的校验
    ----------------------
    workload 文件第 2 列是 PRICE 项目在**原始环境**上跑出来的真值。如果我们新重建的
    PostgreSQL（本地或云端）与之逐条一致，就同时证明了三件事：
      1. CSV 加载正确（NULL 语义、行数、类型都没错）；
      2. 查询语义一致（同一条 SQL 在我们的库里算的是同一个东西）；
      3. 行号对齐正确（池的 idx ↔ pkl 条目 ↔ workload 行）。
    任何"漂移结论"都以这三点为前提；这里不过，后面全部无意义。

    返回 (n_cmp, n_equal, n_diff, 是否通过)。
    """
    base_st = "baseline" if "baseline" in al["states"] else list(al["states"])[0]
    if base_st != "baseline":
        print(f"[verify_baseline] 池中没有 baseline 状态（只有 {list(al['states'])}），跳过")
        return 0, 0, 0, None
    wi, wt = read_workload_true(workload_path)
    idx = al["idx"]
    b = al["states"]["baseline"]
    n_cmp = n_eq = n_diff = 0
    bad = []
    for j, i in enumerate(idx):
        if i >= len(wt) or wt[i] is None:
            continue
        if not (b["ok"][j] and np.isfinite(b["true"][j])):
            continue
        n_cmp += 1
        got, exp = float(b["true"][j]), float(wt[i])
        if abs(got - exp) <= max(1.0, 1e-9 * abs(exp)):
            n_eq += 1
        else:
            n_diff += 1
            if len(bad) < max_show:
                bad.append((i, exp, got))
    pass_ = (n_cmp > 0 and n_diff == 0)
    print("\n" + "=" * 100)
    print("★环境保真度校验：池中 baseline 真值  vs  workload 文件记录真值")
    print("=" * 100)
    print(f"  workload = {workload_path}")
    print(f"  可比对 {n_cmp} 条 | 完全一致 {n_eq} | 不一致 {n_diff}")
    for i, exp, got in bad:
        print(f"    [不一致] idx={i}  期望(文件)={exp:g}  实测(池)={got:g}")
    print("  " + ("通过：新库与原始环境逐条一致 —— 加载/语义/对齐三者都被验证。"
                  if pass_ else
                  "未通过：不要用这批结果下任何结论；先查 NULL 语义、行号对齐或查询语义。"))
    return n_cmp, n_eq, n_diff, pass_


def diagnose(al, alpha=0.10):
    sts = list(al["states"])
    base_st = "baseline" if "baseline" in sts else sts[0]
    print("\n" + "=" * 100)
    print(f"真实漂移池首轮诊断 | 基准状态 = {base_st} | 目标覆盖 {1-alpha:.2f} | 指标口径见 metrics.py")
    print("=" * 100)

    b = al["states"][base_st]
    okb = b["ok"] & np.isfinite(b["true"]) & np.isfinite(b["pg"])
    fv = al["meta"].get("feature_valid", {})
    print("\n表 1  各状态规模与【特征是否可复用】")
    print("%-24s | %8s | %10s | %12s | %s" % ("状态", "ok 条数", "耗时(s)", "特征可复用", "说明"))
    print("-" * 100)
    for st in sts:
        s = al["states"][st]
        ok = s["ok"] & np.isfinite(s["true"]) & np.isfinite(s["pg"])
        fvs = "是" if fv.get(st, True) else "否(需重算)"
        note = "基准" if st == base_st else ("" if fv.get(st, True) else "仅做无需特征的诊断")
        print("%-24s | %8d | %10s | %12s | %s" % (st, ok.sum(),
                                                  ("%.0f" % s["secs"]) if s.get("secs") else "--",
                                                  fvs, note))
    print("  说明：保边际值置换不改统计量 -> 622 维特征逐位不变，可直接复用；")
    print("        尺度漂移改统计量 -> 特征须重算（本地缺 5 个语义/图文件，暂不可算）。")

    # ★ 跨状态比较【实际】使用的集合 = 各状态 ok 的交集。
    #   若它明显小于各状态各自的成功数，说明超时/报错把不同状态的查询集切得不一样，
    #   此时只看"各状态条数"会给人虚假的充足感，必须把这个共同集显式报出来。
    ok_all = None
    per_ok = []
    for st in sts:
        s = al["states"][st]
        m = s["ok"] & np.isfinite(s["true"]) & np.isfinite(s["pg"])
        per_ok.append(int(m.sum()))
        ok_all = m if ok_all is None else (ok_all & m)
    n_all = int(ok_all.sum())
    print("\n★ 共同查询集（所有状态都成功）: %d 条 | 各状态各自成功数: %s" % (n_all, per_ok))
    if n_all < max(per_ok):
        print("  注意：共同集 < 最大单状态数 -> 有查询只在部分状态被超时排除。")
        print("        跨状态的 q* 比较【只能】建立在这 %d 条上；这一句必须写进论文的样本说明，" % n_all)
        print("        否则会高估有效样本量（超时排除偏向'基数小=便宜'的查询，是一种选择偏差）。")
    else:
        print("  各状态查询集一致，可直接做配对比较。")

    print("\n表 2  ★真值变化分布  log(true_state / true_baseline)  —— 现有注入是常数(std=0)，真漂移应 std>0")
    print("  注：真值可能为 0（漂移后结果为空），log(0)=-inf 会把均值/标准差污染成 nan。")
    print("      故先把真值 clip 到 >=1 再取对数；那部分查询单独用最后两列计数，不混进分布统计。")
    print("%-24s | %8s | %8s | %8s | %8s | %8s | %8s | %7s | %10s" %
          ("状态", "变化条数", "均值", "标准差", "q5", "q95", "变0条数", "由0变正", "pg_est变化"))
    print("-" * 122)
    for st in sts:
        if st == base_st:
            continue
        s = al["states"][st]
        m = b["ok"] & s["ok"] & np.isfinite(b["true"]) & np.isfinite(s["true"])
        bt, stv = b["true"][m], s["true"][m]
        # 真值为 0 的单独计数：它们不是"小变化"，而是"结果变空"这种质变
        to_zero = int(((bt > 0) & (stv <= 0)).sum())
        from_zero = int(((bt <= 0) & (stv > 0)).sum())
        lr = np.log(np.clip(stv, 1, None)) - np.log(np.clip(bt, 1, None))
        ch = np.abs(lr) > 1e-9
        mp = b["ok"] & s["ok"] & np.isfinite(b["pg"]) & np.isfinite(s["pg"])
        lrp = np.abs(np.log(np.clip(s["pg"][mp], 1, None)) - np.log(np.clip(b["pg"][mp], 1, None)))
        print("%-24s | %8d | %+8.4f | %8.4f | %+8.3f | %+8.3f | %8d | %7d | %9.4f(中位)" %
              (st, int(ch.sum()), lr[ch].mean() if ch.any() else np.nan,
               lr[ch].std() if ch.any() else np.nan,
               np.percentile(lr[ch], 5) if ch.any() else np.nan,
               np.percentile(lr[ch], 95) if ch.any() else np.nan,
               to_zero, from_zero, float(np.median(lrp))))
    print("  读法：标准差>0 说明修正量随查询变化（消除循环论证）；pg_est 变化≈0 说明基模对边际漂移不变。")

    print("\n表 3  ★各状态的最小诚实宽度 q*(alpha)（未修正分数 |log(true/pg_est)|）")
    print("%-24s | %10s | %12s | %12s" % ("状态", "q*(alpha)", "exp(q*)宽度", "相对基准"))
    print("-" * 100)
    qb = None
    for st in sts:
        s = al["states"][st]
        ok = s["ok"] & np.isfinite(s["true"]) & np.isfinite(s["pg"])
        sc = np.abs(np.log(np.clip(s["true"][ok], 1, None)) - np.log(np.clip(s["pg"][ok], 1, None)))
        q = metrics.q_star(sc, alpha)
        if st == base_st:
            qb = q
        ratio = float(np.exp(q - qb)) if qb else float("nan")
        print("%-24s | %10.3f | %11.2fx | %11.2fx" % (st, q, float(np.exp(q)), ratio))
    print("  读法：真漂移把'诚实所需宽度'撑大的倍数，就是重校准必须付的价格。")

    print("\n附：逐状态自查（口径保证）")
    print("  - 表 2 的 pg_est 变化应≈0（保边际漂移）或 ≈scale（尺度漂移）")
    print("  - 表 3 的 q* 应由真实分数算得，不得由旧校准集算得")


def selftest(alpha=0.10):
    """无需云端：用本地 pkl 伪造一个池（基准 + 合成扰动），验证适配器与诊断链路可用。"""
    print("[selftest] 用本地 pkl 伪造 2 个状态的池（基准 + 对 nfc>=2 乘 exp(2) 的合成扰动）")
    base_all, true_all, feat_all, nfc_all = data.load_stats(DEF_PKL)
    n = 3000
    idx = np.arange(n)
    rng = np.random.RandomState(0)
    shift = np.where(nfc_all[idx] >= 2, np.exp(2.0), 1.0)
    fake = {"meta": dict(db="stats", configs=["baseline", "fake_perm"], idx_semantics="selftest",
                         feature_valid={"baseline": True, "fake_perm": True}),
            "idx": idx.tolist(),
            "states": {
                "baseline": dict(true=true_all[idx].tolist(), pg_est=base_all[idx].tolist(),
                                 ok=[True] * n, secs=0),
                "fake_perm": dict(true=(true_all[idx] * shift).tolist(),
                                  pg_est=base_all[idx].tolist(), ok=[True] * n, secs=0)}}
    p = os.path.join(HERE, "_selftest_pool.pkl")
    with open(p, "wb") as f:
        pickle.dump(fake, f)
    al = load_aligned(p, DEF_PKL)
    diagnose(al, alpha)
    os.remove(p)
    print("\n[selftest] 通过：适配器 + 诊断链路可用。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool")
    ap.add_argument("--pkl", default=DEF_PKL)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--workload", default=None,
                    help="workload 文件（sql||true||pg_est||tag）；给了就做环境保真度校验")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.pool:
        selftest(a.alpha)
        return
    al = load_aligned(a.pool, a.pkl)
    wl = a.workload or al["meta"].get("workload")
    if wl and os.path.exists(os.path.expanduser(wl)):
        verify_baseline(al, wl)
    elif wl:
        print(f"\n[verify_baseline] 跳过：找不到 workload 文件 {wl}")
    diagnose(al, a.alpha)


if __name__ == "__main__":
    main()
