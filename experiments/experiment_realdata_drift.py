# -*- coding: utf-8 -*-
"""真数据漂移 PoC：证明"修正量不是常数"，从而消除标签级注入的循环论证。

对照两个世界
------------
(A) 现有注入（drift.py）：对"过滤列数>=2"的查询把真值乘 exp(mu)。
    -> 修正量 = mu，【对所有多列查询完全相同】(std=0)，且条件变量在 622 维特征里。
(B) 真数据漂移（本脚本）：在真实 CSV 数据上置换 posts.answercount（保边际），
    再【重新执行查询】得到新真基数。
    -> 修正量 = log(true_new/true_old)，随查询【值依赖、非线性】变化。

性能设计（本机无 PG、pip 被墙）
  - sqlite3（stdlib）执行真实 join；连接键建索引；
  - 多进程并行（24 逻辑核）；
  - 每查询硬超时，避免个别病态查询拖死整体。

报告
----
1. 真值复现校验（sqlite vs workload 里的真 PG 真值）
2. 边际不变性校验（posts.answercount 直方图严格不变）
3. 修正量分布：均值/标准差/分位/未变化比例
4. 修正量能否由单特征（过滤列数 / 是否含 answercount）解释 -> R^2
5. 与"现有标签级注入"正面对比
6. 落盘真漂移池 results_realpool.npz
"""
import os, sys, re, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datasim  # noqa: E402

DB = "stats"
TABLE, COL = "posts", "answercount"
WHERE = "posttypeid=1"
FRACS = [0.0, 0.50, 1.00]
N_AC, N_OTHER = 60, 40          # 目标查询（过滤 answercount）比对照重，故总量控制在 100 条
Q_TIMEOUT = 6.0                 # 长超时：目标查询是多表 join，短超时会把它们全剔掉（抽样被速度主导）
N_WORKERS = 12
MATCH_TOL = 0.01                # 只保留 sqlite 与 PG 真值一致（<1%）的查询，保证跨状态口径一致
SEED = 42

_FILT = re.compile(r"st_[a-z]+\.([a-z_0-9]+)\s*(?:<=|>=|<>|!=|<|>|=|like)\s*('?[0-9a-zA-Z_\.\-']+)")


def filt_cols(sql):
    """该查询的【过滤列】集合：只算"列 op 字面量"谓词，排除列=列 的连接条件。"""
    cols = set()
    for col, rhs in _FILT.findall(sql.lower()):
        if rhs.startswith("st_"):
            continue
        cols.add(col)
    return cols


def nfc_of(sql):
    return len(filt_cols(sql))


def r2_from_groups(y, g, mask):
    y = y[mask]; gg = g[mask]
    if len(y) == 0:
        return np.nan
    pred = np.zeros_like(y)
    for gv in np.unique(gg):
        sel = gg == gv
        pred[sel] = y[sel].mean()
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan


def main():
    print("=" * 100)
    print("真数据漂移 PoC（%s.%s，where=%s）| %d 进程并行 | 单条超时 %.1fs"
          % (DB, TABLE, WHERE, N_WORKERS, Q_TIMEOUT))
    print("=" * 100)

    wl = datasim.load_workload(DB, "test")
    print("workload 解析: %d 条带真值查询（含 pg_est）" % len(wl), flush=True)

    rng = np.random.RandomState(SEED)
    all_sqls = [q["sql"] for q in wl]
    fc = [filt_cols(s) for s in all_sqls]
    tgt = np.array([i for i, c in enumerate(fc) if "answercount" in c], dtype=int)
    ctl = np.array([i for i, c in enumerate(fc) if "answercount" not in c], dtype=int)
    print("  全库：过滤含 answercount 的查询 %d 条（漂移直接命中的目标集）" % len(tgt), flush=True)
    rng.shuffle(tgt); rng.shuffle(ctl)
    sel = np.sort(np.concatenate([tgt[:N_AC], ctl[:N_OTHER]]))
    sqls = [all_sqls[i] for i in sel]
    wl_true = np.array([wl[i]["true"] for i in sel], dtype=float)
    print("定向抽样: 目标(含 answercount) %d + 对照 %d = %d 条"
          % (min(N_AC, len(tgt)), min(N_OTHER, len(ctl)), len(sqls)), flush=True)

    # ---------- 1) 基线（并行） ----------
    t0 = time.time()
    base_vals = np.array([v if v is not None else np.nan
                          for v in datasim.exec_pool(DB, sqls, None, N_WORKERS, timeout=Q_TIMEOUT)],
                         dtype=float)
    t_base = time.time() - t0
    ok = ~np.isnan(base_vals)
    rel_full = np.full(len(sqls), np.inf)
    rel_full[ok] = np.abs(base_vals[ok] - wl_true[ok]) / np.maximum(wl_true[ok], 1.0)
    print("\n[1] 基线（并行 %d 线程）%.0fs | 执行成功 %d/%d | |相对差| 中位 %.2e 最大 %.2e"
          % (N_WORKERS, t_base, ok.sum(), len(sqls),
             np.median(rel_full[ok]), rel_full[ok].max()), flush=True)

    ki = np.where(ok & (rel_full <= MATCH_TOL))[0]
    print("  剔除：执行失败/超时 %d 条，与 PG 真值不一致(>%.0f%%) %d 条"
          % (int((~ok).sum()), MATCH_TOL * 100, int((ok & (rel_full > MATCH_TOL)).sum())), flush=True)
    sqls = [sqls[i] for i in ki]
    true0 = base_vals[ki]
    nfc = np.array([nfc_of(s) for s in sqls], dtype=int)
    has_ac = np.array([1 if "answercount" in s.lower() else 0 for s in sqls], dtype=int)
    print("  最终查询集 %d 条 | 过滤列数分布 %s | 含 answercount %d 条"
          % (len(sqls), dict(zip(*np.unique(nfc, return_counts=True))), has_ac.sum()), flush=True)
    print("  目标集存活率: %d/%d（若偏低说明超时仍在剔除重查询）"
          % (int(has_ac.sum()), min(N_AC, len(tgt))), flush=True)

    # ---------- 2) 边际快照（主进程，仅比较边际，不跑查询） ----------
    db = datasim.StatsDB(DB, verbose=False, query_timeout_s=Q_TIMEOUT)
    db.ensure_indices(sqls, verbose=True)
    marg0 = db.column_marginal(TABLE, COL)
    print("\n[2] 边际快照(原数据): n=%d mean=%.4f std=%.4f" % (marg0["n"], marg0["mean"], marg0["std"]), flush=True)

    # ---------- 3) 逐漂移强度（并行执行） ----------
    rows = []
    for frac in FRACS:
        if frac > 0:
            db.restore_from_disk()
            info = db.permute_column(TABLE, COL, frac=frac, where=WHERE, seed=SEED)
            marg = db.column_marginal(TABLE, COL)
            marg_same = (marg["n"] == marg0["n"] and np.array_equal(marg["hist"], marg0["hist"]))
            print("\n  -- frac=%.2f: 置换 %d/%d 行；边际严格不变=%s --"
                  % (frac, info["moved"], info["rows"], marg_same), flush=True)
            spec = dict(type="permute", table=TABLE, col=COL, frac=frac, where=WHERE, seed=SEED)
            t0 = time.time()
            got = np.array([v if v is not None else np.nan
                            for v in datasim.exec_pool(DB, sqls, spec, N_WORKERS, timeout=Q_TIMEOUT)],
                           dtype=float)
            print("     并行执行 %.0fs" % (time.time() - t0), flush=True)
        else:
            marg_same = True
            got = true0.copy()

        v = ~np.isnan(got)
        lr = np.full(len(sqls), np.nan)
        lr[v] = np.log(got[v]) - np.log(true0[v])
        changed = (np.abs(lr) > 1e-9) & v
        r2_nfc = r2_from_groups(lr, nfc, changed)
        r2_ac = r2_from_groups(lr, has_ac, changed)

        print("     真值变化 %d/%d (%.1f%%) | 单特征 R^2: 过滤列数=%.3f, 含answercount=%.3f"
              % (changed.sum(), v.sum(), 100.0 * changed.sum() / max(v.sum(), 1), r2_nfc, r2_ac), flush=True)
        if changed.sum() > 0:
            lrv = lr[changed]
            print("     修正量 log(true_new/true_old): mean=%+.4f std=%.4f | q5=%+.3f q50=%+.3f q95=%+.3f"
                  % (lrv.mean(), lrv.std(), np.percentile(lrv, 5), np.percentile(lrv, 50), np.percentile(lrv, 95)), flush=True)
        rows.append(dict(frac=frac, marg_same=bool(marg_same), n_changed=int(changed.sum()),
                         lr=lr, r2_nfc=float(r2_nfc), r2_ac=float(r2_ac)))

    # ---------- 4) 与现有标签级注入正面对比 ----------
    print("\n" + "=" * 100)
    print("表 对比：现有注入 vs 真数据漂移（在【多列过滤查询】上的修正量）")
    print("%-22s | %10s | %10s | %10s | %10s | %8s" % ("漂移方式", "均值", "标准差", "q5", "q95", "n"))
    print("-" * 100)
    mu = 2.0
    inj = np.where(nfc >= 2, mu, 0.0); injv = inj[inj > 0]
    print("%-22s | %+10.4f | %10.4f | %+10.3f | %+10.3f | %8d"
          % ("现有注入(mu=2, 常数)", injv.mean(), injv.std(), np.percentile(injv, 5), np.percentile(injv, 95), len(injv)))
    for r in rows[1:]:
        lr = r["lr"]; m = (nfc >= 2) & ~np.isnan(lr) & (np.abs(lr) > 1e-9)
        if m.sum() == 0:
            continue
        lrv = lr[m]
        print("%-22s | %+10.4f | %10.4f | %+10.3f | %+10.3f | %8d"
              % ("真漂移 frac=%.2f" % r["frac"], lrv.mean(), lrv.std(),
                 np.percentile(lrv, 5), np.percentile(lrv, 95), m.sum()))
    print("\n要点：现有注入 std=0（常数）；真漂移 std>0 且 q5/q95 跨越大范围，")
    print("      单特征只能解释很小一部分方差 -> 修正量随查询非线性、值依赖变化。")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_realpool.npz")
    np.savez_compressed(out, sql=np.array(sqls, dtype=object), true0=true0, nfc=nfc, has_ac=has_ac,
                        fracs=np.array(FRACS), lr=np.array([r["lr"] for r in rows]))
    print("\n已存 %s" % out)


if __name__ == "__main__":
    main()
