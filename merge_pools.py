# -*- coding: utf-8 -*-
r"""合并两个漂移池（旧 5 状态 + 新增 perm_creationdate），并先核对 baseline 可比性。

为什么需要核对
--------------
两个池是【两次独立运行】。每个状态开头都会独立重载数据，所以 baseline 在两次运行里
应当完全一致（重载 + statistics_target=2000 全表扫描 ANALYZE 是确定性的）。
若不一致，说明两次运行的库状态有差异，则【不能】把新状态与旧状态直接比较。
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
A = os.path.join(HERE, "pool_stats.pkl")
B = os.path.join(HERE, "pool_cd.pkl")
OUT = os.path.join(HERE, "pool_all.pkl")


def arr(v):
    return np.asarray([np.nan if x is None else x for x in v], dtype=float)


def main():
    pa = pickle.load(open(A, "rb"))
    pb = pickle.load(open(B, "rb"))
    print("=" * 100)
    print("合并两个漂移池")
    print("=" * 100)
    print(f"  A: {os.path.basename(A)}  状态 {list(pa['states'])}  n={pa['meta']['n_queries']}")
    print(f"  B: {os.path.basename(B)}  状态 {list(pb['states'])}  n={pb['meta']['n_queries']}")

    ea, eb = pa["meta"]["n_queries"], pb["meta"]["n_queries"]
    if ea != eb:
        print(f"  [FAIL] 查询数不同（{ea} vs {eb}），不可合并")
        return 1
    ia, ib = list(pa["idx"]), list(pb["idx"])
    if ia != ib:
        print("  [FAIL] idx 序列不同，不可合并")
        return 1
    print("  idx 序列一致 ✓")

    # ★ 可比性核对：baseline 的 pg_est 与 true 是否逐位相同
    print("\n★ baseline 可比性核对（两次独立运行）")
    ta, tb = arr(pa["states"]["baseline"]["true"]), arr(pb["states"]["baseline"]["true"])
    ga, gb = arr(pa["states"]["baseline"]["pg_est"]), arr(pb["states"]["baseline"]["pg_est"])
    oa = np.asarray(pa["states"]["baseline"]["ok"], dtype=bool)
    ob = np.asarray(pb["states"]["baseline"]["ok"], dtype=bool)
    m = np.isfinite(ta) & np.isfinite(tb) & np.isfinite(ga) & np.isfinite(gb)
    dt = np.abs(np.log(np.clip(tb[m], 1, None)) - np.log(np.clip(ta[m], 1, None)))
    dg = np.abs(np.log(np.clip(gb[m], 1, None)) - np.log(np.clip(ga[m], 1, None)))
    print(f"  true   逐条相同: {int((dt == 0).sum())}/{m.sum()}   最大 log 差 {dt.max():.3e}")
    print(f"  pg_est 逐条相同: {int((dg == 0).sum())}/{m.sum()}   最大 log 差 {dg.max():.3e}")
    print(f"  ok 掩码相同: {bool(np.array_equal(oa, ob))}")
    same = (dt.max() == 0) and (dg.max() == 0)
    print("  => " + ("可比：baseline 在两次运行里逐位相同，可直接合并。"
                     if same else
                     "不可比！两次运行的 baseline 不同，新状态不能与旧状态直接比较。"))
    if not same:
        return 2

    # 合并：保留 A 的全部状态 + B 里 A 没有的状态
    states = dict(pa["states"])
    added = []
    for k, v in pb["states"].items():
        if k not in states:
            states[k] = v
            added.append(k)
    meta = dict(pa["meta"])
    for key in ("configs", "feature_valid", "reanalyze", "vacuum_tables"):
        if key not in pb["meta"]:
            continue
        va, vb = meta.get(key), pb["meta"][key]
        if isinstance(vb, dict):
            merged = dict(va) if isinstance(va, dict) else {}
            merged.update(vb)
            meta[key] = merged
        elif isinstance(vb, list):
            meta[key] = list(dict.fromkeys((va if isinstance(va, list) else []) + vb))
        else:
            meta[key] = vb
    out = dict(meta=meta, idx=pa["idx"], sqls=pa["sqls"], states=states)
    pickle.dump(out, open(OUT, "wb"))
    print(f"\n合并完成 -> {OUT}")
    print(f"  新增状态: {added}")
    print(f"  全部状态: {list(states)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
