#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PARQO 式期望惩罚稳健计划选择（真 PG + pg_hint_plan Rows() 强制基数）。
对每条多表查询，比较三种"选计划"策略的 plan cost（EXPLAIN Total Cost）：
  - point   : 用点估计基数（漂移后偏大）hint 选计划
  - true    : 用真实基数 hint 选计划（理想下界）
  - robust  : 在 conformal 区间 [est/e^q, est*e^q] 内采候选基数，选期望惩罚最小者
指标：regret = cost_X / cost_true；expected penalty = robust 的 cost / cost_true。

关键：脚本同时输出"各基数下 Total Cost 是否分化"，以判断 Rows() hint 是否真的改变计划选择。
用法: python3 experiment_parqo_penalty.py [n_queries] [tsv] [drift_factor]
前提: stats 库已 CREATE EXTENSION pg_hint_plan 且 shared_preload_libraries 含 pg_hint_plan。
"""
import sys, re, os
import numpy as np
import psycopg2

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 200
TSV = sys.argv[2] if len(sys.argv) > 2 else "stats_multi_table.tsv"
DRIFT = float(sys.argv[3]) if len(sys.argv) > 3 else 7.39
BASE = os.path.dirname(os.path.abspath(__file__))
Q = 2.04   # conformal 分位数（对数残差），区间约 7.7x

def load(tsv):
    qs = []
    with open(os.path.join(BASE, tsv), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line: continue
            p = line.split("\t")
            if len(p) >= 2: qs.append((p[0], float(p[1])))
    return qs

def join_aliases(sql):
    m = re.search(r'\bfrom\s+(.+?)\s+where\b', sql, flags=re.I)
    if not m: return []
    return [re.search(r'\bas\s+(\w+)', x).group(1) for x in m.group(1).split(",") if re.search(r'\bas\s+(\w+)', x)]

def hint_for(sql, n):
    als = join_aliases(sql)
    if len(als) < 2: return ""
    a, b = als[-2], als[-1]
    return f"/*+ Rows({a} {b} #{int(n)}) */\n"

def total_cost(cur, sql, n):
    cur.execute("SET pg_hint_plan.enable_hint = on;")
    try:
        h = hint_for(sql, n)
        cur.execute("EXPLAIN (FORMAT JSON) " + h + sql)
        root = cur.fetchone()[0][0]["Plan"]
        return float(root.get("Total Cost", float("inf")))
    except Exception:
        return float("inf")

def main():
    qs = load(TSV)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(qs))[:NQ]
    conn = psycopg2.connect(dbname=os.environ.get("PGDATABASE", "stats"),
                           user=os.environ.get("PGUSER", "postgres"),
                           password=os.environ.get("PGPASSWORD", "postgres"),
                           host=os.environ.get("PGHOST", "localhost"),
                           port=os.environ.get("PGPORT", "5432"))
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 15000")

    rows = []
    cost_varies = 0
    for i in idx:
        sql, true = qs[i]
        if true <= 0: continue
        als = join_aliases(sql)
        if len(als) < 2: continue
        try:
            ct = total_cost(cur, sql, true)
            if not np.isfinite(ct) or ct <= 0: continue
            est = true * DRIFT   # 漂移后点估计偏大
            cp = total_cost(cur, sql, est)
            # conformal 区间
            lo, hi = est/np.exp(Q), est*np.exp(Q)
            cands = np.unique(np.exp(np.linspace(np.log(max(lo,1)), np.log(max(hi,1)), 7)))
            rob_costs = [total_cost(cur, sql, c) for c in cands]
            cr = min(c for c in rob_costs if np.isfinite(c)) if any(np.isfinite(c) for c in rob_costs) else float("inf")
            # 判断计划是否分化（不同基数下 Total Cost 是否变化）
            if len(set(round(c,2) for c in [ct,cp,cr] if np.isfinite(c))) > 1:
                cost_varies += 1
            rows.append((true, est, ct, cp, cr))
        except Exception:
            continue

    if not rows:
        print("无有效查询"); cur.close(); conn.close(); return
    a = np.array(rows)
    rp = a[:,3]/a[:,2]; rr = a[:,4]/a[:,2]
    print(f"\n完成 {len(a)} 条 (stats, seed 42, drift={DRIFT})")
    print(f"计划分化(Total Cost 随基数变化)的查询数: {cost_varies}/{len(a)}")
    print("=== plan regret (cost/cost_true) ===")
    print(f"  点估计: median={np.median(rp):.2f} p90={np.percentile(rp,90):.2f} max={rp.max():.1f}")
    print(f"  稳健 : median={np.median(rr):.2f} p90={np.percentile(rr,90):.2f} max={rr.max():.1f}")
    print(f"  差   : median={np.median(rp-rr):.2f} p90={np.percentile(rp-rr,90):.2f}")
    with open("/root/results_parqo_penalty.txt","w",encoding="utf-8") as f:
        f.write(f"n={len(a)} drift={DRIFT} cost_varies={cost_varies}\n")
        f.write(f"point median={np.median(rp):.3f} p90={np.percentile(rp,90):.3f} max={rp.max():.2f}\n")
        f.write(f"robust median={np.median(rr):.3f} p90={np.percentile(rr,90):.3f} max={rr.max():.2f}\n")
    print("\n结果已存 /root/results_parqo_penalty.txt")
    cur.close(); conn.close()

if __name__ == "__main__":
    main()