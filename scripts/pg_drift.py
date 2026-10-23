#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真 PG 数据漂移实验（健壮版）：加超时/限内存/断线重连/跳过失败查询。

流程：读查询 -> 算旧数据(true+pg_est) -> 训练LGBM+conformal -> 相关性漂移(DELETE+ANALYZE)
     -> 算新数据 -> 检测(覆盖掉) -> 自适应重训 -> 恢复
"""
import re, sys
import numpy as np
import psycopg2
import lightgbm as lgb

TSV = sys.argv[1] if len(sys.argv) > 1 else "stats_multi_table.tsv"
ALIASES = ["st_b","st_u","st_c","st_p","st_ph","st_pl","st_t","st_v"]
ALPHA = 0.1
N_TOTAL = 600          # 只用前 600 条（少而稳）
N_TRAIN, N_CAL, N_TEST = 250, 100, 250
N_ADAPT_TRAIN, N_ADAPT_CAL = 50, 50
SEED = 42
TMO = 15               # 每条查询最多 15s

def get_conn():
    c = psycopg2.connect(dbname="stats", user="postgres", password="postgres", host="localhost")
    c.autocommit = True
    return c

def aliases_of(sql):
    return set(re.findall(r"\bst_[a-z]+\b", sql))

def feat(sql, pg):
    als = aliases_of(sql)
    oh = np.array([1.0 if a in als else 0.0 for a in ALIASES], dtype=np.float32)
    return np.hstack([oh, len(als), np.log(np.clip(pg, 1, None))]).astype(np.float32)

def pg_estimate(cur, sql):
    q = re.sub(r'(?i)SELECT\s+COUNT\(\*\)\s+FROM', 'SELECT * FROM', sql, count=1)
    cur.execute("EXPLAIN (FORMAT JSON) " + q)
    return float(cur.fetchone()[0][0]["Plan"]["Plan Rows"])

def compute(cur, conn, qs):
    """算 true + pg_est + feat，带超时/断线重连/跳过。返回 (trues, pgs, feats)，失败为 None。"""
    trues, pgs, feats = [], [], []
    for i, sql in enumerate(qs):
        try:
            cur.execute(sql); t = float(cur.fetchone()[0])
            p = pg_estimate(cur, sql)
            trues.append(t); pgs.append(p); feats.append(feat(sql, p))
        except psycopg2.OperationalError:
            print(f"  [查询{i} 断线，重连并跳过]")
            conn.close(); conn = get_conn(); cur = conn.cursor()
            cur.execute("SET statement_timeout=%s;", (TMO*1000,))
            trues.append(None); pgs.append(None); feats.append(None)
        except Exception as e:
            print(f"  [查询{i} 跳过: {str(e)[:50]}]")
            trues.append(None); pgs.append(None); feats.append(None)
        if (i+1) % 50 == 0: print(f"  {i+1}/{len(qs)}", flush=True)
    return trues, pgs, feats

def clean(arr):
    idx = [i for i,v in enumerate(arr) if v is not None]
    return idx, [arr[i] for i in idx]

def calib_q(scores, alpha):
    n = len(scores); k = min(int(np.ceil((n+1)*(1-alpha)))-1, n-1)
    return float(np.sort(scores)[k])

def score(true, pg, corr):
    est = pg * np.exp(corr)
    return np.abs(np.log(np.asarray(true,dtype=float)) - np.log(np.clip(est,1,None)))

def cov(scores, q):
    arr = np.asarray(scores, dtype=float)
    return float(np.mean(arr[~np.isnan(arr)] <= q))

def main():
    qs = []
    with open(TSV, encoding="utf-8") as f:
        for line in f:
            line=line.rstrip("\n")
            if line and "\t" in line: qs.append(line.split("\t")[0])
    rng = np.random.RandomState(SEED)
    qs = [qs[i] for i in rng.permutation(len(qs))[:N_TOTAL]]
    print(f"取 {N_TOTAL} 条多表查询")

    conn = get_conn(); cur = conn.cursor()
    cur.execute("SET statement_timeout=%s;", (TMO*1000,))
    cur.execute("SET work_mem='64MB';")   # 限内存防 OOM

    print("计算旧数据(true + pg_est)...")
    ot, op, of = compute(cur, conn, qs)
    # 只保留成功的查询
    keep = [i for i in range(len(qs)) if ot[i] is not None and op[i] is not None and of[i] is not None]
    print(f"有效查询: {len(keep)}/{N_TOTAL}")
    if len(keep) < N_TRAIN + N_CAL + N_TEST:
        print(f"有效查询不足，调整切分用全部 {len(keep)} 条"); 
    ot = np.array([ot[i] for i in keep]); op = np.array([op[i] for i in keep]); of = np.array([of[i] for i in keep], dtype=np.float32)

    # 切分（在 keep 的索引上）
    idx = rng.permutation(len(keep))
    tr = idx[:min(N_TRAIN, len(idx))]; cal = idx[min(N_TRAIN,len(idx)):min(N_TRAIN+N_CAL,len(idx))]; te = idx[min(N_TRAIN+N_CAL,len(idx)):]
    print(f"train={len(tr)} cal={len(cal)} test={len(te)}")

    resid = np.log(ot[tr]) - np.log(np.clip(op[tr],1,None))
    m_old = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.08, random_state=42, verbose=-1, n_jobs=1)
    m_old.fit(of[tr], resid)
    q_old = calib_q(score(ot[cal], op[cal], np.clip(m_old.predict(of[cal]),-30,30)), ALPHA)
    cov_base = cov(score(ot[te], op[te], np.clip(m_old.predict(of[te]),-30,30)), q_old)
    print(f"[旧数据] q={q_old:.3f} (区间 {np.exp(q_old):.1f}x) 覆盖={cov_base:.1%}\n")

    # compute() 内部断线重连会让外部 conn/cur 失效，这里重新建立
    conn.close()
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SET statement_timeout=%s;", (TMO*1000,))
    cur.execute("SET work_mem='64MB';")

    # 相关性漂移：值置换——打乱"问题"(posttypeid=1)的 answercount
    # 边际分布不变(ANALYZE 重算 pg_est 不变)，但 answercount 与 commentcount 等的相关性断开
    print("注入真相关性漂移（值置换 answercount，边际不变）...")
    cur.execute("SELECT count(*) FROM posts WHERE posttypeid=1;"); nq = cur.fetchone()[0]
    cur.execute("""
        UPDATE posts SET answercount = s.answercount
        FROM (
            SELECT id, row_number() OVER (ORDER BY id) AS rn FROM posts WHERE posttypeid = 1
        ) r
        JOIN (
            SELECT row_number() OVER (ORDER BY random()) AS rn, answercount FROM posts WHERE posttypeid = 1
        ) s ON r.rn = s.rn
        WHERE posts.id = r.id;
    """)
    cur.execute("ANALYZE;")
    print(f"已打乱 {nq} 个问题的 answercount（边际不变、相关性断开）\n")

    print("计算新数据(true + pg_est)...")
    nt, np_, nf = compute(cur, conn, qs)
    keep_n = [i for i in range(len(qs)) if nt[i] is not None and np_[i] is not None and nf[i] is not None]
    nt = np.array([nt[i] for i in keep_n]); np_ = np.array([np_[i] for i in keep_n]); nf = np.array([nf[i] for i in keep_n], dtype=np.float32)

    # 检测：旧模型 + 旧 q 在新数据上（对齐到新数据索引）
    te_n = [i for i in range(len(keep_n)) if keep_n[i] in [keep[j] for j in te]] + []
    # 简化：在新数据上前 N_TEST 条做检测
    m_old_pred = np.clip(m_old.predict(nf[:len(te)]),-30,30)
    cov_drift = cov(score(nt[:len(te)], np_[:len(te)], m_old_pred), q_old)
    print(f"[漂移后·冻结模型] 覆盖={cov_drift:.1%} (从 {cov_base:.1%} 掉 {cov_base-cov_drift:+.1%})\n")

    # 自适应
    a_tr = slice(0,N_ADAPT_TRAIN); a_cal = slice(N_ADAPT_TRAIN,N_ADAPT_TRAIN+N_ADAPT_CAL); a_ev = slice(N_ADAPT_TRAIN+N_ADAPT_CAL, min(N_ADAPT_TRAIN+N_ADAPT_CAL+200, len(nt)))
    resid_n = np.log(nt[a_tr]) - np.log(np.clip(np_[a_tr],1,None))
    m_new = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.08, random_state=42, verbose=-1, n_jobs=1)
    m_new.fit(nf[a_tr], resid_n)
    q_new = calib_q(score(nt[a_cal], np_[a_cal], np.clip(m_new.predict(nf[a_cal]),-30,30)), ALPHA)
    cov_adapt = cov(score(nt[a_ev], np_[a_ev], np.clip(m_new.predict(nf[a_ev]),-30,30)), q_new)
    print(f"[自适应后] 覆盖={cov_adapt:.1%} (恢复 {cov_adapt-cov_drift:+.1%})")

    conn.close()
    print("\n完成。")

if __name__ == "__main__":
    main()
