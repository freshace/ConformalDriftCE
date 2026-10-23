# -*- coding: utf-8 -*-
"""强特征：schema one-hot + 统计 + log(pg)（移植自 ETGE 的 build_schema_feature / stats9）"""
import pickle
import numpy as np

def _ent(h):
    h = h[h > 0]
    if h.size == 0 or h.sum() <= 0:
        return 0.0
    h = h / h.sum()
    return -float(np.sum(h * np.log(h + 1e-9)))

def stats9(ft):
    """9 维统计：fanout mean/max、join 直方图熵、过滤选择性 mean/max/std、过滤熵、表规模、等值占比"""
    jh, ff, tf, ftf = ft
    ff = np.asarray(ff, dtype=np.float64).reshape(-1, 40)
    fan_mean = float(ff.mean()) if len(ff) else 1.0
    fan_max  = float(ff.max()) if len(ff) else 1.0
    jh = np.asarray(jh, dtype=np.float64).reshape(-1, 40) if jh is not None else np.zeros((0, 40))
    H_join = float(np.mean([_ent(x) for x in jh])) if len(jh) else 0.0
    tf = np.asarray(tf, dtype=np.float64).reshape(-1, 4)
    S_table = float(np.log(np.exp(tf[:, 0]).sum() + 1))
    sels = np.array([]); H_filter = 0.0; eq = 0.0
    if ftf is not None:
        ftf = np.asarray(ftf, dtype=np.float64).reshape(-1, 43)
        sels = ftf[:, 42]
        H_filter = float(np.mean([_ent(x[:40]) for x in ftf])) if len(ftf) else 0.0
        eq = float(np.mean([1 if (x[41] - x[40]) < 0.05 else 0 for x in ftf])) if len(ftf) else 0.0
    return np.array([fan_mean, fan_max, H_join,
                     float(sels.mean()) if len(sels) else 0.5,
                     float(sels.max()) if len(sels) else 0.5,
                     float(sels.std()) if len(sels) else 0.0,
                     H_filter, S_table, eq], dtype=np.float32)

def build_schema_feature(ft, extras, n_table_space, n_col_space):
    """schema 身份 + 过滤值域特征（区分查询"是什么结构、过滤在哪个值域"）"""
    table_indices = list(extras.get("table_indices", []))
    col_ids = extras.get("col_indices", [])
    n_join = len(extras.get("join_col_identifiers", []))
    n_filter = len(extras.get("filter_col_identifiers", []))
    n_fanout = len(extras.get("fanout_pairs", []))
    n_tables = len(table_indices)

    bot = np.zeros(n_table_space, dtype=np.float32)
    for ti in table_indices:
        if 0 <= int(ti) < n_table_space:
            bot[int(ti)] += 1.0
    if n_tables > 0:
        bot = bot / n_tables

    boc = np.zeros(n_col_space, dtype=np.float32)
    for ci in col_ids:
        if 0 <= int(ci) < n_col_space:
            boc[int(ci)] += 1.0
    denom = max(n_join + n_filter, 1)
    boc = boc / denom

    qb = extras.get("quantile_buckets", [])
    filter_col_ids = col_ids[len(extras.get("join_col_identifiers", [])):]
    qvec = np.zeros(n_col_space * 10, dtype=np.float32)
    for ci, q in zip(filter_col_ids, qb):
        idx = int(ci) * 10 + int(q)
        if 0 <= idx < n_col_space * 10:
            qvec[idx] += 1.0
    if len(qb) > 0:
        qvec = qvec / len(qb)

    counts = np.array([n_tables, n_join, n_filter, n_fanout], dtype=np.float32)
    x = np.concatenate([bot, boc, qvec, counts])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n = np.linalg.norm(x)
    if n > 1e-9:
        x = x / n
    return x.astype(np.float32)

def load_table_col_space(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    sg = data["schema_graph"]
    return max(sg["table_index"].values()) + 1, max(sg["col_index"].values()) + 1
