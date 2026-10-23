# -*- coding: utf-8 -*-
"""数据加载 + 强特征提取（返回 (base, true, feat, n_filter_cols)）"""
import pickle
import numpy as np
import features

def load_stats(pkl_path):
    """返回 (base, true, feat, n_filter_cols)。
    base = pg 估计；true = 真实基数；
    feat = [build_schema_feature | stats9 | log(pg)]；
    n_filter_cols = 每查询"去重过滤列数"（>=2 视为相关性敏感）。
    """
    n_ts, n_cs = features.load_table_col_space(pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    feats = data["features"]

    base_l, true_l, feat_l, nfc_l = [], [], [], []
    for ft, label, pgv, *_ , ex in feats:
        true = float(np.expm1(label - 1))
        pg   = float(np.expm1(pgv - 1))
        if true <= 0 or pg <= 0:
            continue
        f = np.nan_to_num(
            np.concatenate([features.build_schema_feature(ft, ex, n_ts, n_cs), features.stats9(ft)]),
            nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        f = np.hstack([f, np.log(np.clip(pg, 1.0, None))]).astype(np.float32)
        base_l.append(pg); true_l.append(true); feat_l.append(f)
        nfc_l.append(len(set(ex.get("filter_col_identifiers", []))))
    return (np.array(base_l), np.array(true_l), np.array(feat_l, dtype=np.float32),
            np.array(nfc_l, dtype=np.int32))

def split(feat, base, true, nfc, n_train, n_cal, n_test, rng):
    idx = rng.permutation(len(true))
    tr  = idx[:n_train]
    cal = idx[n_train:n_train + n_cal]
    te  = idx[n_train + n_cal : n_train + n_cal + n_test]
    return ((feat[tr], base[tr], true[tr], nfc[tr]),
            (feat[cal], base[cal], true[cal], nfc[cal]),
            (feat[te], base[te], true[te], nfc[te]))
