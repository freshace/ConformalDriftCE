# -*- coding: utf-8 -*-
"""模型无关性演示：同一个 conformal+LGBM 外挂，套在 PG / ALECE / PRICE 三个底模上，
对相关性漂移都能【检测(覆盖掉) + 修复(覆盖回)】。"""
import os, sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, features, model, conformal

ORACLE_DIR = r"../ETGE/results_expert_oracle"

def load_bases(pkl_path):
    n_ts, n_cs = features.load_table_col_space(pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    base_l, true_l, sf_l, nfc_l, idx_l = [], [], [], [], []
    for i, (ftup, label, pgv, *_ , ex) in enumerate(data["features"]):
        t = float(np.expm1(label-1)); p = float(np.expm1(pgv-1))
        if t <= 0 or p <= 0: continue
        sf = np.nan_to_num(np.concatenate([features.build_schema_feature(ftup, ex, n_ts, n_cs), features.stats9(ftup)]),
                           nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        base_l.append(p); true_l.append(t); sf_l.append(sf); idx_l.append(i)
        nfc_l.append(len(set(ex.get("filter_col_identifiers", []))))
    return (np.array(base_l), np.array(true_l), np.array(sf_l, dtype=np.float32), np.array(nfc_l, dtype=np.int32), np.array(idx_l, dtype=np.int32))

def wrap_exp(name, base_pred, sf, true, nfc):
    rng = np.random.RandomState(config.SEED)
    n = len(true)
    # feature = [schema_feature | stats9 | log(base)]
    F = np.hstack([sf, np.log(np.clip(base_pred,1,None))[:, None]]).astype(np.float32)
    idx = rng.permutation(n)
    tr = idx[:config.N_TRAIN]; cal = idx[config.N_TRAIN:config.N_TRAIN+config.N_CAL]; te = idx[config.N_TRAIN+config.N_CAL:config.N_TRAIN+config.N_CAL+config.N_TEST]
    # 训练 wrapper（LGBM 学 log(true/base)）+ conformal
    m_old = model.fit_corrector(F[tr], base_pred[tr], true[tr], config.LGBM_PARAMS)
    q_old = conformal.calibrate_q(conformal.score(true[cal], base_pred[cal], model.predict_corrector(m_old, F[cal])), config.ALPHA)
    base_cov = conformal.coverage(conformal.score(true[te], base_pred[te], model.predict_corrector(m_old, F[te])), q_old)
    # 相关性漂移（多列查询乘 7.39x）
    drift_t = true[te] * np.where(nfc[te]>=2, np.exp(2.0), 1.0)
    cov_frozen = conformal.coverage(conformal.score(drift_t, base_pred[te], model.predict_corrector(m_old, F[te])), q_old)
    # 修复（几百标签）
    perm = rng.permutation(len(te))
    a_tr_l=perm[:config.N_ADAPT_TRAIN]; a_cal_l=perm[config.N_ADAPT_TRAIN:config.N_ADAPT_TRAIN+config.N_ADAPT_CAL]; a_ev_l=perm[config.N_ADAPT_TRAIN+config.N_ADAPT_CAL:]
    a_tr_g=te[a_tr_l]; a_cal_g=te[a_cal_l]; a_ev_g=te[a_ev_l]
    m_new = model.fit_corrector(F[a_tr_g], base_pred[a_tr_g], drift_t[a_tr_l], config.LGBM_PARAMS)
    q_new = conformal.calibrate_q(conformal.score(drift_t[a_cal_l], base_pred[a_cal_g], model.predict_corrector(m_new, F[a_cal_g])), config.ALPHA)
    cov_fix = conformal.coverage(conformal.score(drift_t[a_ev_l], base_pred[a_ev_g], model.predict_corrector(m_new, F[a_ev_g])), q_new)
    print(f"{name:10s} | 无漂移 {base_cov:6.1%} | 漂移后(冻) {cov_frozen:6.1%} | 修复后 {cov_fix:6.1%} | 检-修 {cov_fix-cov_frozen:+6.1%}")

def main():
    base, true, sf, nfc, idx = load_bases(config.DATA_PATH)
    pg = base
    alece = np.load(os.path.join(ORACLE_DIR, "alece_stats.npy"))[idx]
    price = np.load(os.path.join(ORACLE_DIR, "price_stats.npy"))[idx]
    print("模型无关性：同一个 conformal+LGBM 外挂，套在不同底模上（相关性漂移 7.39x）\n")
    print(f"{'底模':10s} | 无漂移 | 漂移后(冻结) | 修复后 | 检测→修复")
    print("-" * 66)
    wrap_exp("PG", pg, sf, true, nfc)
    wrap_exp("ALECE", alece, sf, true, nfc)
    wrap_exp("PRICE", price, sf, true, nfc)

if __name__ == "__main__":
    main()
