# -*- coding: utf-8 -*-
"""基线对比（修正版）：冻结 vs 我们 vs Oracle。
干净切分：eval(500,所有方法在这评估) / cal(500,conformal校准) / ours_train(500) / oracle_train(2000)。"""
import os, sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, features, model, conformal

ALPHA = config.ALPHA
def load(pkl_path):
    n_ts, n_cs = features.load_table_col_space(pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    base_l, tru_l, feat_l, nfc_l, ntab_l = [], [], [], [], []
    for ftup, label, pgv, *_ , ex in data["features"]:
        t = float(np.expm1(label-1)); p = float(np.expm1(pgv-1))
        if t <= 0 or p <= 0: continue
        f = np.nan_to_num(np.concatenate([features.build_schema_feature(ftup, ex, n_ts, n_cs), features.stats9(ftup)]),
                          nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        f = np.hstack([f, np.log(np.clip(p,1,None))]).astype(np.float32)
        base_l.append(p); tru_l.append(t); feat_l.append(f)
        nfc_l.append(len(set(ex.get("filter_col_identifiers", []))))
        ntab_l.append(len(ex.get("table_indices", [])))
    return (np.array(base_l), np.array(tru_l), np.array(feat_l,dtype=np.float32), np.array(nfc_l,dtype=np.int32), np.array(ntab_l,dtype=np.int32))

def compare(name, m_old, q_old, Df, Db, Dt):
    """Df/Db/Dt = 漂移后的池（足够大）。切分 eval/cal/ours_train/oracle_train。"""
    n = len(Df)
    ne = min(500, n//7)          # eval
    Df,Db,Dt = Df[-ne:], Db[-ne:], Dt[-ne:]  # 直接用末尾做 eval（简化，池本身就来自漂移数据）
    # 实际上需要足够大池；这里统一取前 n 个做适配/校准，末尾 ne 做 eval
    return None

def run(name, old_f, old_b, old_t, old_cal_f, old_cal_b, old_cal_t, pool_f, pool_b, pool_t):
    """pool = 漂移后查询池（>=3500）。三段完全不相交：
       eval(500,所有评估) / cal(500,conformal校准) / ours_train(500) / oracle_train(2000)。"""
    ne = 500
    ev_f, ev_b, ev_t = pool_f[-ne:], pool_b[-ne:], pool_t[-ne:]
    pool_f, pool_b, pool_t = pool_f[:-ne], pool_b[:-ne], pool_t[:-ne]
    cal_f, cal_b, cal_t = pool_f[:500], pool_b[:500], pool_t[:500]
    ot_f, ot_b, ot_t = pool_f[500:1000], pool_b[500:1000], pool_t[500:1000]     # ours_train（与 cal 不相交）
    otr_f, otr_b, otr_t = pool_f[1000:3000], pool_b[1000:3000], pool_t[1000:3000]  # oracle_train（与 cal/ours 不相交）
    # 冻结：旧模型 + 旧 q
    m_old = model.fit_corrector(old_f, old_b, old_t, config.LGBM_PARAMS)
    q_old = conformal.calibrate_q(conformal.score(old_cal_t, old_cal_b, model.predict_corrector(m_old, old_cal_f)), ALPHA)
    frozen = conformal.coverage(conformal.score(ev_t, ev_b, model.predict_corrector(m_old, ev_f)), q_old)
    # 我们：500 训练 + 500 校准（同 cal）
    m_ours = model.fit_corrector(ot_f, ot_b, ot_t, config.LGBM_PARAMS)
    q_ours = conformal.calibrate_q(conformal.score(cal_t, cal_b, model.predict_corrector(m_ours, cal_f)), ALPHA)
    ours = conformal.coverage(conformal.score(ev_t, ev_b, model.predict_corrector(m_ours, ev_f)), q_ours)
    # Oracle：2000 训练（不相交）+ 500 校准
    m_or = model.fit_corrector(otr_f, otr_b, otr_t, config.LGBM_PARAMS)
    q_or = conformal.calibrate_q(conformal.score(cal_t, cal_b, model.predict_corrector(m_or, cal_f)), ALPHA)
    oracle = conformal.coverage(conformal.score(ev_t, ev_b, model.predict_corrector(m_or, ev_f)), q_or)
    print(f"{name:22s} | 冻结({frozen:6.1%}) | 我们({ours:6.1%}) | Oracle({oracle:6.1%}) | 我们-Oracle({ours-oracle:+5.1%}) | 我们-冻结({ours-frozen:+6.1%})")
    return frozen, ours, oracle

def main():
    rng = np.random.RandomState(config.SEED)
    base, true, feat, nfc, ntab = load(config.DATA_PATH)
    n = len(true)

    # 1) 相关性漂移：池=测试集（乘相关因子）
    idx = rng.permutation(n)
    tr = idx[:config.N_TRAIN]; cal = idx[config.N_TRAIN:config.N_TRAIN+config.N_CAL]; te = idx[config.N_TRAIN+config.N_CAL:config.N_TRAIN+config.N_CAL+config.N_TEST]
    drift_t = true[te] * np.where(nfc[te]>=2, np.exp(2.0), 1.0)
    print("相关因子 7.39x:")
    run("相关性(7.39x)", feat[tr],base[tr],true[tr], feat[cal],base[cal],true[cal], feat[te],base[te],drift_t)

    # 2) 工作负载漂移：训练简单，池=复杂查询
    tr_s = rng.choice(np.where(ntab<=3)[0], size=1500, replace=False)
    cal_s = rng.choice(np.setdiff1d(np.where(ntab<=3)[0], tr_s), size=300, replace=False)
    te_c = rng.choice(np.where(ntab>=5)[0], size=4000, replace=False)
    print("\n工作负载(简单→复杂):")
    run("工作负载(简单→复杂)", feat[tr_s],base[tr_s],true[tr_s], feat[cal_s],base[cal_s],true[cal_s], feat[te_c],base[te_c],true[te_c])

if __name__ == "__main__":
    main()
