# -*- coding: utf-8 -*-
"""Shared disjoint setup: initial corrector on no-drift train, conformal cal, then a LARGE
drift pool split into a DISJOINT repair_pool + eval (so repair labels never touch eval)."""
import sys, numpy as np
sys.path.insert(0, r".")
import config, data, model, conformal, drift

def setup():
    b_, t_, f_, nfc_ = data.load_stats(config.DATA_PATH)
    N = len(t_)
    idx = np.random.RandomState(config.SEED).permutation(N)
    train = idx[:config.N_TRAIN]                          # 20000 (initial corrector, no-drift)
    cal   = idx[config.N_TRAIN:config.N_TRAIN+config.N_CAL]  # 3000 (conformal q)
    pool_start = config.N_TRAIN + config.N_CAL            # 23000
    n_rep_pool = 12000; n_eval = 4000
    if pool_start + n_rep_pool + n_eval > N:
        raise SystemExit("drift pool too small; need >= " + str(pool_start+n_rep_pool+n_eval))
    repair_pool = idx[pool_start:pool_start+n_rep_pool]
    eval_idx    = idx[pool_start+n_rep_pool:pool_start+n_rep_pool+n_eval]
    m_old = model.fit_corrector(f_[train], b_[train], t_[train], config.LGBM_PARAMS)
    q_old = conformal.calibrate_q(conformal.score(t_[cal], b_[cal], model.predict_corrector(m_old, f_[cal])), config.ALPHA)
    return dict(b=b_, t=t_, f=f_, nfc=nfc_, train=train, cal=cal,
                repair_pool=repair_pool, eval_idx=eval_idx, m_old=m_old, q_old=q_old)

def gerr(t,e): return np.maximum(t/np.clip(e,1,None), np.clip(e,1,None)/t)
def gmq(t,e): return float(np.exp(np.mean(np.log(gerr(t,e)))))
def repair_corrector(S, L, yd):
    rp = S["repair_pool"]
    return model.fit_corrector(S["f"][rp[:L]], S["b"][rp[:L]], yd[rp[:L]], config.LGBM_PARAMS)


rng = np.random.RandomState(config.SEED)
S = setup()
yd_full = lambda mean: drift.inject_correlation_drift(S["t"], S["nfc"], mean, rng)
E = S["eval_idx"]; be = S["b"][E]; fe = S["f"][E]
rows=[]; lines=[f"{'drift':>6} | {'GMQ裸底模':>9} | {'GMQ冻结':>8} | {'GMQ修复500':>10} | {'裸->修复改善':>11}", "-"*62]
for mean in config.DRIFT_MEANS:
    yd = yd_full(mean); yev = yd[E]
    gb = gmq(yev, be)
    gf = gmq(yev, be*np.exp(model.predict_corrector(S["m_old"], fe)))
    mr = repair_corrector(S, config.N_ADAPT_TRAIN, yd)          # 500 labels, disjoint repair_pool
    gr = gmq(yev, be*np.exp(model.predict_corrector(mr, fe)))
    imp = 100*(gb-gr)/gb
    rows.append([np.exp(mean), gb, gf, gr])
    line = f"{np.exp(mean):>6.2f} | {gb:>9.2f} | {gf:>8.2f} | {gr:>10.2f} | {('+' if imp>0 else '')+format(imp,'.1f')+'%':>11}"
    lines.append(line); print(line)
import os; out=os.path.join(r".","results_repro"); os.makedirs(out,exist_ok=True)
open(os.path.join(out,"experiment_accuracy_gmq.txt"),"w",encoding="utf-8").write("\n".join(lines)+"\n")
np.save(os.path.join(r".","results_accuracy_gmq.npy"), np.array(rows))
print("saved results_repro/experiment_accuracy_gmq.txt + results_accuracy_gmq.npy (disjoint repair/eval)")
