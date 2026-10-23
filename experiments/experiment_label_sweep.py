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
E = S["eval_idx"]; be = S["b"][E]; fe = S["f"][E]
def dist(t,e):
    ge=gerr(t,e); return [gmq(t,e), float(np.percentile(ge,50)), float(np.percentile(ge,90)), float(np.percentile(ge,95)), float(np.percentile(ge,99)), float(ge.max())]
out_lines=[]; rows=[]
for mean in [1.0, 2.0]:
    yd = drift.inject_correlation_drift(S["t"], S["nfc"], mean, rng); yev = yd[E]
    out_lines.append(f"\n## drift={np.exp(mean):.2f}x   (GMQ | p50 | p90 | p95 | p99 | max)")
    print(out_lines[-1])
    for nm,e in [("bare_base",be),("frozen_2e4",be*np.exp(model.predict_corrector(S["m_old"],fe)))]:
        d=dist(yev,e)
        out_lines.append(f"  {nm:<12}: " + "  ".join(f"{v:.1f}" if i>0 else f"{v:.2f}" for i,v in enumerate(d)))
        print(f"  {nm:<12}: " + "  ".join(f"{v:.1f}" if i>0 else f"{v:.2f}" for i,v in enumerate(d)))
    for L in [500,1000,2000,5000,10000]:
        mr = repair_corrector(S, L, yd)
        d = dist(yev, be*np.exp(model.predict_corrector(mr, fe)))
        out_lines.append(f"  {'repair_'+str(L):<12}: " + "  ".join(f"{v:.1f}" if i>0 else f"{v:.2f}" for i,v in enumerate(d)))
        print(f"  {'repair_'+str(L):<12}: " + "  ".join(f"{v:.1f}" if i>0 else f"{v:.2f}" for i,v in enumerate(d)))
        rows.append([np.exp(mean), L]+d)
import os; out=os.path.join(r".","results_repro"); os.makedirs(out,exist_ok=True)
open(os.path.join(out,"experiment_label_sweep.txt"),"w",encoding="utf-8").write("\n".join(out_lines)+"\n")
np.save(os.path.join(r".","results_label_sweep.npy"), np.array(rows))
print("saved results_repro/experiment_label_sweep.txt + results_label_sweep.npy (disjoint)")
print("cols=[drift, n_labels, GMQ, p50, p90, p95, p99, max]")
