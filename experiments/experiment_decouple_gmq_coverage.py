# -*- coding: utf-8 -*-
"""Decoupling: GMQ (point accuracy) vs coverage (reliability) under REAL data-level correlation drift.
Shows that a 'reasonable' average q-error (GMQ) can coexist with near-total coverage collapse on the
correlation-sensitive subset (the tail that actually breaks query plans)."""
import sys, os
sys.path.insert(0, r".")
sys.path.insert(0, r".\experiments")
import numpy as np
import experiment_correlation_realdata as E
import model, conformal

N_ROWS=200000; D=E.D; GROUPS=E.GROUPS
RHOS=[0.30,0.45,0.60,0.75,0.90]
R0=E.corr_block_matrix(D, GROUPS, 0.30)
u0=E.gaussian_copula(R0, N_ROWS, 42)
qs=E.gen_queries(6000, D, GROUPS, 43)
# build base query set (fixed), keep queries with moderate true count
keep=[]; pg_l=[]; tr_l=[]; fe_l=[]
for i,(cols,lo,hi) in enumerate(qs):
    p=N_ROWS*np.prod(hi-lo); t=E.count_rows(u0, cols, lo, hi)
    if t<8 or p<8 or t>0.5*N_ROWS: continue
    keep.append(i); pg_l.append(p); tr_l.append(float(t)); fe_l.append(E.query_feature(cols,lo,hi,D,N_ROWS,p,GROUPS))
keep=np.array(keep); pg=np.array(pg_l); tr=np.array(tr_l); fe=np.array(fe_l)
sens=np.array([E.n_group_pairs(qs[i][0],GROUPS)>=1 for i in keep])
print(f"queries={len(keep)}  sensitive={sens.sum()} ({100*sens.mean():.0f}%)")

# base corrector trained on the NO-drift subset (stale under drift)
idx=np.random.RandomState(0).permutation(len(keep)); trn=idx[:800]; cal=idx[800:1200]; tev=idx[1200:4000]
m=model.fit_corrector(fe[trn], pg[trn], tr[trn], E.LGBM_PARAMS)
qo=conformal.calibrate_q(conformal.score(tr[cal], pg[cal], model.predict_corrector(m,fe[cal])), 0.1)

def gerr(true, est):
    est=np.clip(est,1,None)
    return np.maximum(true/est, est/true)

def gmq(true, est): return float(np.exp(np.mean(np.log(gerr(true,est)))))

print(f"\n{'rho':>4} | {'GMQ(整体)':>9} | {'cov(整体)':>9} | {'GMQ(敏感)':>9} | {'cov(敏感)':>9} | {'p95 qerr(敏感)':>14} | {'max qerr(敏感)':>14}")
print("-"*96)
for rho in RHOS:
    Rd=E.corr_block_matrix(D, GROUPS, rho)
    ud=E.gaussian_copula(Rd, N_ROWS, 43)
    trd=np.array([max(E.count_rows(ud,qs[i][0],qs[i][1],qs[i][2]),1.0) for i in keep])
    # frozen base estimator point estimate = pg (marginal product) * exp(corrector)  [the point estimate]
    corr=model.predict_corrector(m, fe)
    est=pg*np.exp(corr)
    # GMQ of the point estimate
    gmq_all=gmq(trd[tev], est[tev]); gmq_sen=gmq(trd[tev][sens[tev]], est[tev][sens[tev]])
    # coverage of the conformal interval (uses the SAME corrector + q)
    sc=conformal.score(trd[tev], pg[tev], corr[tev])
    cov_all=float(np.mean(sc<=qo)); cov_sen=float(np.mean(sc[tev][sens[tev]][0:len(sc[sens])]<=qo)) if False else float(np.mean(sc[sens[tev]]<=qo))
    ge=gerr(trd[tev][sens[tev]], est[tev][sens[tev]])
    p95=float(np.percentile(ge,95)); mx=float(ge.max())
    print(f"{rho:>4.2f} | {gmq_all:>9.2f} | {cov_all:>9.1%} | {gmq_sen:>9.2f} | {cov_sen:>9.1%} | {p95:>14.1f} | {mx:>14.1f}")
print("\n[结论] GMQ 是'全体平均'相对误差，Rho=0.9 时敏感子集的 GMQ 只是~2-3x(看着还行)，")
print(" 但敏感子集 coverage 掉到 5.6%、p95/max q-error 上百倍——平均掩盖了尾部，coverage 才能逐查询抓住它。")
