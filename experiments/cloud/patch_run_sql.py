# -*- coding: utf-8 -*-
"""Patch run_sql.py so the periodic schedule can be phase-shifted from the shell.

Three edits, each anchored on a unique substring and verified; the file is only
written when every anchor matches exactly once.  A .bak copy is kept.

    python3 patch_run_sql.py code/run_sql.py

Idempotent: running it twice is a no-op.
"""
import pathlib
import shutil
import sys

TARGET = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "code/run_sql.py")
src = TARGET.read_text(encoding="utf-8")

if "PERIOD_PHASE" in src:
    print(f"{TARGET}: already patched, nothing to do")
    raise SystemExit(0)

EDITS = [
    # 1. allow an explicit schedule to be passed in
    (
        "def run_policy(d,p,roles,p0,q0,tau0,acq,initial_cost,policy,seed,condition):",
        "def run_policy(d,p,roles,p0,q0,tau0,acq,initial_cost,policy,seed,condition,schedule=None):",
    ),
    # 2. shift the periodic schedule by PERIOD_PHASE windows
    (
        ";schedules={'periodic':set(np.rint(np.linspace(2,W-3,p['max_repairs'])).astype(int)),"
        "'random':set(rng.choice(W-1,p['max_repairs'],replace=False))}",
        ";schedules={'periodic':set(np.rint(np.linspace(2,W-3,p['max_repairs'])).astype(int)),"
        "'random':set(rng.choice(W-1,p['max_repairs'],replace=False))}\n"
        " _ph=int(os.environ.get('PERIOD_PHASE','0'))\n"
        " if _ph: schedules['periodic']=set(np.clip(np.rint(np.linspace(2,W-3,p['max_repairs'])).astype(int)+_ph,1,W-2))\n"
        " if schedule is not None: schedules.update({k:set(v) for k,v in schedule.items()})",
    ),
    # 3. optional filter so a sweep runs only the periodic policy
    (
        "   for policy in p['policy_orders'][str(seed)][condition]:",
        "   _only=os.environ.get('CE_ONLY_PERIODIC')=='1'\n"
        "   for policy in ([x for x in p['policy_orders'][str(seed)][condition] if x=='periodic'] if _only else p['policy_orders'][str(seed)][condition]):",
    ),
]

for old, new in EDITS:
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"ANCHOR PROBLEM: found {n} matches for {old[:70]!r}; refusing to write")
    src = src.replace(old, new)

shutil.copy(TARGET, TARGET.with_suffix(".py.bak"))
TARGET.write_text(src, encoding="utf-8")
print(f"{TARGET}: patched (backup at {TARGET.with_suffix('.py.bak')})")
