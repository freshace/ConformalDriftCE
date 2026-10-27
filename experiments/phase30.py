# -*- coding: utf-8 -*-
"""30-seed version of the best-post-hoc-phase comparison (phase +4, 10 repairs)."""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "provenance" / "2026-10-12" / "code"
PROV = ROOT / "provenance" / "2026-10-12"
CACHE = ROOT / "data" / "period_sweep" / "cache"
sys.path.insert(0, str(CODE))

import numpy as np                                   # noqa: E402
import run_offline as R                              # noqa: E402
from core import partition                           # noqa: E402

R.np.savez_compressed = lambda *a, **k: None
P = json.loads((PROV / "protocol" / "offline.json").read_text(encoding="utf-8"))
DBS = ["stats", "imdb", "ergastf1", "genome"]
OLD = list(P["seeds"])
NEW = list(range(130, 150))
BASE = np.rint(np.linspace(6, P["windows"] - 4, P["max_repairs"])).astype(int)
SCHED = sorted(set(np.clip(BASE + 4, 1, P["windows"] - 2).tolist()))


def load(name):
    z = np.load(PROV / "inputs" / f"dataset_{name}.npz")
    d = {k: z[k] for k in z.files}
    d["name"] = name
    return d


def roles(d, name, seed):
    p = PROV / "raw" / f"roles_{name}_{seed}.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    sizes = {"train": P["initial_train"], "cal": P["initial_calibration"],
             "val": P["threshold_windows"] * P["monitor_per_window"],
             "negative": P["negative_windows"] * P["monitor_per_window"],
             "stream": P["windows"] * (P["monitor_per_window"] + P["audit_per_window"]),
             "repair_pool": P["max_repairs"] * P["repair_block"]}
    return partition(len(d["base"]), sizes, seed)


def initial(name, seed):
    p = PROV / "raw" / f"offline_initial_{name}_{seed}.npz"
    if p.exists():
        z = np.load(p)
        return z["p0"], float(z["q0"]), z["tau0"]
    z = np.load(CACHE / f"initial_{name}_{seed}.npz")
    return z["p0"], float(z["q0"]), z["tau0"]


cur = {}
for path in [ROOT / "data" / "tables" / "offline_runs.csv",
             ROOT / "data" / "continuous" / "offline_runs.csv"]:
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if r["pattern"] == "cycles" and r["policy"] == "current_coverage":
            cur[(r["dataset"], int(r["seed"]))] = float(r["coverage"])
for r in csv.DictReader((ROOT / "data" / "period_sweep" / "seed_extension.csv").open(encoding="utf-8")):
    if r["policy"] == "current_coverage":
        cur[(r["dataset"], int(r["seed"]))] = float(r["coverage"])

old = {}
for r in csv.DictReader((ROOT / "data" / "period_sweep" / "period_runs.csv").open(encoding="utf-8")):
    if r["setting"] == "A:10 repairs, phase+4":
        old[(r["dataset"], int(r["seed"]))] = float(r["coverage"])

print(f"schedule: {SCHED}")
print(f"{'db':9s} {'n':>3s} {'bestphase':>9s} {'current':>8s} {'gain':>7s} {'ci':>18s}")
rng = np.random.default_rng(13)
for name in DBS:
    d = load(name)
    rows = dict(old)
    for seed in NEW:
        p0, q0, tau0 = initial(name, seed)
        r, _ = R.run_policy(d, P, roles(d, name, seed), p0, q0, tau0, "cycles", "periodic", seed,
                            schedule={"periodic": SCHED})
        rows[(name, seed)] = r["coverage"]
    for label, seeds in (("10", OLD), ("30", OLD + NEW)):
        g = np.array([100 * (cur[(name, s)] - rows[(name, s)]) for s in seeds])
        boots = [g[rng.integers(0, len(g), len(g))].mean() for _ in range(4000)]
        lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
        print(f"{name:9s} {label:>3s} {100*np.mean([rows[(name,s)] for s in seeds]):9.2f} "
              f"{100*np.mean([cur[(name,s)] for s in seeds]):8.2f} {g.mean():7.2f}  [{lo:6.2f},{hi:6.2f}]")
