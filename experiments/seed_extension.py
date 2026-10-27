# -*- coding: utf-8 -*-
"""Extend the 80-window recurring-drift replay from 10 to 30 seeds.

Only three policies are re-run on the new seeds (current coverage, periodic,
initial reference); the archived ten-seed results are read from the result
tables so the two sets stay comparable. Nothing in the archived files is touched.

Output: data/period_sweep/seed_extension.csv and a printed comparison of the
paired current-minus-periodic gain over 10 and over 30 seeds.
"""
import csv, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "provenance" / "2026-10-12" / "code"
PROV = ROOT / "provenance" / "2026-10-12"
OUTDIR = ROOT / "data" / "period_sweep"
CACHE = OUTDIR / "cache"
sys.path.insert(0, str(CODE))

import numpy as np                                   # noqa: E402
import lightgbm as lgb                               # noqa: E402
import run_offline as R                              # noqa: E402
from core import radius, partition                   # noqa: E402

_SAVEZ = np.savez_compressed
R.np.savez_compressed = lambda *a, **k: None

P = json.loads((PROV / "protocol" / "offline.json").read_text(encoding="utf-8"))
DBS = ["stats", "imdb", "ergastf1", "genome"]
OLD = list(P["seeds"])
NEW = list(range(130, 150))
POLICIES = ["current_coverage", "periodic", "frozen_trigger"]


def load_dataset(name):
    z = np.load(PROV / "inputs" / f"dataset_{name}.npz")
    d = {k: z[k] for k in z.files}
    d["name"] = name
    return d


def roles_for(d, name, seed):
    path = PROV / "raw" / f"roles_{name}_{seed}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    sizes = {
        "train": P["initial_train"],
        "cal": P["initial_calibration"],
        "val": P["threshold_windows"] * P["monitor_per_window"],
        "negative": P["negative_windows"] * P["monitor_per_window"],
        "stream": P["windows"] * (P["monitor_per_window"] + P["audit_per_window"]),
        "repair_pool": P["max_repairs"] * P["repair_block"],
    }
    return partition(len(d["base"]), sizes, seed)


def initial_for(d, name, seed, roles):
    archived = PROV / "raw" / f"offline_initial_{name}_{seed}.npz"
    if archived.exists():
        z = np.load(archived)
        return z["p0"], float(z["q0"]), z["tau0"]
    cache = CACHE / f"initial_{name}_{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["p0"], float(z["q0"]), z["tau0"]
    tr = roles["train"]
    params = dict(objective="regression", learning_rate=0.1, num_leaves=31, seed=seed,
                  deterministic=True, force_col_wise=True, num_threads=1, verbose=-1,
                  bagging_freq=0, feature_fraction=1.0)
    y = np.log(np.maximum(d["true"][tr], 1)) - np.log(np.maximum(d["base"][tr], 1))
    bst = lgb.train(params, lgb.Dataset(d["feat"][tr], label=y), num_boost_round=100)
    p0 = np.maximum(np.maximum(d["base"], 1) * np.exp(np.clip(bst.predict(d["feat"], num_threads=1), -30, 30)), 1)
    ca = roles["cal"]
    q0 = radius(np.abs(np.log1p(d["true"][ca]) - np.log1p(p0[ca])), P["alpha"])
    va = roles["val"]
    tau0 = R.thresholds(np.abs(np.log1p(d["true"][va]) - np.log1p(p0[va])), q0,
                        P["monitor_per_window"], P["target_window_fpr"])
    _SAVEZ(cache, p0=p0, q0=q0, tau0=tau0)
    return p0, float(q0), tau0


def archived_curves():
    """seed -> coverage for the archived ten seeds, per database and policy."""
    out = {}
    for path in [ROOT / "data" / "tables" / "offline_runs.csv",
                 ROOT / "data" / "continuous" / "offline_runs.csv"]:
        for r in csv.DictReader(path.open(encoding="utf-8")):
            if r["pattern"] != "cycles":
                continue
            out[(r["dataset"], int(r["seed"]), r["policy"])] = float(r["coverage"])
    return out


def ci(vals, rng, n=4000):
    vals = np.asarray(vals, dtype=float)
    boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    t0 = time.perf_counter()
    archived = archived_curves()
    rows = []
    for name in DBS:
        d = load_dataset(name)
        for seed in NEW:
            roles = roles_for(d, name, seed)
            p0, q0, tau0 = initial_for(d, name, seed, roles)
            for pol in POLICIES:
                row, _ = R.run_policy(d, P, roles, p0, q0, tau0, "cycles", pol, seed)
                rows.append({"dataset": name, "seed": seed, "policy": pol,
                             "coverage": round(row["coverage"], 6),
                             "labels": row["deployment_labels"], "repairs": row["repairs"]})
            print(f"{name:9s} seed={seed} done", flush=True)

    with (OUTDIR / "seed_extension.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"{'db':9s} {'n':>3s} {'cur':>7s} {'per':>7s} {'gain':>7s} {'ci':>18s}")
    rng = np.random.default_rng(11)
    for name in DBS:
        for label, seeds in (("10", OLD), ("30", OLD + NEW)):
            cur, per = [], []
            for s in seeds:
                if (name, s, "current_coverage") in archived:
                    cur.append(archived[(name, s, "current_coverage")])
                    per.append(archived[(name, s, "periodic")])
                else:
                    cur.append(next(r["coverage"] for r in rows
                                    if r["dataset"] == name and r["seed"] == s and r["policy"] == "current_coverage"))
                    per.append(next(r["coverage"] for r in rows
                                    if r["dataset"] == name and r["seed"] == s and r["policy"] == "periodic"))
            gain = 100 * (np.array(cur) - np.array(per))
            lo, hi = ci(gain, rng)
            print(f"{name:9s} {label:>3s} {100*np.mean(cur):7.2f} {100*np.mean(per):7.2f} "
                  f"{gain.mean():7.2f}  [{lo:6.2f},{hi:6.2f}]")
    print(f"wall {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
