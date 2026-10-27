# -*- coding: utf-8 -*-
"""Period-sweep of the periodic baseline for the 80-window recurring-drift replay.

Two sweeps, both on the recurring ("cycles") trajectory only:

  A  budget fixed: ten repairs, phase shifted by delta windows (0, 2, 4, 6, 8)
  B  budget varies: n evenly spaced repairs (n = 4, 6, 8, 10)

The current-coverage policy is not re-run; its per-seed coverage is read from the
archived result tables, so every comparison is paired within seed.

Outputs (data/period_sweep/):
  period_runs.csv   one row per (database, seed, sweep, setting)
  summary.csv       per (database, sweep, setting): mean coverage, labels, paired gain
  summary.json      post-hoc best setting per database, plus the envelope
  cache/            initial predictors for databases without an archived npz
"""
import csv, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "provenance" / "2026-10-12" / "code"
PROV = ROOT / "provenance" / "2026-10-12"
OUTDIR = ROOT / "data" / "period_sweep"
CACHE = OUTDIR / "cache"
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE))

import numpy as np                                   # noqa: E402
import lightgbm as lgb                               # noqa: E402
import run_offline as R                              # noqa: E402
from core import radius, partition                   # noqa: E402

_SAVEZ = np.savez_compressed
R.np.savez_compressed = lambda *a, **k: None         # never write into provenance/raw

PROTOCOL = json.loads((PROV / "protocol" / "offline.json").read_text(encoding="utf-8"))
P = dict(PROTOCOL)
DATABASES = ["stats", "imdb", "ergastf1", "genome"]
SEEDS = list(P["seeds"])
W = P["windows"]
BASE = np.rint(np.linspace(6, W - 4, P["max_repairs"])).astype(int)


def settings():
    """(sweep, label, schedule, repairs) for every candidate."""
    out = []
    for d in (0, 2, 4, 6, 8):
        idx = np.clip(BASE + d, 1, W - 2)
        out.append(("A_phase", f"A:10 repairs, phase+{d}", sorted(set(idx.tolist())), 10))
    # circular phase shift: exactly ten repairs, constant spacing, no edge clipping
    span = W - 4 - 6
    for d in (0, 2, 4, 6, 8):
        idx = np.rint(6 + ((BASE - 6 + d) % span)).astype(int)
        out.append(("A_wrap", f"A_wrap:10 repairs, phase+{d}", sorted(set(idx.tolist())), 10))
    for n in (4, 6, 8, 10):
        idx = np.rint(np.linspace(6, W - 4, n)).astype(int)
        out.append(("B_count", f"B:{n} repairs", sorted(set(idx.tolist())), n))
    return out


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
    """Archived initial predictor, or a deterministic refit of the same recipe."""
    path = PROV / "raw" / f"offline_initial_{name}_{seed}.npz"
    if path.exists():
        z = np.load(path)
        return z["p0"], float(z["q0"]), z["tau0"], "archived"
    cache = CACHE / f"initial_{name}_{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["p0"], float(z["q0"]), z["tau0"], "cache"
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
    return p0, float(q0), tau0, "refit"


def archived_current():
    """Per-seed current-coverage results, read from the archived tables."""
    rows = {}
    for path in [ROOT / "data" / "tables" / "offline_runs.csv",
                 ROOT / "data" / "continuous" / "offline_runs.csv"]:
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["pattern"] != "cycles" or r["policy"] != "current_coverage":
                    continue
                rows[(r["dataset"], int(r["seed"]))] = float(r["coverage"])
    return rows


def main():
    t0 = time.perf_counter()
    current = archived_current()
    specs = settings()
    runs, summary = [], []
    for name in DATABASES:
        d = load_dataset(name)
        for seed in SEEDS:
            roles = roles_for(d, name, seed)
            p0, q0, tau0, source = initial_for(d, name, seed, roles)
            for sweep, label, idx, nrep in specs:
                row, _ = R.run_policy(d, P, roles, p0, q0, tau0, "cycles", "periodic", seed,
                                      schedule={"periodic": idx})
                runs.append({
                    "dataset": name, "seed": seed, "sweep": sweep, "setting": label,
                    "repairs": row["repairs"], "labels": row["deployment_labels"],
                    "coverage": round(row["coverage"], 6), "width": round(row["width"], 6),
                    "current_coverage": round(current[(name, seed)], 6),
                    "gain_pp": round(100 * (current[(name, seed)] - row["coverage"]), 6),
                    "initial_source": source,
                })
                print(f"{name:9s} seed={seed} {label:28s} cov={row['coverage']:.4f} "
                      f"labels={row['deployment_labels']} [{source}]", flush=True)
        # per-database summary
        for sweep, label, idx, nrep in specs:
            sel = [r for r in runs if r["dataset"] == name and r["setting"] == label]
            cov = np.array([r["coverage"] for r in sel])
            gain = np.array([r["gain_pp"] for r in sel])
            rng = np.random.default_rng(7)
            boot = [gain[rng.integers(0, len(gain), len(gain))].mean() for _ in range(2000)]
            summary.append({
                "dataset": name, "sweep": sweep, "setting": label, "repairs": nrep,
                "labels": int(np.mean([r["labels"] for r in sel])),
                "mean_coverage": float(cov.mean()),
                "mean_gain_pp": float(gain.mean()),
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
            })

    with (OUTDIR / "period_runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)
    with (OUTDIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    best = {}
    for name in DATABASES:
        sub = [s for s in summary if s["dataset"] == name]
        for sweep in ("A_phase", "B_count"):
            ss = [s for s in sub if s["sweep"] == sweep]
            top = max(ss, key=lambda s: s["mean_coverage"])
            worst = min(ss, key=lambda s: s["mean_coverage"])
            best.setdefault(name, {})[sweep] = {
                "best_setting": top["setting"], "best_coverage": top["mean_coverage"],
                "best_gain_pp": top["mean_gain_pp"],
                "ci": [top["ci_low"], top["ci_high"]],
                "envelope": [worst["mean_coverage"], top["mean_coverage"]],
            }
    (OUTDIR / "summary.json").write_text(json.dumps({
        "protocol": PROTOCOL["version"], "trajectory": "cycles", "seeds": SEEDS,
        "settings": [s[1] for s in specs], "best": best,
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"done in {time.perf_counter()-t0:.1f}s -> {OUTDIR}")


if __name__ == "__main__":
    main()
