# -*- coding: utf-8 -*-
"""§4.10 消融的重复切分版（P4）：给"标签量 100/200/500/1000 → 90.1/93.1/91.2/90.6（已饱和）"
与"树配置 88.7–91.0%（不敏感）"这两条结论补 95% CI。

【为什么需要】`experiment_ablation.py` 只有一次 `RandomState(config.SEED)` 划分，论文里
消融 C/D 的数字都是**单次读数**。

【复现门怎么保证】本脚本**按原脚本完全相同的顺序**依次跑 消融 A(2 次) → B(2) → C(4) → D(4)，
每次都像原脚本那样消耗一次 `rng.permutation`。因此第 1 次重复(seed=42)与原脚本
**逐位一致**（A/B 也一并记录，顺带补上它们的 CI）。

【产出】results_repro/experiment_ablation_ci.txt + ablation_ci.npz
"""
import os, sys, warnings, importlib.util
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import config, features, model, conformal
# 复用原脚本的提取器与 run_all，保证口径同源
_spec = importlib.util.spec_from_file_location("abl", os.path.join(HERE, "experiment_ablation.py"))
abl = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(abl)

ROOT = os.path.dirname(HERE)
R = int(os.environ.get("ABL_R", "20"))


def ci95(a):
    a = np.asarray(a, dtype=np.float64); a = a[np.isfinite(a)]
    if len(a) < 2:
        return float(a.mean()), float("nan")
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def main():
    out = []
    def log(s):
        out.append(s); print(s, flush=True)

    n_ts, n_cs = features.load_table_col_space(config.DATA_PATH)
    import pickle
    with open(config.DATA_PATH, "rb") as f:
        data = pickle.load(f)
    rows = []
    for ft, label, pgv, *_, ex in data["features"]:
        t = float(np.expm1(label - 1)); p = float(np.expm1(pgv - 1))
        if t <= 0 or p <= 0:
            continue
        rows.append((ft, ex, len(set(ex.get("filter_col_identifiers", [])))))
    true = np.array([float(np.expm1(l - 1)) for ft, l, p, *_, ex in data["features"]])
    pg = np.array([float(np.expm1(p - 1)) for ft, l, p, *_, ex in data["features"]])
    mask = np.array([(float(np.expm1(l - 1)) > 0 and float(np.expm1(p - 1)) > 0)
                     for ft, l, p, *_, ex in data["features"]])
    true = true[mask]; pg = pg[mask]
    rows = [rows[i] for i in np.arange(len(rows))[mask]]
    nfc = np.array([r[2] for r in rows], dtype=np.int32)

    log(f"loaded {len(true)} queries；R={R} 次独立重复划分")
    log("调用顺序与原 experiment_ablation.py 完全一致 ⇒ 第 1 次重复(seed=42)应逐位复现")
    F = np.array([abl.extract_full(r[0], p, r[1], n_ts, n_cs) for r, p in zip(rows, pg)], dtype=np.float32)
    G = np.array([abl.extract_s9(r[0], p, r[1], n_ts, n_cs) for r, p in zip(rows, pg)], dtype=np.float32)
    log(f"F dim {F.shape[1]} ; G dim {G.shape[1]}")

    P = config.LGBM_PARAMS
    A_LABELS = ["base-only", "+LGBM"]
    B_LABELS = ["s9-only(10d)", "full(622d)"]
    C_NS = [100, 200, 500, 1000]
    D_CFG = [("100t", dict(P, n_estimators=100)), ("300t", dict(P)),
             ("300t127l", dict(P, num_leaves=127)), ("1000t", dict(P, n_estimators=1000))]

    acc = {}
    def rec(key, fr, ad, wq):
        acc.setdefault(key, {"frozen": [], "adapt": [], "width": []})
        acc[key]["frozen"].append(fr); acc[key]["adapt"].append(ad); acc[key]["width"].append(wq)

    for r in range(R):
        rng = np.random.RandomState(config.SEED + r)
        n = len(true)
        idx = rng.permutation(n)
        tr = idx[:config.N_TRAIN]
        cal = idx[config.N_TRAIN:config.N_TRAIN + config.N_CAL]
        te = idx[config.N_TRAIN + config.N_CAL:config.N_TRAIN + config.N_CAL + config.N_TEST]
        # —— 顺序必须与原脚本一致：A → B → C → D ——
        for uc, lab in zip([False, True], A_LABELS):
            fr, ad, wq, _ = abl.run_all(F, tr, cal, te, pg, true, nfc, P, uc, config.N_ADAPT_TRAIN, rng)
            rec(("A", lab), fr, ad, wq)
        for Xg, lab in zip([G, F], B_LABELS):
            fr, ad, wq, _ = abl.run_all(Xg, tr, cal, te, pg, true, nfc, P, True, config.N_ADAPT_TRAIN, rng)
            rec(("B", lab), fr, ad, wq)
        for N in C_NS:
            fr, ad, wq, _ = abl.run_all(F, tr, cal, te, pg, true, nfc, P, True, N, rng)
            rec(("C", N), fr, ad, wq)
        for name, prm in D_CFG:
            fr, ad, wq, _ = abl.run_all(F, tr, cal, te, pg, true, nfc, prm, True, config.N_ADAPT_TRAIN, rng)
            rec(("D", name), fr, ad, wq)
        if (r + 1) % 5 == 0 or r == R - 1:
            log(f"  ... replicate {r+1}/{R}")

    def blk(title, part, keys):
        log(""); log(f"== 消融 {part}: {title}（R={R} 次；mean ± 95%CI，方括号为单次读数范围）==")
        log(f"{'配置':<22} {'冻结覆盖':>22} {'适应后覆盖':>22} {'适应后宽度':>20}")
        log("-" * 92)
        for k in keys:
            d = acc[(part, k)]
            def cell(name, pct=True):
                a = np.array(d[name], dtype=float) * (100.0 if pct else 1.0)
                m, h = ci95(a)
                return f"{m:6.1f}±{h:4.1f} [{a.min():.1f},{a.max():.1f}]" if pct else f"{m:6.2f}±{h:4.2f}"
            log(f"{str(k):<22} {cell('frozen'):>22} {cell('adapt'):>22} {cell('width', False):>20}")

    blk("校正器有无", "A", A_LABELS)
    blk("特征维度", "B", B_LABELS)
    blk("自适应标签量", "C", C_NS)
    blk("LightGBM 树配置", "D", [k for k, _ in D_CFG])

    log("")
    log("== 复现门：第 1 次重复(seed=42) 应与 experiment_ablation.py 的单次读数一致 ==")
    for part, keys in (("A", A_LABELS), ("B", B_LABELS), ("C", C_NS), ("D", [k for k, _ in D_CFG])):
        for k in keys:
            d = acc[(part, k)]
            log(f"  {part}/{str(k):<12} frozen {d['frozen'][0]*100:6.1f}% | adapt {d['adapt'][0]*100:6.1f}% | "
                f"width {d['width'][0]:6.1f}x")

    log("")
    log("== 论文口径的两条结论 ==")
    ca = [np.mean(acc[("C", N)]["adapt"]) * 100 for N in C_NS]
    cb = [np.mean(acc[("C", N)]["frozen"]) * 100 for N in C_NS]
    log("  消融 C 适应后覆盖（N=100/200/500/1000）：" + " / ".join(f"{x:.1f}%" for x in ca))
    log("  消融 C 冻结覆盖（同序）：" + " / ".join(f"{x:.1f}%" for x in cb))
    da = [np.mean(acc[("D", k)]["adapt"]) * 100 for k, _ in D_CFG]
    log("  消融 D 适应后覆盖：" + " / ".join(f"{x:.1f}%" for x in da)
        + f"  ⇒ 极差 {max(da)-min(da):.1f}pp")

    OUT = os.path.join(ROOT, "results_repro")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "experiment_ablation_ci.txt"), "w", encoding="utf-8") as f:
        f.write("exit=0\n" + "\n".join(out) + "\n")
    np.savez(os.path.join(OUT, "ablation_ci.npz"),
             keys=np.array([f"{p}|{k}" for (p, k) in acc]),
             frozen=np.array([acc[k]["frozen"] for k in acc]),
             adapt=np.array([acc[k]["adapt"] for k in acc]),
             width=np.array([acc[k]["width"] for k in acc]), R=R)
    print("\nsaved results_repro/experiment_ablation_ci.txt", flush=True)


if __name__ == "__main__":
    main()
