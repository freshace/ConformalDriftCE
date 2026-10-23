# -*- coding: utf-8 -*-
"""分组覆盖的重复划分版（P2 闸门 G2）。

【为什么需要】`experiment_groupcoverage.py` 只用了单次 `RandomState(config.SEED)` 的**一次划分**，
因此论文头条"组间极差 29.6pp → 6.0pp"**没有置信区间**。本脚本重复 R 次划分，
给出 mean ± 95%CI 与单次读数的极差。

【口径保证】直接复用 `experiment_groupcoverage` 的 `load_all / calib_groups / summarize`
与常量 `MIN_N / MIN_TEST / GROUP_KEYS`，因此统计量与单次版**逐位同源**（唯一差别是划分）。
第 1 次重复（seed=42）应与 `results_groupcov.npy` 一致。

【产出】results_repro/experiment_groupcoverage_ci.txt  +  groupcov_ci.npz
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import config, model, conformal, drift
from experiment_groupcoverage import (load_all, calib_groups, summarize,
                                      MIN_N, MIN_TEST, GROUP_KEYS)

ROOT = os.path.dirname(HERE)
R = int(os.environ.get("GC_R", "50"))


def ci95(a):
    a = np.asarray(a, dtype=np.float64)
    if len(a) < 2:
        return float(a.mean()), float("nan")
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def main():
    out = []
    def log(s):
        out.append(s); print(s, flush=True)

    alpha = config.ALPHA
    base, true, feat, nfc, ntab = load_all(config.DATA_PATH)
    n = len(true)
    log(f"加载 {n} 条 (MIN_N={MIN_N}, MIN_TEST={MIN_TEST}, alpha={alpha});R={R} 次独立重复划分")
    log("口径与 experiment_groupcoverage.py 完全同源（复用其 load_all/calib_groups/summarize）")

    # 每次重复记录: [(setting, key)] -> dict of metrics
    acc = {}
    for r in range(R):
        rng = np.random.RandomState(config.SEED + r)
        idx = rng.permutation(n)
        tr = idx[:config.N_TRAIN]
        ca = idx[config.N_TRAIN:config.N_TRAIN + config.N_CAL]
        te = idx[config.N_TRAIN + config.N_CAL:config.N_TRAIN + config.N_CAL + config.N_TEST]

        m_old = model.fit_corrector(feat[tr], base[tr], true[tr], config.LGBM_PARAMS)
        y_drift = drift.inject_correlation_drift(true[te], nfc[te], 2.0, rng)

        perm = rng.permutation(len(te))
        a_tr = perm[:config.N_ADAPT_TRAIN]
        a_ca = perm[config.N_ADAPT_TRAIN:config.N_ADAPT_TRAIN + config.N_ADAPT_CAL]
        a_ev = perm[config.N_ADAPT_TRAIN + config.N_ADAPT_CAL:]
        m_new = model.fit_corrector(feat[te][a_tr], base[te][a_tr], y_drift[a_tr], config.LGBM_PARAMS)

        s_cal_ab = conformal.score(true[ca], base[ca], model.predict_corrector(m_old, feat[ca]))
        s_ev_a = conformal.score(true[te], base[te], model.predict_corrector(m_old, feat[te]))
        s_ev_b = conformal.score(y_drift, base[te], model.predict_corrector(m_old, feat[te]))
        s_cal_c = conformal.score(y_drift[a_ca], base[te][a_ca],
                                  model.predict_corrector(m_new, feat[te][a_ca]))
        s_ev_c = conformal.score(y_drift[a_ev], base[te][a_ev],
                                 model.predict_corrector(m_new, feat[te][a_ev]))

        settings = [
            ("A 无漂移", s_cal_ab, nfc[ca], ntab[ca], s_ev_a, nfc[te], ntab[te]),
            ("B 7.39x(冻结)", s_cal_ab, nfc[ca], ntab[ca], s_ev_b, nfc[te], ntab[te]),
            ("C 7.39x(适应)", s_cal_c, nfc[te][a_ca], ntab[te][a_ca], s_ev_c,
             nfc[te][a_ev], ntab[te][a_ev]),
        ]
        for title, s_cal, gc_nfc, gc_ntab, s_ev, ge_nfc, ge_ntab in settings:
            gcal = {"nfc": np.asarray(gc_nfc), "ntab": np.asarray(gc_ntab)}
            gev = {"nfc": np.asarray(ge_nfc), "ntab": np.asarray(ge_ntab)}
            for _, key in GROUP_KEYS:
                gl, qs = calib_groups(s_cal, gcal[key], alpha, MIN_N)
                Sm = summarize(s_ev, gev[key], gl, qs)
                acc.setdefault((title, key), []).append(dict(
                    spread_glob=Sm["spread_glob"] * 100, spread_grp=Sm["spread_grp"] * 100,
                    min_glob=Sm["min_glob"] * 100, min_grp=Sm["min_grp"] * 100,
                    cov_all_glob=Sm["cov_all_glob"] * 100, cov_all_grp=Sm["cov_all_grp"] * 100,
                    w_all_glob=Sm["w_all_glob"], w_all_grp=Sm["w_all_grp"],
                    gl=gl, n_fallback=len([r_ for r_ in Sm["rows"] if r_["fallback"]]),
                    n_groups=len(Sm["rows"])))
        if (r + 1) % 10 == 0 or r == R - 1:
            log(f"  ... replicate {r+1}/{R}")

    # ---- 汇总 ----
    log("")
    log(f"== 分组(Mondrian)覆盖:R={R} 次独立重复划分,mean ± 95%CI,方括号为单次读数极差 ==")
    log("")
    for title, _ in [("A 无漂移", "ntab"), ("B 7.39x(冻结)", "ntab"), ("C 7.39x(适应)", "ntab")]:
        for label, key in GROUP_KEYS:
            recs = acc[(title, key)]
            log(f"--- {title}  |  按 {label} ---")
            log(f"{'指标':<26} {'全局 q':>22} | {'分组 q (Mondrian)':>26}")
            log("-" * 80)
            for name, k in (("组间极差 (pp)", "spread"), ("最差组覆盖 (%)", "min"),
                            ("整体覆盖 (%)", "cov_all"), ("平均宽度 (×)", "w_all")):
                for arm in ("glob", "grp"):
                    a = np.array([x[f"{k}_{arm}"] for x in recs])
                    m, h = ci95(a)
                    lo, hi = float(a.min()), float(a.max())
                    if arm == "glob":
                        s_glob = f"{m:8.2f} ± {h:5.2f} [{lo:.1f}, {hi:.1f}]"
                    else:
                        s_grp = f"{m:8.2f} ± {h:5.2f} [{lo:.1f}, {hi:.1f}]"
                log(f"{name:<26} {s_glob:>22} | {s_grp:>26}")
            fb = np.array([x["n_fallback"] for x in recs], dtype=float)
            ng = np.array([x["n_groups"] for x in recs], dtype=float)
            log(f"{'回退组数 / 计分组数':<24} {fb.mean():>22.2f} | 分组内一致 ({ng.mean():.1f} 组计分)")
            log("")

    log("== 论文头条数字的 CI（无漂移，按 join 宽度）==")
    recs = acc[("A 无漂移", "ntab")]
    for k, name in (("spread_glob", "组间极差 全局 q"), ("spread_grp", "组间极差 分组 q"),
                    ("min_glob", "最差组 全局 q"), ("min_grp", "最差组 分组 q")):
        a = np.array([x[k] for x in recs]); m, h = ci95(a)
        log(f"  {name:<22} = {m:6.2f} ± {h:4.2f}   [{a.min():.1f}, {a.max():.1f}]")
    a = np.array([x["w_all_grp"] - x["w_all_glob"] for x in recs]); m, h = ci95(a)
    log(f"  {'平均宽度变化(分组−全局)':<20} = {m:+6.2f} ± {h:4.2f} ×  [{a.min():+.2f}, {a.max():+.2f}]")

    log("")
    log("== 复现门:第 1 次重复(seed=42)应与单次版 results_groupcov.npy 一致 ==")
    r0 = acc[("A 无漂移", "ntab")][0]
    log("  seed42: 组间极差 全局 %.1fpp / 分组 %.1fpp ; 最差组 全局 %.1f%% / 分组 %.1f%% ; 整体 %.1f%%/%.1f%% ; 宽 %.2fx/%.2fx"
        % (r0["spread_glob"], r0["spread_grp"], r0["min_glob"], r0["min_grp"],
           r0["cov_all_glob"], r0["cov_all_grp"], r0["w_all_glob"], r0["w_all_grp"]))

    OUT = os.path.join(ROOT, "results_repro")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "experiment_groupcoverage_ci.txt"), "w", encoding="utf-8") as f:
        f.write("exit=0\n" + "\n".join(out) + "\n")
    keys = sorted({k for k in acc})
    arr = np.array([[acc[k][i]["spread_glob"] for k in keys] for i in range(R)], dtype=np.float64)
    np.savez(os.path.join(OUT, "groupcov_ci.npz"),
             keys=np.array([f"{a}|{b}" for a, b in keys]), R=R, spread_glob=arr)
    print("\nsaved results_repro/experiment_groupcoverage_ci.txt", flush=True)


if __name__ == "__main__":
    main()
