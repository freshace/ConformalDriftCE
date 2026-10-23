# -*- coding: utf-8 -*-
"""§4.2 相关性漂移的重复切分版（P4）：给"45.3% → 91.0%"这条头条数字补 95% CI。

【为什么需要】`experiment_correlation.py` 只有**一次** `RandomState(config.SEED)` 的划分，
论文 §4.2 的冻结/适应覆盖都是**单次读数**。本脚本重复 R 次划分（同时重抽 训练/校准/测试
与漂移后的 训练/校准/评估 三段），给出 mean ± 95%CI 与单次读数极差。

口径与 `experiment_correlation.py` **同源**：同 drift.inject_correlation_drift、
同 conformal.score/calibrate_q/coverage、同 N_TRAIN/N_CAL/N_TEST/N_ADAPT_*。
第 1 次重复（seed=42）即原脚本的那次读数 —— 用它做复现门。

【产出】results_repro/experiment_correlation_ci.txt + correlation_ci.npz
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, data, model, conformal, drift

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = int(os.environ.get("CORR_R", "50"))


def ci95(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return float(a.mean()), float("nan")
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def main():
    out = []
    def log(s):
        out.append(s); print(s, flush=True)

    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    log(f"加载 {len(true)} 条查询；R={R} 次独立重复划分（重抽 训练/校准/测试 与 漂移后三段）")
    log("口径与 experiment_correlation.py 同源；第 1 次重复(seed=42)应复现其单次读数")

    recs = {k: [] for k in ("cov_base", "feat0", "multi_g0", "single_g0")}
    per_mean = {m: {k: [] for k in ("cov_all", "cov_multi", "cov_multi_adapt", "cov_single",
                                    "q_frozen", "q_new")} for m in config.DRIFT_MEANS}

    for r in range(R):
        rng = np.random.RandomState(config.SEED + r)
        (f_tr, b_tr, y_tr, _), (f_cal, b_cal, y_cal, _), (f_te, b_te, y_te, nfc_te) = \
            data.split(feat, base, true, nfc, config.N_TRAIN, config.N_CAL, config.N_TEST, rng)

        m_old = model.fit_corrector(f_tr, b_tr, y_tr, config.LGBM_PARAMS)
        q_old = conformal.calibrate_q(
            conformal.score(y_cal, b_cal, model.predict_corrector(m_old, f_cal)), config.ALPHA)
        c_old_te = model.predict_corrector(m_old, f_te)
        recs["cov_base"].append(conformal.coverage(conformal.score(y_te, b_te, c_old_te), q_old))
        if r == 0:
            log("")
            log(f"[seed=42 单次] 无漂移 q={q_old:.3f} (区间 {np.exp(q_old):.1f}x) "
                f"覆盖={recs['cov_base'][0]:.1%}")

        single = nfc_te <= 1
        multi = nfc_te >= 2
        if r == 0:
            log(f"[seed=42 单次] 测试集 单列 {int(single.sum())} 条 / 多列 {int(multi.sum())} 条")

        for mean in config.DRIFT_MEANS:
            y_d = drift.inject_correlation_drift(y_te, nfc_te, mean, rng)
            sc = conformal.score(y_d, b_te, c_old_te)
            per_mean[mean]["cov_all"].append(conformal.coverage(sc, q_old))
            per_mean[mean]["cov_multi"].append(conformal.coverage(sc[multi], q_old))
            per_mean[mean]["cov_single"].append(conformal.coverage(sc[single], q_old))
            per_mean[mean]["q_frozen"].append(float(np.exp(q_old)))

            # 适应：抽 训练/校准/评估 三段。
            # 注意：这里**刻意沿用与原脚本相同的 rng 消耗方式**（在 mean 循环内顺序调用
            # rng.permutation），这样第 1 次重复(seed=42)能与 experiment_correlation.py
            # 的单次读数**逐位对齐**（含"多列适应"列），复现门才完整。
            perm = rng.permutation(len(f_te))
            a_tr = perm[:config.N_ADAPT_TRAIN]
            a_cal = perm[config.N_ADAPT_TRAIN:config.N_ADAPT_TRAIN + config.N_ADAPT_CAL]
            a_ev = perm[config.N_ADAPT_TRAIN + config.N_ADAPT_CAL:]
            m_new = model.fit_corrector(f_te[a_tr], b_te[a_tr], y_d[a_tr], config.LGBM_PARAMS)
            q_new = conformal.calibrate_q(
                conformal.score(y_d[a_cal], b_te[a_cal], model.predict_corrector(m_new, f_te[a_cal])),
                config.ALPHA)
            a_ev_m = a_ev[nfc_te[a_ev] >= 2]
            per_mean[mean]["cov_multi_adapt"].append(conformal.coverage(
                conformal.score(y_d[a_ev_m], b_te[a_ev_m], model.predict_corrector(m_new, f_te[a_ev_m])),
                q_new))
            per_mean[mean]["q_new"].append(float(np.exp(q_new)))
        if (r + 1) % 10 == 0 or r == R - 1:
            log(f"  ... replicate {r+1}/{R}")

    mb, hb = ci95(recs["cov_base"])
    log("")
    log(f"== §4.2 相关性漂移（R={R} 次独立重复；mean ± 95%CI，方括号为单次读数范围）==")
    log(f"  无漂移整体覆盖 = {mb*100:.1f}% ± {hb*100:.1f}%")
    log("")
    hdr = (f"{'相关因子':>9} | {'整体(冻结)':>20} | {'多列(冻结)':>20} | {'多列(适应)':>20} | "
           f"{'单列(冻结)':>18}")
    log(hdr); log("-" * len(hdr))
    for mean in config.DRIFT_MEANS:
        g = float(np.exp(mean)); d = per_mean[mean]

        def cell(k, pct=True):
            a = np.array(d[k], dtype=float) * (100.0 if pct else 1.0)
            m, h = ci95(a)
            return f"{m:6.1f}±{h:4.1f} [{a.min():.1f},{a.max():.1f}]" if pct else f"{m:6.2f}±{h:4.2f}"
        log(f"{g:>8.2f}x | {cell('cov_all'):>20} | {cell('cov_multi'):>20} | "
            f"{cell('cov_multi_adapt'):>20} | {cell('cov_single'):>18}")

    log("")
    log("== 单次读数极差（不是 CI，是该指标在 R 次划分里的 max−min）==")
    for mean in config.DRIFT_MEANS:
        g = float(np.exp(mean)); d = per_mean[mean]
        sp = [float(np.ptp(np.array(d[k]) * 100)) for k in ("cov_all", "cov_multi", "cov_multi_adapt")]
        log(f"  {g:5.2f}x  整体 {sp[0]:5.1f}pp | 多列冻结 {sp[1]:5.1f}pp | 多列适应 {sp[2]:5.1f}pp")

    log("")
    log("== 复现门：第 1 次重复(seed=42) 应与 experiment_correlation.py 的单次读数一致 ==")
    for mean in config.DRIFT_MEANS:
        g = float(np.exp(mean)); d = per_mean[mean]
        log(f"  {g:5.2f}x  整体 {d['cov_all'][0]*100:5.1f}% | 多列冻结 {d['cov_multi'][0]*100:5.1f}% | "
            f"多列适应 {d['cov_multi_adapt'][0]*100:5.1f}% | 单列冻结 {d['cov_single'][0]*100:5.1f}%")

    log("")
    log("== 区间宽度（倍，越小越好）==")
    for mean in config.DRIFT_MEANS:
        g = float(np.exp(mean)); d = per_mean[mean]
        f_, hf = ci95(d["q_frozen"]); n_, hn = ci95(d["q_new"])
        log(f"  {g:5.2f}x  冻结 {f_:9.2f}±{hf:5.2f}  →  适应 {n_:8.2f}±{hn:5.2f}")

    OUT = os.path.join(ROOT, "results_repro")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "experiment_correlation_ci.txt"), "w", encoding="utf-8") as f:
        f.write("exit=0\n" + "\n".join(out) + "\n")
    np.savez(os.path.join(OUT, "correlation_ci.npz"),
             means=np.array(config.DRIFT_MEANS),
             **{f"{k}__{m}": np.array(v) for m, dd in per_mean.items() for k, v in dd.items()},
             cov_base=np.array(recs["cov_base"]), R=R)
    print("\nsaved results_repro/experiment_correlation_ci.txt", flush=True)


if __name__ == "__main__":
    main()
