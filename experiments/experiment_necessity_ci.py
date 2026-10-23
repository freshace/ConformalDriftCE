# -*- coding: utf-8 -*-
"""§4.1 必要性解耦实验的重复划分版（P4）：给"点精度 vs 可靠性"的关键数字补 95% CI。

【为什么需要】原 plot_fig_necessity_decouple.py 是**单次种子**（config.SEED=42）的一次读数，
且只把结果存到仓库根的 results_decouple.npy，**未注册进 run_all.py**（审计 B 的 MAJOR：
§4.1 的动机数字无法一键复现）。而漂移注入 drift.inject_correlation_drift 对给定 mean 是**确定性**的
（不使用 rng），所以唯一的不确定性来源是 **训练/校准/测试划分** —— 本脚本就重复划分 R 次。

【产出】
  results_repro/experiment_necessity_ci.txt   —— 关键数字的 mean ± 95%CI 与单次极差
  results_repro/results_decouple_mean.npy     —— R 次重复的均值曲线（列布局同 results_decouple.npy）
  results_repro/results_decouple_reps.npy     —— R 次重复的原始曲线（R × 7）
"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, data, model, conformal, drift

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = int(os.environ.get("NEC_R", "50"))


def gerr(t, e):
    e = np.clip(e, 1, None)
    return np.maximum(t / e, e / t)


def gmq(t, e):
    return float(np.exp(np.mean(np.log(gerr(t, e)))))


def ci95(a):
    a = np.asarray(a, dtype=np.float64)
    if len(a) < 2:
        return float(a.mean()), float("nan")
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def main():
    out = []
    def log(s):
        out.append(s); print(s, flush=True)

    base_, true_, feat_, nfc_ = data.load_stats(config.DATA_PATH)
    log(f"加载 {len(base_)} 条查询；R={R} 次独立重复（重抽 训练/校准/测试 划分）")
    log("漂移注入 drift.inject_correlation_drift 对给定 mean 是确定性的 -> 变异来源=划分")

    reps = []
    for r in range(R):
        rng = np.random.RandomState(config.SEED + r)
        (f_tr, b_tr, y_tr, _), (f_cal, b_cal, y_cal, _), (f_te, b_te, y_te, nfc_te) = \
            data.split(feat_, base_, true_, nfc_, config.N_TRAIN, config.N_CAL, config.N_TEST, rng)
        m = model.fit_corrector(f_tr, b_tr, y_tr, config.LGBM_PARAMS)
        q = conformal.calibrate_q(
            conformal.score(y_cal, b_cal, model.predict_corrector(m, f_cal)), config.ALPHA)
        corr = model.predict_corrector(m, f_te)
        est = b_te * np.exp(corr)
        multi = nfc_te >= 2
        row = []
        for mean in config.DRIFT_MEANS:
            yd = drift.inject_correlation_drift(y_te, nfc_te, mean, rng)
            sc = conformal.score(yd, b_te, corr)
            ge = gerr(yd[multi], est[multi])
            row.append([np.exp(mean), gmq(yd, est), float(np.mean(sc <= q)),
                        gmq(yd[multi], est[multi]), float(np.mean(sc[multi] <= q)),
                        float(np.percentile(ge, 95)), float(ge.max())])
        reps.append(row)
        if (r + 1) % 10 == 0 or r == R - 1:
            log(f"  ... replicate {r+1}/{R}")

    reps = np.array(reps)              # (R, len(DRIFT_MEANS), 7)
    mean_curve = reps.mean(axis=0)

    log("")
    log(f"== §4.1 关键数字（R={R} 次重复划分；mean ± 95%CI，方括号为单次读数极差）==")
    log("")
    k = 0            # 基线（无漂移）
    k2 = len(config.DRIFT_MEANS) - 1   # 最强漂移
    fac0, fac2 = mean_curve[k, 0], mean_curve[k2, 0]

    def line(name, col, scale=1.0, pct=False):
        m, h = ci95(reps[:, k, col] * scale)
        lo, hi = float((reps[:, k, col] * scale).min()), float((reps[:, k, col] * scale).max())
        f = "{:.1f}%" if pct else "{:.3g}"
        log(f"  [基线 {fac0:.2f}x] {name:<26} = {f.format(m)} ± {f.format(h)}  [{f.format(lo)}, {f.format(hi)}]")

    def line2(name, col, scale=1.0, pct=False):
        m, h = ci95(reps[:, k2, col] * scale)
        lo, hi = float((reps[:, k2, col] * scale).min()), float((reps[:, k2, col] * scale).max())
        f = "{:.1f}%" if pct else "{:.3g}"
        log(f"  [漂移 {fac2:.2f}x] {name:<26} = {f.format(m)} ± {f.format(h)}  [{f.format(lo)}, {f.format(hi)}]")

    line("GMQ 整体（点精度）", 1)
    line("GMQ 多列（敏感性）", 3)
    line("覆盖率 整体", 2, 100.0, pct=True)
    line("覆盖率 多列", 4, 100.0, pct=True)
    line("q-error p95（多列）", 5)
    line("q-error max（多列）", 6)
    log("")
    line2("GMQ 整体（点精度）", 1)
    line2("GMQ 多列（敏感性）", 3)
    line2("覆盖率 整体", 2, 100.0, pct=True)
    line2("覆盖率 多列", 4, 100.0, pct=True)
    line2("q-error p95（多列）", 5)
    log("")
    log("== 复现门：第 1 次重复（seed=42）应与原 results_decouple.npy 逐位一致 ==")
    log("  baseline(seed42): GMQ_all=%.2f  cov_all=%.1f%%  p95_multi=%.1f  max_multi=%.0f"
        % (reps[0, k, 1], reps[0, k, 2] * 100, reps[0, k, 5], reps[0, k, 6]))
    log("  7.39x  (seed42): cov_all=%.1f%%  cov_multi=%.1f%%  p95_multi=%.1f  max_multi=%.0f"
        % (reps[0, k2, 2] * 100, reps[0, k2, 4] * 100, reps[0, k2, 5], reps[0, k2, 6]))

    log("")
    log("== 覆盖率（多列）随漂移的完整曲线 ==")
    log(f"{'漂移因子':>10} | {'整体覆盖':>18} | {'多列覆盖':>18} | {'单次极差(多列)':>16}")
    log("-" * 72)
    for j, mean in enumerate(config.DRIFT_MEANS):
        a, ha = ci95(reps[:, j, 2] * 100); b, hb = ci95(reps[:, j, 4] * 100)
        sp = float((reps[:, j, 4] * 100).max() - (reps[:, j, 4] * 100).min())
        log(f"{mean_curve[j,0]:>9.2f}x | {a:>10.1f}% ± {ha:>4.1f} | {b:>10.1f}% ± {hb:>4.1f} | {sp:>14.1f}pp")

    OUT = os.path.join(ROOT, "results_repro")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "experiment_necessity_ci.txt"), "w", encoding="utf-8") as f:
        f.write("exit=0\n" + "\n".join(out) + "\n")
    np.save(os.path.join(OUT, "results_decouple_mean.npy"), mean_curve)
    np.save(os.path.join(OUT, "results_decouple_reps.npy"), reps)
    print("\nsaved results_repro/experiment_necessity_ci.txt + results_decouple_{mean,reps}.npy", flush=True)


if __name__ == "__main__":
    main()
