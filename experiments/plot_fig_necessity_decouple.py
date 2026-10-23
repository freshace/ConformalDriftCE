# -*- coding: utf-8 -*-
"""正文 §4.2.1 图 fig:necessity：点精度与可靠性在相关性漂移下的背离。

数据源（只读，不训练、不重算实验）：
    results_repro/results_decouple_mean.npy   50 次划分的均值曲线
        列 = [漂移因子, GMQ整体, 覆盖率整体, GMQ多列, 覆盖率多列, q-error p95(多列), max(多列)]
    results_repro/results_decouple_reps.npy   同列，但保留 50 次划分的逐次读数 (50, 6, 7)

    （本脚本原先自己 load 数据、fit 修正器、重算曲线并回写 results_decouple*.npy，
      现改为只读既有产物；数字仍与 results_repro/experiment_necessity_ci.txt 逐位一致。）

面板：
    (a) 点精度 GMQ（对数轴）：整体只从 2.36 涨到 5.11，看起来"没坏多少"。
    (b) 覆盖率：同一次漂移下多列敏感子集由 86.6% 塌到 46.7%。
    两图并置即"平均点误差对严重失效不敏感"的证据。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vldb_style as vs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEAN = os.path.join(ROOT, "results_repro", "results_decouple_mean.npy")
REPS = os.path.join(ROOT, "results_repro", "results_decouple_reps.npy")


def main():
    vs.apply()
    m = np.load(MEAN)
    reps = np.load(REPS)
    fac = m[:, 0]

    gmq_all, gmq_mul = m[:, 1], m[:, 3]
    cov_all, cov_mul = m[:, 2] * 100, m[:, 4] * 100
    # 最坏查询 q-error：单次读数跨 4 个数量级（range [1.7e3, 6.5e5]），均值会被离群划分
    # 拉高，故正文与图注引用的是 50 次划分上的中位数。
    worst_med = float(np.median(reps[:, 0, 6]))

    a = vs.Assertor()
    a.approx("漂移因子(基线)", fac[0], 1.0, 1e-6)
    a.approx("漂移因子(最大)", fac[-1], 7.3891, 1e-3)
    for i, (ga, gm) in enumerate(zip((2.36, 2.45, 2.74, 3.08, 3.90, 5.11),
                                     (2.84, 3.01, 3.56, 4.24, 6.04, 9.05))):
        a.approx(f"{fac[i]:.2f}× GMQ 整体", gmq_all[i], ga, 0.006)
        a.approx(f"{fac[i]:.2f}× GMQ 多列", gmq_mul[i], gm, 0.006)
    for i, (ca, cm) in enumerate(zip((90.1, 89.6, 87.3, 84.3, 76.0, 63.5),
                                     (86.6, 85.8, 82.4, 77.9, 65.4, 46.7))):
        a.approx(f"{fac[i]:.2f}× 覆盖 整体", cov_all[i], ca, 0.05)
        a.approx(f"{fac[i]:.2f}× 覆盖 多列", cov_mul[i], cm, 0.05)
    a.approx("无漂移 最坏 q-error 中位数", worst_med / 1e4, 2.8, 0.05)
    a.assert_all("fig_necessity_decouple")

    fig, axes = vs.subplots(1, 2, width=vs.COL, height=1.78, sharex=True)
    series = ((gmq_all, cov_all, vs.C_FIFTH, "^", "整体"),
              (gmq_mul, cov_mul, vs.C_ALT, "o", "多列敏感子集"))

    ax = axes[0]
    for (gm, _, color, mk, lab), col in zip(series, (1, 3)):
        half = 1.96 * reps[:, :, col].std(axis=0, ddof=1) / np.sqrt(len(reps))
        ax.errorbar(fac, gm, yerr=half, color=color, marker=mk, label=lab,
                    capsize=1.4, elinewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylim(2.0, 11)
    ax.set_yticks([2, 3, 5, 10])
    ax.set_yticklabels(["2", "3", "5", "10"])
    ax.set_ylabel("GMQ（对数轴）")
    ax.set_title("(a) 点精度", fontsize=7.5, pad=3)

    ax = axes[1]
    for (_, cv, color, mk, lab), col in zip(series, (2, 4)):
        half = 100 * 1.96 * reps[:, :, col].std(axis=0, ddof=1) / np.sqrt(len(reps))
        ax.errorbar(fac, cv, yerr=half, color=color, marker=mk, label=lab,
                    capsize=1.4, elinewidth=0.6)
    vs.refline(ax, 90)
    ax.set_ylim(40, 100)
    ax.set_ylabel("覆盖率 (%)")
    ax.set_title("(b) 可靠性", fontsize=7.5, pad=3)

    for ax in axes:
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks([1, 2, 4, 7.39])
        ax.set_xticklabels(["1$\\times$", "2$\\times$", "4$\\times$", "7.39$\\times$"])
        ax.set_xlim(0.95, 8.3)
        ax.set_xlabel("相关性漂移因子")

    fig.subplots_adjust(wspace=0.42)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.16))
    outdir = os.environ.get("DRIFT_FIG_OUT")
    if outdir and os.environ.get("DRIFT_FIG_QA_PATH"):
        sys.path.insert(0, os.environ["DRIFT_FIG_QA_PATH"])
        from audit_panel_alignment import require_matplotlib_panel_alignment
        os.makedirs(outdir, exist_ok=True)
        require_matplotlib_panel_alignment(fig, json_out=os.path.join(outdir, "fig_necessity_decouple.alignment.json"))
    vs.save(fig, "fig_necessity_decouple", outdir=outdir)


if __name__ == "__main__":
    main()
