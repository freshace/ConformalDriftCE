# -*- coding: utf-8 -*-
"""正文 §4.2.1 图 fig1_correlation_drift：相关性漂移下的覆盖率下降与适应恢复。

数据源（只读）：results_repro/correlation_ci.npz
    R=50 次独立划分；列说明：cov_all__<lv> 全局、cov_multi__<lv> 多列敏感子集（冻结）、
    cov_multi_adapt__<lv> 多列敏感子集（500 标签适应后）、cov_single__<lv> 单列子集。
    lv 为对数漂移参数，因子 = exp(lv)。

绘图前对每个读数做硬断言：图上曲线必须与正文数字逐位一致，否则拒绝出图。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vldb_style as vs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "results_repro", "correlation_ci.npz")


def main():
    vs.apply()
    z = np.load(NPZ, allow_pickle=True)
    lv = np.asarray(z["means"], float)
    fac = np.exp(lv)

    def curve(tag):
        vals = np.stack([np.asarray(z[f"{tag}__{v:.1f}"], float) for v in lv])
        m = vals.mean(axis=1) * 100
        h = 1.96 * vals.std(axis=1, ddof=1) / np.sqrt(vals.shape[1]) * 100
        return m, h

    all_m, all_h = curve("cov_all")
    mul_m, mul_h = curve("cov_multi")
    adp_m, adp_h = curve("cov_multi_adapt")
    sin_m, sin_h = curve("cov_single")

    a = vs.Assertor()
    a.approx("R", int(z["R"]), 50)
    a.approx("无漂移 全局覆盖", all_m[0], 90.1, 0.05)
    a.approx("无漂移 多列敏感子集覆盖", mul_m[0], 86.6, 0.05)
    a.approx("7.39x 多列冻结覆盖", mul_m[-1], 46.7, 0.05)
    a.approx("7.39x 多列适应后覆盖", adp_m[-1], 87.3, 0.05)
    a.approx("7.39x 单列覆盖", sin_m[-1], 97.0, 0.05)
    a.approx("7.39x 全局冻结覆盖", all_m[-1], 63.5, 0.05)
    a.assert_all("fig1_correlation_drift")

    fig, ax = vs.subplots(1, 1, width=vs.COL, height=1.95)
    ax.fill_between(fac, adp_m - adp_h, adp_m + adp_h, color=vs.C_MAIN, alpha=0.15, lw=0)
    ax.fill_between(fac, mul_m - mul_h, mul_m + mul_h, color=vs.C_ALT, alpha=0.15, lw=0)
    ax.plot(fac, all_m, color=vs.C_FIFTH, marker="^", label="全局")
    ax.plot(fac, sin_m, color=vs.C_THIRD, marker="D", label="单列子集")
    ax.plot(fac, adp_m, color=vs.C_MAIN, marker="s", ls="--", label="多列敏感·适应后")
    ax.plot(fac, mul_m, color=vs.C_ALT, marker="o", label="多列敏感·冻结")
    vs.refline(ax, 90, "目标 90%")

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 7.39])
    ax.set_xticklabels(["1$\\times$", "2$\\times$", "4$\\times$", "7.39$\\times$"])
    ax.xaxis.set_minor_formatter(__import__("matplotlib").ticker.NullFormatter())
    ax.set_xlim(0.95, 8.2)
    ax.set_ylim(38, 101)
    ax.set_xlabel("相关性漂移因子")
    ax.set_ylabel("覆盖率 (%)")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.52),
              columnspacing=0.9, handletextpad=0.4)
    vs.save(fig, "fig1_correlation_drift")


if __name__ == "__main__":
    main()
