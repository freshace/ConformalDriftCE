# -*- coding: utf-8 -*-
"""附录 B 图 fig_ablation_bcd：特征维度 / 自适应标签量 / 树配置三组补充消融。

合并原来的 fig17_ablationB、fig18_ablationC、fig19_ablationD（三张各自独立的小图）：
三组实验讲的是同一件事——适应后覆盖率一律饱和在 90%，配置差异只体现在
冻结覆盖（失效可见性）与区间宽度 e^q（区间效率）上，故合成一张三联图。

数据源（只读）：results_repro/ablation_ci.npz
    keys 共 12 个臂：A| 为修正器开关（另见 plot_fig_width_visibility）、
    B| 特征维度、C| 标签量、D| 树配置。frozen/adapt 是覆盖率（0-1），width 是 e^q（倍）。
    R=20 次独立划分；误差棒取 95% CI，与附录 B 正文的 ± 口径一致。

画法：每臂一条"冻结 → 适应后"的哑铃，柱高差 = 被看见的失效幅度；
柱旁数字是该臂冻结时的区间宽度，越窄则同一残差变化越难被吸收。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vldb_style as vs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "results_repro", "ablation_ci.npz")


def main():
    vs.apply()
    z = np.load(NPZ, allow_pickle=True)
    keys = [str(k) for k in z["keys"]]

    def stat(name, key):
        v = np.asarray(z[name], float)[keys.index(key)]
        return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(v.size)

    def load(key):
        """-> (冻结覆盖, 冻结 CI, 适应后覆盖, 适应后 CI, 冻结区间宽度)。"""
        fm, fh = stat("frozen", key)
        am, ah = stat("adapt", key)
        wm, _ = stat("width", key)
        return fm * 100, fh * 100, am * 100, ah * 100, wm

    b10, bfull = load("B|s9-only(10d)"), load("B|full(622d)")
    c_lab = [load(f"C|{n}") for n in (100, 200, 500, 1000)]
    d_cfg = [load(k) for k in ("D|100t", "D|300t", "D|300t127l", "D|1000t")]

    a = vs.Assertor()
    a.approx("R(消融 B/C/D)", int(z["R"]), 20)
    # 覆盖率对 experiment_ablation_ci.txt 的 1 位小数口径（也是附录 B 正文的数字）。
    a.approx("B 10 维 冻结覆盖", b10[0], 74.2, 0.05)
    a.approx("B 622 维 冻结覆盖", bfull[0], 63.2, 0.05)
    a.approx("B 10 维 冻结覆盖 CI", b10[1], 0.9, 0.05)
    a.approx("B 622 维 冻结覆盖 CI", bfull[1], 0.8, 0.05)
    for i, (n, want_c) in enumerate(zip((100, 200, 500, 1000), (90.2, 90.1, 90.5, 90.3))):
        a.approx(f"C {n} 条 适应后覆盖", c_lab[i][2], want_c, 0.05)
    for i, (n, want_c) in enumerate(zip(("100t", "300t", "300t127l", "1000t"),
                                        (90.3, 90.3, 90.0, 90.3))):
        a.approx(f"D {n} 适应后覆盖", d_cfg[i][2], want_c, 0.05)
    a.approx("D 100t 冻结覆盖", d_cfg[0][0], 69.1, 0.05)
    a.approx("D 1000t 冻结覆盖", d_cfg[-1][0], 60.0, 0.05)
    a.approx("D 100t 冻结覆盖 CI", d_cfg[0][1], 0.8, 0.05)
    a.approx("D 1000t 冻结覆盖 CI", d_cfg[-1][1], 0.8, 0.05)
    # 区间宽度对 experiment_ablation_ci.txt 的 2 位小数口径。附录 B 正文把它四舍五入到
    # 1 位（41.15→41.2、43.45→43.5），故断言打在产物原值上才不会被舍入规则带偏。
    for tag, want_w, got in (("B 10 维", 41.15, b10[4]), ("B 622 维", 28.46, bfull[4]),
                             ("C 100", 62.04, c_lab[0][4]), ("C 200", 43.45, c_lab[1][4]),
                             ("C 500", 29.57, c_lab[2][4]), ("C 1000", 23.06, c_lab[3][4]),
                             ("D 100t", 28.96, d_cfg[0][4]), ("D 300t", 30.30, d_cfg[1][4]),
                             ("D 300t127l", 28.43, d_cfg[2][4]), ("D 1000t", 29.68, d_cfg[3][4])):
        a.approx(f"{tag} 区间宽度", got, want_w, 0.006)
    a.assert_all("fig_ablation_bcd")

    fig, axes = vs.subplots(1, 3, width=vs.DCOL, height=1.7)
    panels = [
        (axes[0], "(a) 特征维度", ["10 维", "622 维"], [b10, bfull], None),
        (axes[1], "(b) 自适应标签量", ["100", "200", "500", "1000"],
         c_lab, "自适应标签量（条）"),
        (axes[2], "(c) 树配置", ["100", "300", "300·127", "1000"],
         d_cfg, "树数量"),
    ]

    for ax, title, ticks, arms, xlabel in panels:
        x = np.arange(len(arms))
        f_m = [t[0] for t in arms]
        f_h = [t[1] for t in arms]
        a_m = [t[2] for t in arms]
        a_h = [t[3] for t in arms]
        # 竖线先画：把"冻结 -> 适应后"这段差距做成看得见的长度
        for xi, fm, am in zip(x, f_m, a_m):
            ax.plot([xi, xi], [fm, am], color=vs.C_LIGHT, lw=0.9, zorder=1)
        ax.errorbar(x, f_m, yerr=f_h, fmt="o", ms=3.2, color=vs.C_ALT,
                    elinewidth=0.7, capsize=1.6, zorder=3, label="冻结")
        ax.errorbar(x, a_m, yerr=a_h, fmt="s", ms=3.2, color=vs.C_MAIN,
                    elinewidth=0.7, capsize=1.6, zorder=3, label="适应后")
        vs.refline(ax, 90)
        # 宽度标在冻结点右侧，不与竖直连线重叠
        for xi, arm in zip(x, arms):
            ax.annotate(f"{arm[4]:.0f}×", (xi, arm[0]), textcoords="offset points",
                        xytext=(5, 0), va="center", ha="left",
                        fontsize=6.2, color=vs.C_REF)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks)
        ax.set_xlim(-0.5, len(arms) - 0.5)
        ax.set_ylim(50, 100)
        ax.set_yticks([50, 60, 70, 80, 90, 100])
        ax.set_title(title, fontsize=7.5, pad=3)
        if xlabel:
            ax.set_xlabel(xlabel)
    axes[0].set_ylabel("覆盖率 (%)")

    fig.subplots_adjust(wspace=0.34)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.17))
    vs.save(fig, "fig_ablation_bcd")


if __name__ == "__main__":
    main()
