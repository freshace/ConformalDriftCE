# -*- coding: utf-8 -*-
"""正文 §4.4.2 图 fig_temporal：20 个窗口上的覆盖轨迹（冻结 vs 触发式修复）。

数据源（只读）：
    results_temporal.npy
        traj[<mode>] = {cov_fro, cov_our, rho}  主配置 L=150 的跨种子均值轨迹，20 窗口
        summary['<mode>|150|mean_fro'|'mean_our'] = [mean, std]
    （上面的 mean 与 results_repro/experiment_temporaldrift.txt 表 A 逐位一致，
      本脚本把表 A 的 10 个覆盖率当作断言，防止图画对了但和正文数字不一致。）

画法：5 个漂移模式的小多图（small multiples）排成 2×3，第 6 格放图例与灰柱说明。
单栏尺寸：固有宽 = 显示宽（COL=3.35in），正文 width=\\linewidth，不缩放。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vldb_style as vs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPY = os.path.join(ROOT, "results_temporal.npy")

ORDER = [("none", "无漂移（对照）"), ("uniform", "uniform"), ("periodic", "periodic"),
         ("trend", "trend"), ("long-tail", "long-tail")]
# 表 A（L=150，5 种子）的 冻结覆盖 / 触发式修复后覆盖
WANT = {"none": ("89.9", "90.1"), "uniform": ("62.7", "89.9"),
        "periodic": ("79.9", "89.3"), "trend": ("81.2", "89.3"),
        "long-tail": ("85.5", "88.6")}


def main():
    vs.apply()
    if not os.path.exists(NPY):
        sys.exit("产物缺失: " + NPY)
    d = np.load(NPY, allow_pickle=True).item()
    traj, summ = d["traj"], d["summary"]

    a = vs.Assertor()
    for key, _ in ORDER:
        t = traj[key]
        a.approx("%s 窗口数 = 20" % key, len(t["cov_fro"]), 20)
        a.approx("%s 轨迹长度对齐" % key, len(t["cov_fro"]), len(t["cov_our"]))
        # 断言的字符串必须来自产物本身的打印口径：experiment_temporaldrift.txt
        # 用 f"{x*100:.1f}" 输出，85.55 这个值在浮点上略小于 85.55，故打印成 85.5。
        got_fro = "%.1f" % (summ["%s|150|mean_fro" % key][0] * 100)
        got_our = "%.1f" % (summ["%s|150|mean_our" % key][0] * 100)
        a.approx("%s 冻结覆盖(表A)" % key, got_fro, WANT[key][0])
        a.approx("%s 修复后覆盖(表A)" % key, got_our, WANT[key][1])
        # 绘图用的是轨迹；轨迹均值必须与 summary 同值，否则图与表会各说一套。
        # 这里比数值而不是比格式化字符串：long-tail 的 0.8555 在两种算法下末位差 1 ulp，
        # 会一个印成 85.5、一个印成 85.6，比字符串会误报。
        a.approx("%s 轨迹=summary(冻结)" % key,
                 np.asarray(t["cov_fro"]).mean() * 100,
                 summ["%s|150|mean_fro" % key][0] * 100, 1e-6)
    a.assert_all("fig_temporal")

    fig, axes = vs.subplots(2, 3, width=vs.COL, height=2.75)
    axes = np.asarray(axes).ravel()

    for i, (ax, (key, title)) in enumerate(zip(axes, ORDER)):
        t = traj[key]
        x = np.arange(len(t["cov_fro"]))
        # 灰柱 = 漂移强度 ρ(t)，画在底层、不占刻度（量纲与覆盖率无关）
        ax2 = ax.twinx()
        ax2.bar(x, t["rho"], color=vs.C_LIGHT, width=0.85, zorder=0)
        ax2.set_ylim(0, 2.6)
        ax2.set_yticks([])
        ax2.spines["right"].set_visible(False)
        vs.refline(ax, 90)
        ax.plot(x, np.asarray(t["cov_fro"]) * 100, color=vs.C_ALT, ls="--",
                marker="o", ms=1.8, markerfacecolor="white", markeredgewidth=0.6,
                label="冻结")
        ax.plot(x, np.asarray(t["cov_our"]) * 100, color=vs.C_MAIN, ls="-",
                marker="s", ms=1.8, markerfacecolor="white", markeredgewidth=0.6,
                label="触发式修复")
        ax.set_title(title, fontsize=7.2, pad=2)
        ax.set_ylim(40, 100)
        ax.set_xlim(-0.7, len(x) - 0.3)
        ax.set_xticks([0, 10, 19])
        ax.set_xticklabels(["0", "10", "19"])
        # 小多图共享纵轴：所有面板同一组刻度，只留左列刻度标签
        ax.set_yticks([50, 70, 90])
        ax.tick_params(labelleft=(i % 3 == 0))
        if i >= 3:
            ax.set_xlabel("时间窗口")

    axes[0].set_ylabel("覆盖率 (%)")

    ax_lg = axes[5]
    ax_lg.axis("off")
    h, lb = axes[0].get_legend_handles_labels()
    ax_lg.legend(h, lb, loc="center", fontsize=7.0, borderaxespad=0.0)
    ax_lg.text(0.5, 0.13, "灰柱 = 漂移强度 ρ(t)", ha="center", fontsize=6.2,
               color=vs.C_REF, transform=ax_lg.transAxes)

    fig.subplots_adjust(wspace=0.22, hspace=0.46)
    vs.save(fig, "fig20_temporal_drift")


if __name__ == "__main__":
    main()
