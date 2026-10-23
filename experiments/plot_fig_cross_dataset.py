# -*- coding: utf-8 -*-
"""正文 §4.5.2 图 fig_cross_dataset：换数据集后监控→修复是否仍然成立。

数据源（只读）：results_repro/crossdataset_ci.npz（R=20 次独立重复划分）
    列语义：<ds>__corr 为 (无漂移, 冻结, 适应后)；
            <ds>__work / <ds>__schema 为 (冻结, 适应后)。

画法：三块面板 = 三类漂移，横轴 = 数据集，实线 = 冻结、虚线 = 适应后。
按审稿意见把这张图放到双栏宽度：12 组点如果挤在单栏里，只能靠缩字体解决，
而点的高度差（失效幅度）恰恰是要读的东西。三块面板共用纵轴，跨库跨漂移的
跌幅才可比；"适应后一律回到 90%" 表现为所有虚线都贴在参考线上。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vldb_style as vs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "results_repro", "crossdataset_ci.npz")

DATASETS = ["stats", "imdb", "ergastf1", "genome"]


def main():
    vs.apply()
    z = np.load(NPZ, allow_pickle=True)

    def col(ds, drift, j):
        v = np.asarray(z[f"{ds}__{drift}"], float)[:, j] * 100
        return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(v.size)

    a = vs.Assertor()
    a.approx("R(跨数据集)", int(z["R"]), 20)
    # 相关性漂移：三列依次为 无漂移 / 冻结 / 适应后
    corr_want = {"stats": (90.1, 63.2, 90.3), "imdb": (90.2, 70.2, 90.1),
                 "ergastf1": (90.0, 56.9, 89.7), "genome": (90.0, 14.2, 90.1)}
    for ds, want in corr_want.items():
        for j, tag in enumerate(("无漂移", "冻结", "适应后")):
            a.approx(f"{ds} 相关性 {tag}", col(ds, "corr", j)[0], want[j], 0.05)
    # 工作负载 / schema 漂移：两列依次为 冻结 / 适应后
    for drift, want_tab in (("work", {"stats": (50.9, 89.3), "imdb": (32.1, 90.0),
                                      "ergastf1": (50.5, 90.3), "genome": (57.6, 90.0)}),
                            ("schema", {"stats": (66.8, 90.2), "imdb": (62.4, 90.1),
                                        "ergastf1": (72.3, 90.4), "genome": (82.0, 90.8)})):
        for ds, want in want_tab.items():
            for j, tag in enumerate(("冻结", "适应后")):
                a.approx(f"{ds} {drift} {tag}", col(ds, drift, j)[0], want[j], 0.05)
    a.assert_all("fig_cross_dataset")

    fig, axes = vs.subplots(1, 3, width=vs.DCOL, height=1.85)
    # 二三两列下标：corr 是 (无漂移, 冻结, 适应后)，work/schema 是 (冻结, 适应后)
    panels = [(axes[0], "(a) 相关性漂移", "corr", 1),
              (axes[1], "(b) 工作负载漂移", "work", 0),
              (axes[2], "(c) schema 漂移", "schema", 0)]
    x = np.arange(len(DATASETS))

    for ax, title, drift, off in panels:
        fr = [col(ds, drift, off) for ds in DATASETS]
        ad = [col(ds, drift, off + 1) for ds in DATASETS]
        for vals, color, ls, mk, lab in ((fr, vs.C_ALT, "-", "o", "冻结"),
                                         (ad, vs.C_MAIN, "--", "s", "适应后")):
            ax.errorbar(x, [v[0] for v in vals], yerr=[v[1] for v in vals],
                        color=color, ls=ls, marker=mk, elinewidth=0.7,
                        capsize=1.5, label=lab)
        vs.refline(ax, 90)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS)
        ax.set_xlim(-0.4, len(DATASETS) - 0.6)
        ax.set_ylim(5, 97)
        ax.set_yticks([10, 30, 50, 70, 90])
        ax.set_title(title, fontsize=7.5, pad=3)
    axes[0].set_ylabel("覆盖率 (%)")

    fig.subplots_adjust(wspace=0.28)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.13))
    vs.save(fig, "fig_cross_dataset")


if __name__ == "__main__":
    main()
