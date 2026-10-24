# -*- coding: utf-8 -*-
"""真实 30 窗回放图（Figure 9）中文重绘脚本。

数据源：data/supplement/sql_runs.csv、data/supplement/sql_windows.csv
输出  ：figures/final/real_cost_cn.{pdf,svg,png}

设计要点（都是踩过坑之后定下来的）：
  · (a) 不画任何"相位灰条"——正压在 100% 位置上会被读成"灰色曲线在 100% 持平"；
        改用竖向点线标分布切换（每 6 窗一次）。
  · (a) 不画带 caps 的误差棒——短横线会被读成箱线图的须；不确定性由图注与正文交代。
  · (b) 不把 compute_seconds（0.005–0.008 秒）放进堆叠——在 0–45 秒轴上占比 0.02%，
        物理不可见却仍占图例，属于误导；改为段内标注省略说明。
  · 六策略全部入图，与顶部图例一一对应（图 3 与图 11 各用两张图，按顺序数图号会错位，
    所以这里的策略数与图例必须自洽）。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "data" / "supplement"
OUT = ROOT / "figures" / "final"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "axes.unicode_minus": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})

POL = ["frozen", "frozen_trigger", "current_coverage", "current_residual", "periodic", "random"]
COL = ["#888888", "#D28C45", "#0072B2", "#009E73", "#9466A4", "#AF725B"]
MARK = ["o", "s", "^", "D", "v", "X"]
NAME = dict(zip(POL, ["不更新", "初始参照", "当前覆盖", "当前残差", "周期", "随机"]))
SHORT = dict(zip(POL, ["不更新", "初始", "覆盖", "残差", "周期", "随机"]))
FOCAL = ["frozen", "current_coverage"]
CW = "workload_cycles"
TAG_Y = 1.045


def tag(ax, letter, title):
    ax.text(0, TAG_Y, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")
    ax.text(.062, TAG_Y, title, transform=ax.transAxes, fontsize=8, va="bottom")


def boot_ci(v, seed=61012, n=10000):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    bs = np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))
    m = v[bs].mean(1)
    return v.mean(), np.quantile(m, .025), np.quantile(m, .975)


runs = pd.read_csv(TAB / "sql_runs.csv")
wins = pd.read_csv(TAB / "sql_windows.csv")

fig = plt.figure(figsize=(7.0, 2.6))
gs = fig.add_gridspec(1, 3, left=.112, right=.982, bottom=.235, top=.735,
                      wspace=1.02, width_ratios=[1.18, 1.12, 1.0])
axa = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[0, 1])
axc = fig.add_subplot(gs[0, 2])

# ---------------------------------------------------------------- (a) 每窗均值折线
W = int(wins[wins.condition == CW].window.max()) + 1
x = np.arange(W)
for j, pol in enumerate(POL):
    g = wins[(wins.condition == CW) & (wins.policy == pol)]
    m = g.groupby("window").coverage.mean().reindex(x).to_numpy() * 100
    focal = pol in FOCAL
    axa.plot(x, m, color=COL[j], lw=1.5 if focal else .9,
             alpha=1.0 if focal else .45, zorder=4 if focal else 2,
             marker="o" if focal else None, ms=2.6, mew=0, label=NAME[pol])

for w in (6, 12, 18, 24):                     # 每 6 窗一次分布切换
    axa.axvline(w - .5, color="#9aa3ad", lw=.6, ls=(0, (2, 2.5)), zorder=1)
axa.text(6, 101.5, "分布切换", ha="center", va="bottom", fontsize=6.2, color="#6b7480")
axa.axhline(90, color="#777777", lw=.65, ls=":", zorder=1)
axa.set_xlim(-.6, W - .4)
axa.set_ylim(55, 106)
axa.set_yticks([60, 70, 80, 90, 100])
axa.set_xlabel("窗口")
axa.set_ylabel("审计覆盖 (%)")
tag(axa, "a", "联合变化下的审计覆盖")

# ---------------------------------------------------------------- (b) 成本：监控 / 修复
grp = runs[runs.condition == CW].groupby("policy")
yy = np.arange(len(POL))
mon = np.asarray([grp.get_group(k).monitor_wall_seconds.mean() for k in POL])
rep = np.asarray([grp.get_group(k).repair_wall_seconds.mean() for k in POL])
comp = np.asarray([grp.get_group(k).compute_seconds.mean() for k in POL])
tot = mon + rep

axb.barh(yy, mon, height=.62, color="#B8CCD8", edgecolor="white", lw=.4, zorder=2)
axb.barh(yy, rep, left=mon, height=.62, color="#267BA3", edgecolor="white", lw=.4, zorder=2)
err_lo, err_hi = [], []
for i, k in enumerate(POL):
    _m, lo, hi = boot_ci(grp.get_group(k).deployment_wall_seconds.to_numpy())
    err_lo.append(max(tot[i] - lo, 0))
    err_hi.append(max(hi - tot[i], 0))
axb.errorbar(tot, yy, xerr=[err_lo, err_hi], fmt="none", ecolor="#444444",
             capsize=2, lw=.8, zorder=4)
j0 = POL.index("frozen_trigger")
axb.text(mon[j0] / 2 + .8, yy[j0], "监控", ha="left", va="center", fontsize=6.4, color="#33485c")
axb.text(mon[j0] + rep[j0] / 2, yy[j0], "修复", ha="center", va="center", fontsize=6.4, color="white")
axb.set_yticks(yy, [SHORT[k] for k in POL])
axb.invert_yaxis()
axb.tick_params(axis="y", length=0, pad=2)
axb.set_xlabel("实测在线耗时 (秒)")
axb.set_xlim(0, max(tot) * 1.28)
tag(axb, "b", "维护成本分解")

# ---------------------------------------------------------------- (c) 覆盖 vs 标签
MARKC = {"correlation_control": "o", "workload_cycles": "s"}
for cond in ["correlation_control", "workload_cycles"]:
    for j, pol in enumerate(POL):
        g = runs[(runs.condition == cond) & (runs.policy == pol)]
        if g.empty:
            continue
        v = g[["deployment_labels", "coverage"]].to_numpy()
        bs = np.random.default_rng(61012).integers(0, len(v), size=(10000, len(v)))
        means = v[bs].mean(1)
        lo, hi = np.quantile(means, .025, axis=0), np.quantile(means, .975, axis=0)
        m = v.mean(0)
        axc.errorbar(m[0], m[1] * 100,
                     xerr=[[m[0] - lo[0]], [hi[0] - m[0]]],
                     yerr=[[(m[1] - lo[1]) * 100], [(hi[1] - m[1]) * 100]],
                     fmt=MARKC[cond], ms=3.6, color=COL[j],
                     mfc=COL[j] if cond == "correlation_control" else "white",
                     mew=.8, capsize=1.4, lw=.7, zorder=3)
axc.axhline(90, color="#777777", lw=.65, ls=":")
axc.set_xlabel("在线标签数")
axc.set_ylabel("覆盖 (%)")
axc.set_ylim(78, 97)
axc.set_xlim(60, 1300)
axc.text(.98, .04, "实心：相关对照\n空心：联合变化", transform=axc.transAxes,
         ha="right", va="bottom", fontsize=6.4, color="#5b6470", linespacing=1.35)
tag(axc, "c", "覆盖与标签消耗")

handles = [Line2D([], [], color=COL[i], marker=MARK[i], lw=1.0, ms=3, label=NAME[POL[i]])
           for i in range(len(POL))]
fig.legend(handles=handles, ncol=6, loc="upper center", bbox_to_anchor=(.52, .995),
           columnspacing=1.1, handlelength=1.5, handletextpad=.5)

for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / ("real_cost_cn." + ext), dpi=600 if ext == "png" else 300, facecolor="white")
plt.close(fig)

print("已输出:", OUT / "real_cost_cn.png")
print("(b) 数值复核（应与正文表 5 一致）")
for i, k in enumerate(POL):
    print("  %-16s 监控=%.2f 修复=%.2f 合计=%.2f 计算=%.3f 标签=%.0f"
          % (k, mon[i], rep[i], tot[i], comp[i], grp.get_group(k).deployment_labels.mean()))
