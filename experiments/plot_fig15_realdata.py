# -*- coding: utf-8 -*-
"""附录 B 图 fig:realdata-correlation：数据级相关性漂移下的覆盖率。

这一格回答的是"漂移是不是只是标签级乘了个 exp(mu)"：这里直接改写数据本身。
构造：200000x8 Gaussian copula，列按"组内相关、组间独立"分块并做经验秩变换，
边际都是 [0,1] 上的 uniform；相关性漂移把\emph{组内} rho 从 0.30 提到 0.90，
各列\emph{边际不变}。底模取 N*prod(边际选择性)（PG 直方图式的边际估计），
边际不变则底模不动，真基数却随相关性改变而重算——漂移是真的数据级。

数据源（只读）：
  results_repro/correlation_realdata_ci.npz   R=50 次独立重复，每次同时重抽
      数据实现与查询/划分。reps=rho 网格；cov_f/cov_r=整体冻结/适应后，
      cov_f_sens/cov_r_sens=相关性敏感子集；base_cov=同分布基础覆盖。
  （断言全部打在 results_repro/experiment_correlation_realdata_ci.txt 的原文数字上。
    该 txt 用 "%.1f" 打印覆盖率、用 "%.2f" 打印 SE，故断言也用同一格式化口径。）

面板：(a) 整体（全部查询）；(b) 相关性敏感子集（组内多列过滤）。
两格共享纵轴，正是为了让"整体只掉 30pp、敏感子集塌掉 70pp"的对比一眼可见。
单栏尺寸：固有宽 = 显示宽（COL=3.35in），正文 width=\linewidth，不缩放。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vldb_style as vs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "results_repro", "correlation_realdata_ci.npz")

C_FROZEN, C_REPAIR = vs.C_ALT, vs.C_MAIN


def ms(a):
    """跨重复的均值与 95% CI 半宽（图里的阴影）。"""
    m = a.mean(axis=0) * 100
    h = 1.96 * a.std(axis=0, ddof=1) / np.sqrt(a.shape[0]) * 100
    return m, h


def main():
    vs.apply()
    if not os.path.exists(NPZ):
        sys.exit("产物缺失: " + NPZ)
    z = np.load(NPZ, allow_pickle=True)
    rho = np.asarray(z["reps"], float)
    cov_f, cov_r = z["cov_f"], z["cov_r"]
    cov_f_s, cov_r_s = z["cov_f_sens"], z["cov_r_sens"]

    ov_f_m, ov_f_h = ms(cov_f)
    ov_r_m, ov_r_h = ms(cov_r)
    se_f_m, se_f_h = ms(cov_f_s)
    se_r_m, se_r_h = ms(cov_r_s)

    # ---------------- 数值断言：对齐 experiment_correlation_realdata_ci.txt ----------------
    a = vs.Assertor()
    a.approx("rho 档位数", len(rho), 5)
    a.approx("R = 50", cov_f.shape[0], 50)
    a.approx("rho 起点", rho[0], 0.30, 1e-9)
    a.approx("rho 终点", rho[-1], 0.90, 1e-9)
    # 表 1：四条曲线逐档对齐（产品按 "%.1f" 打印）
    for name, got, want in (("整体冻结", ov_f_m, (89.8, 80.7, 67.2, 62.0, 59.3)),
                            ("整体适应后", ov_r_m, (90.2, 90.4, 89.8, 90.0, 90.3)),
                            ("敏感冻结", se_f_m, (77.9, 56.4, 24.9, 12.5, 6.2)),
                            ("敏感适应后", se_r_m, (77.7, 77.8, 76.5, 77.0, 77.3))):
        for i, w in enumerate(want):
            a.approx("%s rho=%.2f" % (name, rho[i]), "%.1f" % got[i], "%.1f" % w)
    # 表 2：敏感子集的修复增益；适应后覆盖贴回无漂移水平（正文引用的两个端点）
    d = se_r_m - se_f_m
    for i, w in enumerate((-0.2, 21.4, 51.6, 64.5, 71.1)):
        a.approx("敏感 修复-冻结 rho=%.2f" % rho[i], "%+.1f" % d[i], "%+.1f" % w)
    a.approx("正文 敏感适应后端点(rho=0.90)", "%.1f" % se_r_m[-1], "77.3")
    a.approx("正文 整体适应后端点(rho=0.90)", "%.1f" % ov_r_m[-1], "90.3")
    # 闸门 G0：rho=0.30 整体冻结覆盖须落在 90% 附近；同分布基础覆盖另有一组读数
    se0 = cov_f[:, 0].std(ddof=1) / np.sqrt(cov_f.shape[0]) * 100
    a.approx("G0 rho=0.30 整体冻结覆盖", "%.1f" % ov_f_m[0], "89.8")
    a.approx("G0 rho=0.30 的 SE", "%.2f" % se0, "0.13")
    a.approx("G0 CI 下界", "%.1f" % (ov_f_m[0] - 1.96 * se0), "89.6")
    a.approx("G0 CI 上界", "%.1f" % (ov_f_m[0] + 1.96 * se0), "90.1")
    b = np.asarray(z["base_cov"], float) * 100
    a.approx("同分布基础覆盖", "%.1f" % b.mean(), "89.9")
    a.approx("同分布基础覆盖 SE", "%.2f" % (b.std(ddof=1) / np.sqrt(len(b))), "0.14")
    # 不确定度段：rho=0.90 的单次读数范围（图上用散点如实画出来）
    a.approx("rho=0.90 整体冻结 读数下界", "%.1f" % (cov_f[:, -1].min() * 100), "57.2")
    a.approx("rho=0.90 整体冻结 读数上界", "%.1f" % (cov_f[:, -1].max() * 100), "61.4")
    a.approx("rho=0.90 敏感冻结 读数下界", "%.1f" % (cov_f_s[:, -1].min() * 100), "4.7")
    a.approx("rho=0.90 敏感冻结 读数上界", "%.1f" % (cov_f_s[:, -1].max() * 100), "8.2")
    a.assert_all("fig_realdata_correlation")

    # ---------------- 出图 ----------------
    fig, axes = vs.subplots(1, 2, width=vs.COL, height=1.95, sharey=True)

    def panel(ax, fm, fh, rm, rh, fro_s, base, title):
        # 无漂移参考线 = 该口径在 rho=0.30 的冻结覆盖；两格各自取值，
        # 因为 split conformal 只给边际覆盖，敏感子集的无漂移水平本就远低于 90%。
        vs.refline(ax, base, lw=0.8)
        # 单次读数如实画成细点：只画均值曲线会让人以为 50 次重复几乎没有波动，
        # 而正文特意用这个范围说明"原脚本报的只是其中一次抽样"。
        ax.plot(np.full(len(fro_s), rho[-1]), fro_s * 100, ".", ms=1.6,
                color=C_FROZEN, alpha=0.45, zorder=2)
        ax.fill_between(rho, fm - fh, fm + fh, color=C_FROZEN, alpha=0.18, lw=0)
        ax.fill_between(rho, rm - rh, rm + rh, color=C_REPAIR, alpha=0.18, lw=0)
        ax.plot(rho, fm, color=C_FROZEN, ls="--", lw=1.2, marker="o", ms=2.2,
                markerfacecolor="white", markeredgewidth=0.7, label="冻结（不修复）")
        ax.plot(rho, rm, color=C_REPAIR, ls="-", lw=1.2, marker="s", ms=2.2,
                markerfacecolor="white", markeredgewidth=0.7, label="适应后（500+400 标签）")
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.set_xlabel("漂移后的组内相关 ρ")
        ax.set_xticks(rho)
        ax.set_xticklabels(["%.2f" % v for v in rho])
        ax.set_xlim(rho.min() - 0.03, rho.max() + 0.05)
        ax.text(rho[-1] + 0.045, base + 2.0, "无漂移\n%.1f%%" % base, ha="right",
                va="bottom", fontsize=5.8, color=vs.C_REF)

    axes[0].set_ylabel("覆盖率 (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_yticks([0, 20, 40, 60, 80, 100])
    panel(axes[0], ov_f_m, ov_f_h, ov_r_m, ov_r_h, cov_f[:, -1],
          ov_f_m[0], "(a) 整体（全部查询）")
    panel(axes[1], se_f_m, se_f_h, se_r_m, se_r_h, cov_f_s[:, -1],
          se_f_m[0], "(b) 相关性敏感子集")

    # 图例挂到横轴标题下面：两格共用一条 x 轴标题，图例挤在同一高度会糊成一行
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    fig.subplots_adjust(wspace=0.14)
    vs.save(fig, "fig15_realdata_correlation")
    print("整体 %.1f→%.1f，适应后 %.1f | 敏感 %.1f→%.1f，适应后 %.1f"
          % (ov_f_m[0], ov_f_m[-1], ov_r_m[-1], se_f_m[0], se_f_m[-1], se_r_m[-1]))


if __name__ == "__main__":
    main()
