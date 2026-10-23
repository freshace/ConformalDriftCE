# -*- coding: utf-8 -*-
"""正文 §4.3.3 图 fig_detection：有标签可靠性信号 vs 无标签特征监控。

面板（数据源：results_repro/detection_ci.npz，R=50 次独立重复）：
  (a) 对**有害**的相关性漂移：有标签检测器（共形覆盖率 conf / 残差 resid）
      随漂移单调上升，无标签特征监控器（其最强版本 featZ）平躺在随机水平。
  (b) **检测反转**：横轴=真实危害（相对同族基线的覆盖损失 pp），纵轴=该点上的 featZ TPR。
      尺度漂移（无害、残差关系不变）点上 featZ 报警极强；相关性漂移（有害）点上贴随机线。

阈值口径（三者互不重叠）：阈值只在标定负窗口上定，FPR 在独立留出负窗口上测，
TPR 在正窗口上测。这决定了本图数字的可信度，正文一致。

单栏尺寸：固有宽 = 显示宽（COL），正文用 width=\\linewidth，不缩放。
所有绘制数值先与 results_repro/experiment_detection_ci.txt 的 R=50 口径逐位比对，
断言不过即退出。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vldb_style as vs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "results_repro", "detection_ci.npz")

C_CONF = vs.C_ALT    # 共形覆盖率（需标签）
C_RES = vs.C_MAIN    # 残差 / q-error（需标签）
C_FEAT = vs.C_THIRD  # 特征 z 统计量（无标签）


def main():
    vs.apply()
    if not os.path.exists(NPZ):
        sys.exit("产物缺失: " + NPZ)
    d = np.load(NPZ, allow_pickle=True)
    recs = d["recs"]
    dets = [str(x) for x in d["dets"]]
    di = {n: i for i, n in enumerate(dets)}
    # recs 列: 0 fam_i, 1 df, 2 det_i, 3 tpr, 4 fpr, 5 cov_整体, 6 cov_敏感子集

    def ci95(a):
        a = np.asarray(a, dtype=np.float64)
        return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))

    def get(fam, df, det):
        m = ((recs[:, 0] == fam) & (np.abs(recs[:, 1] - df) < 1e-9)
             & (recs[:, 2] == di[det]))
        t, ht = ci95(recs[m, 3] * 100)
        f, hf = ci95(recs[m, 4] * 100)
        cov = float(recs[m, 5].mean() * 100)
        return t, ht, f, hf, cov

    dfs_corr = sorted(set(recs[recs[:, 0] == 0, 1]))
    dfs_scal = sorted(set(recs[recs[:, 0] == 1, 1]))

    def r1(v):
        """experiment_detection_ci.txt 按四舍五入到 1 位小数打印（11.25 -> 11.3）。
        断言必须打在同一个口径上：直接用 ±0.05 容差会在 .5 处被浮点误差判错。"""
        return int(v * 10 + (0.5 if v >= 0 else -0.5)) / 10.0

    # ---------------- 数值断言：对齐 experiment_detection_ci.txt 的表 A / 表 B ----------------
    a = vs.Assertor()
    a.approx("相关性漂移档位 = 1.00..7.39（6 档）", len(dfs_corr), 6)
    a.approx("尺度漂移档位 = 1.00..50（5 档）", len(dfs_scal), 5)
    # 表 A：有标签检测器。TPR 随漂移上升，FPR 恒为标称水平。
    for tag, want in (("conf", (10.7, 100.0)), ("resid", (11.3, 100.0))):
        a.approx(f"corr {tag} TPR 基线(1.00x)", r1(get(0, dfs_corr[0], tag)[0]), want[0])
        a.approx(f"corr {tag} TPR 最大(7.39x)", r1(get(0, dfs_corr[-1], tag)[0]), want[1])
    a.approx("corr conf FPR（恒定）", r1(get(0, dfs_corr[-1], "conf")[2]), 9.2)
    a.approx("corr resid FPR（恒定）", r1(get(0, dfs_corr[-1], "resid")[2]), 9.8)
    # 表 B：无标签监控器。featZ 在相关性漂移上逐档不动，在尺度漂移上很强。
    for df in dfs_corr:
        a.approx(f"corr featZ TPR（{df:g}x）恒定", r1(get(0, df, "featZ")[0]), 11.9)
    a.approx("corr featZ FPR（恒定）", r1(get(0, dfs_corr[-1], "featZ")[2]), 11.2)
    for df, want in zip(dfs_scal, (11.9, 56.9, 100.0, 100.0, 100.0)):
        a.approx(f"scale featZ TPR（{df:g}x）", r1(get(1, df, "featZ")[0]), want)
    # 危害口径：整体覆盖。相关性漂移损失 26.9pp，尺度漂移只损失 6.0pp。
    a.approx("corr 基线整体覆盖", r1(get(0, dfs_corr[0], "featZ")[4]), 89.7)
    a.approx("corr 最大漂移整体覆盖", r1(get(0, dfs_corr[-1], "featZ")[4]), 62.8)
    a.approx("scale 最大漂移整体覆盖", r1(get(1, dfs_scal[-1], "featZ")[4]), 83.7)
    a.approx("检测反转 Pearson r", float(d["pearson"]), -0.138, 0.0006)
    a.approx("检测反转 Spearman rho", float(d["spearman"]), 0.273, 0.0006)
    a.assert_all("fig_detection")

    fig, axes = vs.subplots(1, 2, width=vs.COL, height=1.95)

    # ---------------- (a) 有害漂移下的三个检测器 ----------------
    ax = axes[0]
    x = np.array(dfs_corr)
    for det, col, lab, mk in (("conf", C_CONF, "覆盖率（需标签）", "o"),
                              ("resid", C_RES, "残差（需标签）", "^"),
                              ("featZ", C_FEAT, "特征 z（无标签）", "s")):
        y = np.array([get(0, d_, det)[0] for d_ in dfs_corr])
        e = np.array([get(0, d_, det)[1] for d_ in dfs_corr])
        ax.errorbar(x, y, yerr=e, color=col, marker=mk, elinewidth=0.7,
                    capsize=1.4, label=lab)
    # featZ 的随机水平就是它的误报率：一条虚线说明"平躺"不是画得平，是统计上无差别
    ch = get(0, dfs_corr[0], "featZ")[2]
    ax.axhline(ch, color=vs.C_REF, ls="--", lw=0.7)
    # 贴在右端、压住虚线之上：放左下会被 featZ 自己的点和绿线盖住
    ax.text(x[-1], ch + 6, "随机水平 %.0f%%" % ch, fontsize=6.0,
            color=vs.C_REF, ha="right", va="bottom")
    ax.set_xticks(x)
    # 1.35 与 2.00 在单栏宽度下只差 0.65 个横轴单位，横排必叠，转 40° 才放得下
    ax.set_xticklabels(["%g" % v for v in x], rotation=40, ha="right")
    ax.set_xlim(x[0] * 0.78, x[-1] * 1.10)
    ax.set_ylim(-6, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("相关性漂移因子（倍）")
    ax.set_ylabel("TPR (%)")
    ax.set_title("(a) 有害漂移：相关性漂移", fontsize=7.5, pad=3)
    # 两条曲线在 x≈2--4.5 之间从 20 爬到 100，左上与中右都被穿过；
    # 只有"绿线 12% 以上、平台 100% 以下"这条带子是空的，用 bbox 把图例钉在带内右端。
    ax.legend(fontsize=5.8, loc="upper right", bbox_to_anchor=(1.0, 0.82),
              borderaxespad=0.2)

    # ---------------- (b) 检测反转：危害 vs 无标签监控器 TPR ----------------
    ax = axes[1]
    series = {}
    for fam, col, lab, mk, dfs in ((0, C_CONF, "相关性漂移（有害）", "o", dfs_corr),
                                   (1, C_RES, "尺度漂移（无害）", "s", dfs_scal)):
        # 危害 = 相对同族基线的整体覆盖损失（pp），基线点因此落在 x=0
        base_cov = get(fam, dfs[0], "featZ")[4]
        xs = np.array([base_cov - get(fam, d_, "featZ")[4] for d_ in dfs])
        ys = np.array([get(fam, d_, "featZ")[0] for d_ in dfs])
        es = np.array([get(fam, d_, "featZ")[1] for d_ in dfs])
        series[fam] = (xs, ys)
        ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, elinewidth=0.7,
                    capsize=1.4, ls="none", label=lab)
    # 相关性漂移的 6 个点 TPR 全部相同（都贴在随机线上），逐点标注只会糊成一团；
    # 真正要读的档位是尺度漂移那 4 个分得开的点，相关性一侧只标末端。
    # 5×/10×/50× 三点同在 100% 一线上，必须一个朝上一个朝下，横排会连成一串。
    off = {2.0: (4, 0, "left"), 5.0: (0, 6, "center"),
           10.0: (0, -11, "center"), 50.0: (4, 5, "left")}
    for (xx, yy), dd in zip(zip(*series[1]), dfs_scal):
        if dd not in off:
            continue
        dx, dy, ha = off[dd]
        ax.annotate("%g×" % dd, (xx, yy), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=5.8, color=C_RES)
    xc, yc = series[0][0][-1], series[0][1][-1]
    ax.annotate("%g×" % dfs_corr[-1], (xc, yc), textcoords="offset points",
                xytext=(-2, -11), ha="right", fontsize=5.8, color=C_CONF)
    ch2 = get(0, dfs_corr[0], "featZ")[2]
    ax.axhline(ch2, color=vs.C_REF, ls="--", lw=0.7)
    # 相关性的 6 个点连成一段水平线，注释贴在它的右端上方
    ax.text(30.5, ch2 + 4, "随机水平 %.0f%%（=featZ 误报率）" % ch2, fontsize=6.0,
            color=vs.C_REF, ha="right", va="bottom")
    ax.set_xlim(-2.0, 31)
    ax.set_ylim(-6, 124)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("真实危害：覆盖损失 (pp)")
    ax.set_title("(b) 检测反转：featZ", fontsize=7.5, pad=3)
    # 尺度漂移点群止于 x≈6、相关性点群压在 12% 附近，中右那条空带留给图例
    ax.legend(fontsize=5.8, loc="center right", bbox_to_anchor=(1.0, 0.52),
              borderaxespad=0.2)

    fig.subplots_adjust(wspace=0.30)
    vs.save(fig, "fig14_detection")
    print("Pearson r = %+.3f  Spearman rho = %+.3f" % (float(d["pearson"]), float(d["spearman"])))


if __name__ == "__main__":
    main()
