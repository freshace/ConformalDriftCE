# -*- coding: utf-8 -*-
"""正文 §4.3.2 图 fig_width_visibility：区间宽度中介失效可见性（C2）。

合并原来的 fig10_ablation_corrector（消融 A：修正器有无）与 fig13_alphatradeoff
（α 扫描），因为二者讲的是同一件事：同一个漂移能被看见多少，由区间宽度决定。

数据源（只读）：
    results_repro/ablation_ci.npz          消融 A：A|base-only（无修正器）/ A|+LGBM（有修正器）
    results_repro/experiment_alphatradeoff.txt   α 扫描表（5 行文本表）

面板：
    (a) 修正器开关：冻结/适应后覆盖（柱），另标注适应后半宽。
        去掉修正器后冻结覆盖反而更高（83.0% vs 63.2%），；1203×是适应后半宽，不能解释冻结结果。
        适应后两臂都回到 90%。
    (b) α 扫描：横轴=区间宽度 e^q（log），纵轴=漂移后冻结覆盖。目标覆盖越高 → 区间越宽
        → 漂移后覆盖越高（失效越被掩盖）。直接给出"宽度 → 可见性"的剂量-响应。
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vldb_style as vs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "results_repro", "ablation_ci.npz")
TXT = os.path.join(ROOT, "results_repro", "experiment_alphatradeoff.txt")


def parse_alpha_table(path):
    """解析 α 扫描文本表 -> [(alpha, cov0, w0, cov1, w1)]，解析失败直接退出。"""
    if not os.path.exists(path):
        sys.exit(f"缺产物 {path}，拒绝出图")
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.match(
                r"\s*(0\.\d+)\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)x\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)x",
                line)
            if m:
                rows.append(tuple(float(g) for g in m.groups()))
    if len(rows) != 5:
        sys.exit(f"α 表解析出 {len(rows)} 行（期望 5 行），格式变了，拒绝出图")
    return rows


def main():
    vs.apply()
    z = np.load(NPZ, allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    ia, ib = keys.index("A|base-only"), keys.index("A|+LGBM")

    def ms(arr, i, scale):
        """第 i 行的 mean 与 95%CI 半宽。覆盖率传 scale=100，宽度本身就是倍率故传 1。"""
        v = np.asarray(arr, float)[i] * scale
        return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(v.size)

    def cov(arr, i):
        return ms(arr, i, 100.0)

    def width(arr, i):
        return ms(arr, i, 1.0)

    fr_a, fr_a_h = cov(z["frozen"], ia)
    fr_b, fr_b_h = cov(z["frozen"], ib)
    ad_a, ad_a_h = cov(z["adapt"], ia)
    ad_b, ad_b_h = cov(z["adapt"], ib)
    wd_a, _ = width(z["width"], ia)
    wd_b, _ = width(z["width"], ib)

    alpha_rows = parse_alpha_table(TXT)
    al = np.array([r[0] for r in alpha_rows])
    cov_drift = np.array([r[3] for r in alpha_rows])
    wid = np.array([r[4] for r in alpha_rows])

    a = vs.Assertor()
    a.approx("R(消融A)", int(z["R"]), 20)
    a.approx("无修正器 冻结覆盖", fr_a, 83.0, 0.05)
    a.approx("有修正器 冻结覆盖", fr_b, 63.2, 0.05)
    a.approx("无修正器 区间宽度", wd_a, 1203.0, 0.6)
    a.approx("有修正器 区间宽度", wd_b, 30.6, 0.05)
    a.approx("有修正器 适应后覆盖", ad_b, 90.3, 0.05)
    a.approx("α=0.01 漂移后覆盖", cov_drift[0], 98.5, 0.05)
    a.approx("α=0.01 区间宽度", wid[0], 188.3, 0.05)
    a.approx("α=0.30 漂移后覆盖", cov_drift[-1], 37.8, 0.05)
    a.approx("α=0.30 区间宽度", wid[-1], 2.7, 0.05)
    a.approx("α=0.10 区间宽度", wid[2], 7.7, 0.05)
    # 柱上直标的宽度字符串必须与正文引用的数字同源：曾经用 :,.0f 打成了"31×"，
    # 而正文写 30.6×，读者一眼能看出不一致。这里把显示串本身也钉住。
    a.approx("(a) 有修正器宽度显示串", "%.1f" % wd_b, "30.6")
    a.approx("(a) 无修正器宽度显示串", "%.1f" % wd_a, "1203.1")
    a.assert_all("fig_width_visibility")

    # α 表按 α 升序给行，宽度随之递减（α=0.01 最宽 188.3×，α=0.30 最窄 2.7×）。
    # 绘图前按宽度升序重排，下标顺序才与横轴一致，否则标注的左右让位会整体弄反。
    order = np.argsort(wid)
    pw, pc, pa = wid[order], cov_drift[order], al[order]

    fig, axes = vs.subplots(1, 2, width=vs.COL, height=1.75)

    # ---------------- (a) 修正器开关 ----------------
    # x = 状态（冻结 / 适应后），色 = 臂。图例与颜色一一对应，不用透明度表达第二个变量。
    ax = axes[0]
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2, [fr_a, ad_a], w, yerr=[fr_a_h, ad_a_h], capsize=1.6,
           color=vs.C_ALT, error_kw=dict(lw=0.7, ecolor=vs.C_INK), label="无修正器")
    ax.bar(x + w / 2, [fr_b, ad_b], w, yerr=[fr_b_h, ad_b_h], capsize=1.6,
           color=vs.C_MAIN, error_kw=dict(lw=0.7, ecolor=vs.C_INK), label="有修正器")
    vs.refline(ax, 90)
    # These saved widths are exp(q_new), after adaptation, not frozen q_old.
    ax.text(0.5, -0.24,
            "适应后半宽（无 / 有）\n%.1f× / %.1f×" % (wd_a, wd_b),
            transform=ax.transAxes, ha="center", va="top", fontsize=6.2,
            color=vs.C_REF, bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xticks(x)
    ax.set_xticklabels(["冻结", "适应后"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 104)
    ax.set_ylabel("覆盖率 (%)")
    ax.set_title("(a) 修正器开关", fontsize=7.5, pad=3)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.59))

    # ---------------- (b) α 扫描 ----------------
    ax = axes[1]
    ax.plot(pw, pc, color=vs.C_MAIN, marker="o", ls="-")
    # 曲线自左下向右上走，标注一律放在点的正上方：首点贴 y 轴、末点贴画布上沿，
    # 各自向轴内让位。白底让标注不压到折线上。
    for i, (wv, cv, av) in enumerate(zip(pw, pc, pa)):
        dx, ha = (2, "left") if i == 0 else \
                 ((-2, "right") if i == len(pw) - 1 else (0, "center"))
        ax.annotate(f"{100*(1-av):.0f}%", (wv, cv), textcoords="offset points",
                    xytext=(dx, 4), ha=ha, fontsize=6.2, color=vs.C_REF,
                    bbox=dict(fc="white", ec="none", pad=0.4))
    vs.refline(ax, 90)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xlim(2.0, 500)
    ax.set_ylim(30, 106)
    ax.set_xticks([3, 10, 30, 100])
    ax.set_xticklabels(["3", "10", "30", "100"])
    # 轴标签不写 $e^{q}$：与中文同串会丢字形（见 vldb_style._guard_text），
    # 符号写在图注里。
    ax.set_xlabel("乘性半宽（倍）")
    ax.set_ylabel("漂移后冻结覆盖 (%)")
    ax.set_title("(b) 目标覆盖 α 扫描", fontsize=7.5, pad=3)
    # 曲线自左下向右上走，画布右下角是空的；放左上会压到最宽那点的标注。
    ax.text(0.97, 0.05, "标注 = 目标覆盖 1−α", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.2, color=vs.C_REF)

    fig.subplots_adjust(wspace=0.55)
    vs.save(fig, "fig_width_visibility")


if __name__ == "__main__":
    main()
