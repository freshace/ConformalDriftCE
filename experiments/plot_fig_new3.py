# -*- coding: utf-8 -*-
"""附录的两张补充图（fig23 已于 2026-09-20 下线，见文末说明）。

  fig21_group_coverage : 分组(Mondrian) vs 全局 q 的逐组覆盖（无漂移；按表数 / 按过滤列数）
  fig22_selective      : 选择性预测 risk-coverage（R1 有保证 vs R2 启发式；设置 A/C）

fig23_hysteresis 不再出图，但从 results_hysteresis.npy 读数的**断言保留**：§4.4.2 正文
仍引用这批数字（85.5% / 93.0% / 99% / 320），删断言等于撤掉它们的守卫。

★ 两套数据源，务必分清（本脚本用断言把这条界线钉死）：
  - 逐组曲线只有【单次划分】的产物 results_groupcov.npy（seed=42，即 R=50 复现门的第 1 次
    重复）。R=50 的 50 条逐组曲线【从未落盘】——experiment_groupcoverage_ci.py 只保存了
    每次重复的组间极差（groupcov_ci.npz 的 spread_glob），逐组均值不在任何产物里，
    重算违反"不重算既有产物"，故 (a)(b) 的折线画 seed-42 单次划分，并在图上标明；
    R=50 的汇总量（极差 22.5±1.0 → 5.3±0.5pp、25.9±1.5 → 17.7±2.8pp）以文字框随图给出，
    这些数字与 appendix 正文引用的完全同源。
  - fig22 的产物本身即含全部所画数据（含 5 种子控制）；fig23 已下线，其产物只用于断言。

断言全部打在产物原文数字上（experiment_groupcoverage_ci.txt 的 R=50 表、复现门行；
results_selective.npy；results_hysteresis.npy），断言不过即退出。
单栏尺寸：固有宽 = 显示宽（COL=3.35in），正文/附录用 width=\\linewidth，不缩放。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vldb_style as vs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "results_repro", "groupcov_ci.npz")
GCNPY = os.path.join(ROOT, "results_groupcov.npy")
SELNPY = os.path.join(ROOT, "results_selective.npy")
HYSNPY = os.path.join(ROOT, "results_hysteresis.npy")

C_GLOB, C_GRP = vs.C_ALT, vs.C_MAIN       # 全局 q（现行对照）/ 分组 q（Mondrian）
C_R1, C_R2 = vs.C_MAIN, vs.C_ALT          # 有保证 / 启发式


def load(p):
    if not os.path.exists(p):
        sys.exit("产物缺失: " + p)
    return np.load(p, allow_pickle=True)


def main():
    vs.apply()

    z = load(NPZ)
    gc = load(GCNPY).item()
    sel = load(SELNPY).item()
    hy = load(HYSNPY).item()

    a = vs.Assertor()

    # ---------------- fig21 的数值：R=50 汇总 + seed-42 复现门 ----------------
    ks = [str(k) for k in z["keys"]]
    sg = np.asarray(z["spread_glob"], float)
    a.approx("R=50 复现次数", sg.shape[0], 50)
    a.approx("分组键数", sg.shape[1], 6)
    # spread_glob 以 pp 存放（22.49 即 22.49pp），不是比例；乘 100 会差两个数量级
    i_ntab, i_nfc = ks.index("A 无漂移|ntab"), ks.index("A 无漂移|nfc")
    m_g, h_g = vs.ci95(sg[:, i_ntab])
    a.approx("A 无漂移 按表数 极差(全局 q) 均值", "%.2f" % m_g, "22.49")
    a.approx("A 无漂移 按表数 极差(全局 q) CI", "%.2f" % h_g, "0.95")
    a.approx("A 无漂移 按表数 极差 读数下界", "%.1f" % sg[:, i_ntab].min(), "15.7")
    a.approx("A 无漂移 按表数 极差 读数上界", "%.1f" % sg[:, i_ntab].max(), "31.0")
    m_f, h_f = vs.ci95(sg[:, i_nfc])
    a.approx("A 无漂移 按过滤列数 极差(全局 q) 均值", "%.2f" % m_f, "25.93")
    a.approx("A 无漂移 按过滤列数 极差(全局 q) CI", "%.2f" % h_f, "1.52")

    rows_t = gc[("A 无漂移", "ntab")]["rows"]
    rows_f = gc[("A 无漂移", "nfc")]["rows"]
    cg_t = [r["cov_glob"] * 100 for r in rows_t]
    cp_t = [r["cov_grp"] * 100 for r in rows_t]
    cg_f = [r["cov_glob"] * 100 for r in rows_f]
    cp_f = [r["cov_grp"] * 100 for r in rows_f]
    # 复现门原文：seed42 组间极差 全局 29.6pp / 分组 6.0pp；最差组 全局 68.5% / 分组 86.7%
    a.approx("seed42 按表数 组数", len(rows_t), 6)
    a.approx("seed42 按表数 极差(全局)", "%.1f" % (max(cg_t) - min(cg_t)), "29.6")
    a.approx("seed42 按表数 极差(分组)", "%.1f" % (max(cp_t) - min(cp_t)), "6.0")
    a.approx("seed42 按表数 最差组(全局)", "%.1f" % min(cg_t), "68.5")
    a.approx("seed42 按表数 最差组(分组)", "%.1f" % min(cp_t), "86.7")
    a.approx("seed42 按过滤列数 组数", len(rows_f), 7)
    a.approx("seed42 按过滤列数 回退组数", sum(bool(r["fallback"]) for r in rows_f), 2)

    # ---------------- fig22 的数值：两条规则 + 5 种子控制 ----------------
    r1a, r2a = sel[("A 无漂移", "R1")], sel[("A 无漂移", "R2")]
    r1c = sel[("C 7.39x(适应)", "R1")]
    a.approx("A R1 保留率下界", "%.1f" % (min(r["ret"] for r in r1a) * 100), "14.2")
    a.approx("A R1 保留率上界", "%.1f" % (max(r["ret"] for r in r1a) * 100), "95.9")
    a.approx("A R1 接受集覆盖下界", "%.1f" % (min(r["cov"] for r in r1a) * 100), "90.7")
    a.approx("A R1 接受集覆盖上界", "%.1f" % (max(r["cov"] for r in r1a) * 100), "92.1")
    a.approx("A R2 接受集覆盖下界", "%.1f" % (min(r["cov"] for r in r2a) * 100), "90.9")
    a.approx("A R2 接受集覆盖上界", "%.1f" % (max(r["cov"] for r in r2a) * 100), "97.8")
    worst = min(r1c, key=lambda r: r["cov"])
    a.approx("C R1 最低接受集覆盖（正文 87.1%）", "%.1f" % (worst["cov"] * 100), "87.1")
    a.approx("C R1 最低覆盖处的保留率（正文 14.5%）", "%.1f" % (worst["ret"] * 100), "14.5")
    ct = sel[("control_R1", "5seeds")]
    cv = [r["cov"] * 100 for r in ct]
    a.approx("5 种子控制 覆盖下界", "%.1f" % min(cv), "89.8")
    a.approx("5 种子控制 覆盖上界", "%.1f" % max(cv), "91.7")
    a.approx("5 种子控制 覆盖均值", "%.1f" % np.mean(cv), "90.8")
    a.approx("5 种子控制 保留率均值", "%.1f" % (np.mean([r["ret"] for r in ct]) * 100), "62.6")

    # ---------------- fig23 的数值：long-tail 三策略 ----------------
    sm, tj = hy["summary"], hy["traj"]
    a.approx("long-tail 冻结覆盖（正文 85.5%）",
             "%.1f" % (sm["long-tail|single|mean_fro"][0] * 100), "85.5")
    a.approx("long-tail single 修复后覆盖",
             "%.1f" % (sm["long-tail|single|mean_our"][0] * 100), "88.6")
    a.approx("long-tail recal_only 修复后覆盖（正文 93.0%）",
             "%.1f" % (sm["long-tail|recal_only|mean_our"][0] * 100), "93.0")
    a.approx("long-tail recal_only 达标窗口率（正文 99%）",
             "%.0f" % (sm["long-tail|recal_only|pass_rate"][0] * 100), "99")
    a.approx("long-tail recal_only 标签量（正文 320）",
             "%.0f" % sm["long-tail|recal_only|labels"][0], "320")
    a.approx("long-tail sustained2 覆盖停在冻结水平",
             "%.1f" % (sm["long-tail|sustained2|mean_our"][0] * 100), "85.5")
    a.approx("long-tail sustained2 标签消耗为 0", sm["long-tail|sustained2|labels"][0], 0.0)
    # 画的轨迹必须与 summary 同源，否则图与正文各说一套
    for tag in ("single", "sustained2", "recal_only"):
        a.approx("%s 轨迹均值 = summary(冻结)" % tag,
                 np.mean(tj[tag]["cov_fro"]) * 100,
                 sm["long-tail|%s|mean_fro" % tag][0] * 100, 1e-6)
    a.assert_all("fig_new3")

    # ================= fig21：分组 vs 全局 q =================
    fig, axes = vs.subplots(1, 2, width=vs.COL, height=2.05, sharey=True)
    for ax, (rows, cg, cp, xlab, r50) in zip(
            axes,
            [(rows_t, cg_t, cp_t, "组：join 表数", (m_g, h_g, 5.26, 0.50)),
             (rows_f, cg_f, cp_f, "组：过滤列数", (m_f, h_f, 17.68, 2.82))]):
        x = np.array([r["g"] for r in rows], float)
        fb = [i for i, r in enumerate(rows) if r["fallback"]]
        vs.refline(ax, 90)
        ax.plot(x, cg, color=C_GLOB, ls="--", marker="o", ms=2.4,
                markerfacecolor="white", markeredgewidth=0.7, label="全局 q（现行）")
        # 回退组的分组 q 按定义等于全局 q。若把它接进折线，读者会读成"分组在这一组
        # 上塌了"；所以折线在回退点断开（置 NaN），该点单独用空心方块标出。
        cp_line = np.asarray(cp, float).copy()
        cp_line[fb] = np.nan
        ax.plot(x, cp_line, color=C_GRP, ls="-", marker="s", ms=2.4,
                markerfacecolor="white", markeredgewidth=0.7, label="分组 q（Mondrian）")
        if fb:
            ax.plot(x[fb], [cp[i] for i in fb], "s", ms=5.0, mfc="none",
                    mec=vs.C_REF, mew=0.7, zorder=6, label="样本不足，回退全局 q")
        ax.set_xticks(x)
        ax.set_xticklabels(["%d" % v for v in x])
        ax.set_xlabel(xlab)
        ax.set_ylim(62, 102)
        ax.set_yticks([70, 80, 90, 100])
        m, h, mg, hg = r50
        ax.text(0.97, 0.04, "R=50 组间极差\n%.1f±%.1f → %.1f±%.1f pp" % (m, h, mg, hg),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0,
                color=vs.C_REF, bbox=dict(facecolor="white", edgecolor="none", pad=1.8))
    axes[0].set_ylabel("组内覆盖率 (%)")
    h_, l_ = axes[1].get_legend_handles_labels()   # (b) 才有回退点，取它拿全三条目
    # 图例下移到横轴标题之下：两格各有一条 xlabel，图例挤在同一高度会糊在一起
    fig.legend(h_, l_, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    fig.text(0.5, -0.32, "折线 = 单次划分（seed 42）示例；框内为 R=50 汇总",
             ha="center", fontsize=5.8, color=vs.C_REF)
    fig.subplots_adjust(wspace=0.12)
    vs.save(fig, "fig21_group_coverage")

    # ================= fig22：选择性预测 =================
    fig, axes = vs.subplots(1, 2, width=vs.COL, height=2.05, sharey=True)
    for ax, name in zip(axes, ["A 无漂移", "C 7.39x(适应)"]):
        for tag, col, mk, ls, lab in (("R1", C_R1, "o", "-", "R1 分组 q"),
                                      ("R2", C_R2, "s", "--", "R2 bootstrap 方差")):
            rows = sel[(name, tag)]
            ax.plot([r["ret"] * 100 for r in rows], [r["cov"] * 100 for r in rows],
                    color=col, ls=ls, marker=mk, ms=2.3, lw=1.1,
                    markerfacecolor="white", markeredgewidth=0.7, label=lab)
        vs.refline(ax, 90)
        ax.set_xlabel("保留率 (%)")
        ax.set_title(name, fontsize=7.5, pad=3)
        ax.set_xlim(0, 100)
        ax.set_ylim(84, 100)
        ax.set_yticks([85, 90, 95, 100])
    # 5 个种子在保留率约 62% 处的接受集覆盖（正文另行引用）：点分散而非贴线，
    # 正好说明"硬保证"在换划分时并不抖，只有组粒度带来的离散。
    ax = axes[0]
    ax.plot([r["ret"] * 100 for r in ct], cv, "D", ms=2.4, mfc="none",
            mec=vs.C_INK, mew=0.6, zorder=6, label="R1 的 5 种子控制")
    # 说明文字压到左下角空白区（x<25 处蓝线之下是空的），贴点放会被 R2 橙线压住
    ax.text(0.02, 0.04, "5 种子（保留率 ≈62%%）\n接受集覆盖 %.1f–%.1f%%" % (min(cv), max(cv)),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8, color=vs.C_REF,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    axes[1].annotate("%.1f%%（激进保留）" % (worst["cov"] * 100),
                     (worst["ret"] * 100, worst["cov"] * 100), textcoords="offset points",
                     xytext=(5, -1), ha="left", va="top", fontsize=5.8, color=vs.C_REF)
    axes[0].set_ylabel("接受集覆盖率 (%)")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.20))
    fig.subplots_adjust(wspace=0.12)
    vs.save(fig, "fig22_selective")

    # ================= fig23 已下线（2026-09-20）=================
    # 原本画 long-tail 下 single / sustained2 / recal_only 的覆盖轨迹。撤掉的理由：
    #   (1) 它要讲的三件事（sustained2 从不触发、recal_only 320 标签到 93.0%、
    #       single 88.6%）已在 §4.4.2 正文逐条写清，数字同源；
    #   (2) §4.4.2 的结论本就是**依赖模式**的——long-tail/periodic 优先重校准，
    #       trend 上重训更好。单格 long-tail 会被读成"重校准胜出"，恰是结论的反面；
    #   (3) 端点数字已落在 tab:temporal 的行里。
    # 但上面 fig23 的那组断言**必须保留**：§4.4.2 仍引用这批数字，删断言=撤守卫。

    print("R=50 极差(全局 q)：按表数 %.2f±%.2f pp，按过滤列数 %.2f±%.2f pp"
          % (m_g, h_g, m_f, h_f))
    print("seed42 逐组：按表数 极差 %.1f→%.1f pp，最差组 %.1f→%.1f%%"
          % (max(cg_t) - min(cg_t), max(cp_t) - min(cp_t), min(cg_t), min(cp_t)))


if __name__ == "__main__":
    main()
