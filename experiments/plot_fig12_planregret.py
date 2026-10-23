# -*- coding: utf-8 -*-
"""正文 §4.6 图 fig_plan_regret：真实 PostgreSQL 上的下游 regret（3 面板）。

【协议提醒】修正版 F2：\emph{选择}用注入基数信念后的估计代价，\emph{评估}用真数据实测耗时。
原图那版"稳健"一列来自一个恒等式（cost_robust = min(重新枚举同一批实验耗时)），
比值恒为 1、只剩计时噪声，所以"稳健始终贴近最优(1.00)"是假象，本脚本不再复现它。

数据源（只读，results_repro/）：
  planregret_native_matrices_MAT.npz     逐查询估计代价矩阵（--dump）
  planregret_native_D{1,7.39,16q}.npz    三档信念误差的逐查询实测 regret R = (n, 2)
  断言全部打在 experiment_planregret_native_*.txt 的原文数字上。

面板：
  (a) 机制：x=注入的基数信念（相对真值 t，log），y=该信念下估计代价 / 同信念下最小估计代价。
      三条线在所有信念下排序不变 ⇒ 共形区间（只表达基数不确定）没有决策信息量；
      右端方块=真数据实测耗时的同口径归一值，常与曲线排序矛盾 ⇒ 主导误差是方法排序。
  (b) 实测 regret 的 ECDF（D=16：区间收益在尾部转显著的那一档）。
  (c) 剂量-响应：P(regret>1.5) 随信念误差 D 变化（点估计 vs 稳健，bootstrap 95%），
      并标注"排序会变"子集里的更好/更坏条数与符号检验。

双栏尺寸：固有宽 = 显示宽（DCOL=7.03in），正文 width=\\textwidth，不缩放。
"""
import os
import sys

import numpy as np
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vldb_style as vs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results_repro")

C = {"hash": vs.C_ALT, "merge": vs.C_MAIN, "nestloop": vs.C_THIRD}
LBL = {"hash": "Hash Join", "merge": "Merge Join", "nestloop": "Nested Loop"}
C_PT, C_RB = vs.C_ALT, vs.C_MAIN


def binom_test(k, n, p=0.5):
    """双侧符号检验精确 p 值。"""
    if n == 0:
        return float("nan")
    pmf = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = pmf[k]
    return float(sum(x for x in pmf if x <= obs + 1e-12))


def pick_exemplar(dm):
    """挑一条【估计排序不变、但与实测排序矛盾】的查询——最能说明问题。"""
    best = None
    for k in range(len(dm["truth"])):
        est = dm["q%d_est" % k]
        real = dm["q%d_real" % k]
        arg_est, arg_real = int(np.argmin(est[-1])), int(np.argmin(real))
        if arg_est == arg_real:
            continue
        ratio = float(est[-1][arg_real] / est[-1][arg_est])
        if best is None or ratio > best[0]:
            best = (ratio, k)
    return (best[1] if best else 0), (best[0] if best else float("nan"))


def boot_ci(x, stat, B=2000, seed=7):
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.RandomState(seed)
    v = np.array([stat(x[rng.randint(0, n, n)]) for _ in range(B)])
    return float(stat(x)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired_tail_ci(regrets, threshold=1.5, B=2000, seed=7):
    """Point minus robust tail risk; bootstrap paired queries, not improvement rate."""
    regrets = np.asarray(regrets, float)
    if regrets.ndim != 2 or regrets.shape[1] < 2 or len(regrets) == 0:
        raise ValueError("regrets must contain nonempty paired point/robust columns")
    delta = (regrets[:, 0] > threshold).astype(float) - (regrets[:, 1] > threshold).astype(float)
    return boot_ci(delta, np.mean, B=B, seed=seed)


def main():
    vs.apply()
    # 三档信念误差：(标签, D 值, 有效条数, 排序不变占比, 排序会变的条数, 稳健改善条数,
    #               P(>1.5) 点估计, P(>1.5) 稳健)  —— 全部取自 results_repro/*.txt 原文
    DOSES = [("D1", 1.0, 260, 162, 98, 10, 12.3, 11.9),
             ("D7.39", 7.39, 257, 194, 63, 14, 8.6, 7.4),
             ("D16q", 16.0, 257, 199, 58, 17, 8.6, 7.0)]

    a = vs.Assertor()

    # ---------------- (a) 机制 ----------------
    dm = np.load(os.path.join(RES, "planregret_native_matrices_MAT.npz"), allow_pickle=True)
    shapes = [str(s) for s in dm["shapes"]]
    k, ratio = pick_exemplar(dm)
    scen = np.asarray(dm["q%d_scen" % k], float)
    est = np.asarray(dm["q%d_est" % k], float)
    real = np.asarray(dm["q%d_real" % k], float)
    a.approx("候选计划 = 3 种 join 方法", len(shapes), 3)
    a.approx("示例查询 估计/实测 排序矛盾倍数", round(ratio), 58)
    a.approx("示例查询 实测最优(ms)", round(float(real.min()), 1), 182.6, 0.05)
    a.approx("示例查询 实测 Nested Loop(ms)", round(float(real[shapes.index("nestloop")]), 1),
             228.3, 0.05)

    # ---------------- (b)(c) 三档剂量 ----------------
    for tag, D, n, n_unch, n_var, n_better, p_pt, p_rb in DOSES:
        d = np.load(os.path.join(RES, "planregret_native_%s.npz" % tag), allow_pickle=True)
        R = np.asarray(d["R"], float)
        ri = np.asarray(d["rank_inv"], bool)
        a.approx("%s 有效条数" % tag, len(R), n)
        # 排序不变 = rank_inv 为真；判定必须与 txt 的 "N/M 条排序不变" 一致
        a.approx("%s 排序不变条数" % tag, int(ri.sum()), n_unch)
        a.approx("%s 排序会变条数" % tag, int(len(R) - ri.sum()), n_var)
        a.approx("%s 稳健改善条数" % tag,
                 int(np.sum(R[~ri, 1] < R[~ri, 0] - 1e-9)), n_better)
        a.approx("%s P(regret>1.5) 点估计" % tag,
                 "%.1f" % (np.mean(R[:, 0] > 1.5) * 100), "%.1f" % p_pt)
        a.approx("%s P(regret>1.5) 稳健" % tag,
                 "%.1f" % (np.mean(R[:, 1] > 1.5) * 100), "%.1f" % p_rb)
    # 正文 §4.6 直接引用的 D=7.39 尾部数字
    d739 = np.load(os.path.join(RES, "planregret_native_D7.39.npz"), allow_pickle=True)
    R7 = np.asarray(d739["R"], float)
    ri7 = np.asarray(d739["rank_inv"], bool)
    a.approx("D=7.39 排序可变子集中位 regret 点估计", np.median(R7[~ri7, 0]), 1.311, 0.0005)
    a.approx("D=7.39 排序可变子集中位 regret 稳健", np.median(R7[~ri7, 1]), 1.222, 0.0005)
    a.approx("D=7.39 计划相同占比", "%.1f" % (np.mean(np.abs(R7[:, 1] - R7[:, 0]) < 1e-9) * 100), "93.0")
    nb, nw = int(np.sum(R7[~ri7, 1] < R7[~ri7, 0] - 1e-9)), int(np.sum(R7[~ri7, 1] > R7[~ri7, 0] + 1e-9))
    a.approx("D=7.39 符号检验 p", round(binom_test(nb, nb + nw), 3), 0.031, 0.0005)
    a.assert_all("fig_plan_regret")

    fig, axes = vs.subplots(1, 3, width=vs.DCOL, height=2.35)

    # ---------------- (a) 机制 ----------------
    ax = axes[0]
    for j, sh in enumerate(shapes):
        ax.plot(scen, est[:, j] / est.min(axis=1), "-o", ms=2.4, lw=1.1,
                color=C[sh], label=LBL[sh])
    xr = scen.max() * 2.3
    sq = np.array([real[j] / real.min() for j in range(len(shapes))])
    for j, sh in enumerate(shapes):
        # 三个实测方块落在 1.00/1.19/1.25 上（对数轴只隔约 4pt），必然互相压住；
        # 描一圈深色边才能看出是三块，而不是一块被叠色的方块。
        ax.plot([xr], [sq[j]], "s", ms=4.6, color=C[sh],
                markeredgecolor=vs.C_INK, markeredgewidth=0.4, zorder=5 + j)
    ax.axvline(1.0, color=vs.C_REF, ls=":", lw=0.7)
    ymax = float((est / est.min(axis=1, keepdims=True)).max())
    ax.set_ylim(0.85, ymax * 4.5)
    ax.set_xlim(scen.min() * 0.72, xr * 1.65)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1", "10", "100", "1000"])
    ax.minorticks_off()
    ax.set_xticks([1, 7.39, 56.8])
    ax.set_xticklabels(["t", "7.4t", "57t"])
    ax.set_xlabel("注入的基数信念（相对真值 t）")
    ax.set_ylabel("代价 / 同信念下最小代价")
    ax.set_title("(a) 代价模型的方法排序不随信念改变", fontsize=7.5, pad=3)
    # 三条线自左下平行上行、右端有实测方块，左上角整块是空的
    ax.legend(fontsize=6.2, loc="upper left", borderaxespad=0.3)
    # "实测"标在方块那一列的正上方（方块在数据坐标 x=xr 处）。方块最高只到 2.3×，
    # 原来贴轴底放会压在 1.0 附近的绿线上；抬到方块之上那一大段空白里才干净。
    ax.annotate("实测", (xr, float(sq.max()) * 2.0), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=6.2, color=vs.C_REF)

    # ---------------- (b) ECDF ----------------
    ax = axes[1]
    d16 = np.load(os.path.join(RES, "planregret_native_D16q.npz"), allow_pickle=True)
    R16 = np.asarray(d16["R"], float)
    for name, v, col, ls in (("点估计规则", R16[:, 0], C_PT, "-"),
                             ("稳健规则", R16[:, 1], C_RB, "--")):
        sv = np.sort(np.asarray(v, float))
        ax.step(sv, np.arange(1, len(sv) + 1) / len(sv), where="post",
                color=col, ls=ls, lw=1.2, label=name)
    m, lo, hi = paired_tail_ci(R16)
    # 1.5 那条参考虚线正从这段文字中间穿过，套一个白底把线遮掉：文字右对齐后
    # 左端仍落在 1.5 左边，靠调位置躲不开。文字压到轴底、白底加厚，虚线在轴上方
    # 才不留一小截戳出来的残段。
    ax.text(0.97, 0.02,
            "D=16，n=%d\nP(regret>1.5)：%.1f%% → %.1f%%\n差 %+.1fpp，95%% [%+.1f, %+.1f]"
            % (len(R16), np.mean(R16[:, 0] > 1.5) * 100, np.mean(R16[:, 1] > 1.5) * 100,
               m * 100, lo * 100, hi * 100),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=vs.C_REF,
            bbox=dict(facecolor="white", edgecolor="none", pad=2.0))
    ax.axvline(1.0, color=vs.C_REF, ls=":", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlim(0.99, 2.6)
    ax.set_xticks([1.0, 1.25, 1.5, 2.0, 2.5])
    ax.set_xticklabels(["1.0", "1.25", "1.5", "2.0", "2.5"])
    ax.minorticks_off()
    ax.set_xlabel("实测 plan regret（所选/后见最优）")
    ax.set_ylabel("累计比例")
    ax.set_title("(b) 实测 regret 的累积分布", fontsize=7.5, pad=3)
    # 两条 ECDF 几乎重合、平台贴在 y=1.0；坐标轴多留一截顶部，图例才不压平台线
    ax.legend(fontsize=6.2, loc="upper right", borderaxespad=0.3)
    ax.set_ylim(0, 1.18)

    # ---------------- (c) 剂量-响应 ----------------
    ax = axes[2]
    xs, pt_m, rb_m, pt_e, rb_e, subs = [], [], [], [], [], []
    for tag, D, n, n_unch, n_var, n_better, p_pt, p_rb in DOSES:
        d = np.load(os.path.join(RES, "planregret_native_%s.npz" % tag), allow_pickle=True)
        R = np.asarray(d["R"], float)
        m1, l1, h1 = boot_ci(R[:, 0], lambda x: float(np.mean(x > 1.5)))
        m2, l2, h2 = boot_ci(R[:, 1], lambda x: float(np.mean(x > 1.5)))
        xs.append(D)
        pt_m.append(m1 * 100)
        pt_e.append([m1 * 100 - l1 * 100, h1 * 100 - m1 * 100])
        rb_m.append(m2 * 100)
        rb_e.append([m2 * 100 - l2 * 100, h2 * 100 - m2 * 100])
        ri = np.asarray(d["rank_inv"], bool)
        nb = int(np.sum(R[~ri, 1] < R[~ri, 0] - 1e-9))
        nw = int(np.sum(R[~ri, 1] > R[~ri, 0] + 1e-9))
        subs.append((n_var, nb, nw, binom_test(nb, nb + nw)))
    xpos = np.arange(len(xs))
    w = 0.36
    ax.bar(xpos - w / 2, pt_m, w, yerr=np.array(pt_e).T, capsize=1.6,
           color=C_PT, label="点估计规则", error_kw=dict(lw=0.7, ecolor=vs.C_INK))
    ax.bar(xpos + w / 2, rb_m, w, yerr=np.array(rb_e).T, capsize=1.6,
           color=C_RB, label="稳健规则", error_kw=dict(lw=0.7, ecolor=vs.C_INK))
    ax.set_xticks(xpos)
    # 三档的 x 标签：上行是信念误差，下行是"排序会变"子集的符号检验摘要
    ax.set_xticklabels(["D=1\n%d 条可变\n%d 好 / %d 坏\np=%.3f" % subs[0],
                        "D=7.39\n%d 条可变\n%d 好 / %d 坏\np=%.3f" % subs[1],
                        "D=16\n%d 条可变\n%d 好 / %d 坏\np=%.3f" % subs[2]],
                       fontsize=6.2)
    ax.set_ylabel("P(regret > 1.5)  (%)")
    ax.set_title("(c) 剂量-响应：区间只在尾部起作用", fontsize=7.5, pad=3)
    ax.legend(fontsize=6.2, loc="upper right", borderaxespad=0.3)
    ax.set_ylim(0, max(pt_m + rb_m) * 1.45)

    fig.subplots_adjust(wspace=0.34)
    outdir = os.environ.get("DRIFT_FIG_OUT")
    if outdir and os.environ.get("DRIFT_FIG_QA_PATH"):
        sys.path.insert(0, os.environ["DRIFT_FIG_QA_PATH"])
        from audit_panel_alignment import require_matplotlib_panel_alignment
        os.makedirs(outdir, exist_ok=True)
        require_matplotlib_panel_alignment(fig, json_out=os.path.join(outdir, "fig12_plan_regret.alignment.json"))
    vs.save(fig, "fig12_plan_regret", outdir=outdir)
    print("Paired tail difference (point minus robust), pp:", np.array([m, lo, hi]) * 100)
    print("P(>1.5) 点估计:", np.round(pt_m, 2), " 稳健:", np.round(rb_m, 2))
    print("子集(排序可变) (n, 更好, 更坏, p):", subs)


if __name__ == "__main__":
    main()
