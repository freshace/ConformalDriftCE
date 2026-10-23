# -*- coding: utf-8 -*-
"""画论文 Figure 1（method.tex 的 fig:method-arch）：方法总览。

★ 口径纪律（与正文/图注逐字对齐，禁止 Level-3 主张）：
  1. 闭环名用「监控 → 选择性取标签 → 修复 → 独立审计」（生命周期），
     不用 "detect → repair → verify loop" 这类把闭环等同于漂移检测的标题。
  2. 监控框只写「窗口经验覆盖率跌破目标 → 有标签可靠性失效信号」，并显式标注
     「不识别漂移原因、不是漂移检测器」；图中不出现"漂移报警"。
  3. 输出框不写「cov ≥ 90% ✓ 可靠」这类无条件下保证：cov 标注为「窗口级
     经验统计量，非逐查询保证」，覆盖保证以「校准样本与后续审计样本可交换」为前提。
  4. 审计框写明「同时报覆盖率与宽度」——只看覆盖率可被"放宽区间"作弊。
  （历史：旧 arch.png 是外部工具产物、无可编辑源，含"漂移报警"/"cov ≥ 90% ✓ 可靠"
    两处越级主张，且无法复现；本脚本取代它，输出矢量 PDF。）

用法：python experiments/plot_fig_arch.py
输出：paper/figures/arch.pdf + arch.png(300dpi)，供 method.tex 的 figure* 引用。
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vldb_style as vs

vs.apply()
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "paper", "figures")
os.makedirs(FIG, exist_ok=True)

# 配色沿用全论文的 Okabe–Ito 语义族，与其它图一致：底模=主线蓝、机器学习=对照朱红、
# 共形/可靠性=第三条臂绿、闭环/运维=第四条臂品红。浅底由实色兑白得到。
C_DB, C_ML, C_ST, C_OP = vs.C_MAIN, vs.C_ALT, vs.C_THIRD, vs.C_FOURTH


def tint(hexc, f=0.15):
    """实色按 f 比例兑白，得到同色系的浅底。"""
    r, g, b = (int(hexc[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(int(round(f * c + (1 - f) * 255)) for c in (r, g, b))


C_DB_F, C_ML_F, C_ST_F, C_OP_F = tint(C_DB), tint(C_ML), tint(C_ST), tint(C_OP)
C_INK, C_EDGE, C_WARN = vs.C_INK, vs.C_REF, "#B3301E"

# 固有宽 = 显示宽：正文 method.tex 用 width=\textwidth 引用，不缩放。
# 画布按 130x78 的数据坐标铺满 DCOL，字号同步放大同一比例，最小字号因此不低于 6pt。
S = 1.22
W_IN = 7.03
fig, ax = plt.subplots(figsize=(W_IN, W_IN * 78.0 / 130.0))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_xlim(0, 130)
ax.set_ylim(0, 78)
ax.set_aspect("equal")
ax.axis("off")


def box(x0, x1, y0, y1, fc, ec, lw=1.4, r=1.0, z=2):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=%.2f" % r,
                                fc=fc, ec=ec, lw=lw, zorder=z))


def txt(x, y, s, size=6.0, color=C_INK, ha="center", va="center", weight="normal", z=6):
    ax.text(x, y, s, fontsize=size * S, color=color, ha=ha, va=va,
            fontweight=weight, zorder=z, linespacing=1.5)


def arrow(x0, y0, x1, y1, color=C_EDGE, lw=1.4, ls="solid"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, shrinkA=0, shrinkB=0), zorder=4)


def dashed_path(pts, color=C_OP, lw=1.3):
    """折线（虚线）+ 末段箭头；用于生命周期回环。"""
    ls = (0, (4, 2.5))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=3, solid_capstyle="butt")
    arrow(xs[-2], ys[-2], xs[-1], ys[-1], color=color, lw=lw, ls=ls)


# ================= 上排：外挂主流程 (1)~(5) =================
# --- (1) 输入特征 ---
box(2, 22, 42, 78, C_DB_F, C_DB)
txt(12, 74.6, "(1) 输入特征", 7.2, weight="bold")
box(6, 18, 66, 71.6, "#FFFFFF", C_EDGE, 0.9)
txt(12, 68.8, "查询 Q", 6.4)
txt(12, 64.2, "+", 8)
box(3.2, 20.8, 44, 62, "#FFFFFF", C_EDGE, 0.9)
txt(12, 59.6, "数据库统计特征 x", 6.2)
txt(12, 57.4, "(622 维)", 5.8)
for i, s in enumerate(["• 表/列 one-hot", "• fanout 与表规模", "• 过滤选择性",
                       "• log b̂ (底模估计)"]):
    txt(4.4, 53.6 - 2.4 * i, s, 5.4, ha="left")
arrow(22.4, 60, 24.6, 60)

# --- (2) 底模点估计器 ---
box(25, 47, 42, 78, C_DB_F, C_DB)
txt(36, 74.6, "(2) 底模点估计器", 7.2, weight="bold")
txt(36, 70.6, "任意已部署的基数估计器", 5.8)
txt(36, 68.2, "(PostgreSQL / ALECE / PRICE)", 5.4)
box(27.2, 44.8, 55, 65, "#FFFFFF", C_EDGE, 0.9)
txt(36, 62.4, "数据库执行计划", 6.2)
txt(36, 60.1, "与统计信息", 6.2)
box(27.2, 44.8, 45, 52.6, "#FFFFFF", C_EDGE, 0.9)
txt(36, 50.4, "点估计 b̂", 6.4)
txt(36, 47.9, "(底模；不改动底模)", 5.4)
arrow(47.4, 60, 49.6, 60)

# --- (3) 残差修正器 ---
box(50, 79, 42, 78, C_ML_F, C_ML)
txt(64.5, 74.6, "(3) 残差修正器", 7.2, weight="bold")
txt(64.5, 72.0, "$f_\\theta$ (LightGBM)", 6.2)
box(52, 69.4, 49, 60, "#FFFFFF", C_EDGE, 0.9)
txt(60.7, 56.6, "特征输入 x + b̂", 6.2)
txt(60.7, 54.0, "(统计特征 / 底模估计)", 5.4)
txt(60.7, 51.2, "输出残差 r = fθ(x)", 5.8)
box(70.8, 77.6, 49, 60, "#FFFFFF", C_EDGE, 0.9)
txt(74.2, 57.4, "校正点", 5.6)
txt(74.2, 55.2, "估计", 5.6)
txt(74.2, 52.4, "$\\hat y$", 6.4)
txt(74.2, 50.2, "截断至 ≥1", 5.0)
arrow(79.4, 60, 81.6, 60)

# ================= 上排（续）(4)(5) =================
# --- (4) 共形校准 ---
box(82, 107, 42, 78, C_ST_F, C_ST)
txt(94.5, 74.6, "(4) 共形校准", 7.2, weight="bold")
txt(94.5, 72.0, "(Split Conformal)", 5.4)
box(83.8, 105.2, 62, 69.4, "#FFFFFF", C_EDGE, 0.9)
txt(94.5, 67.2, "校准分数(残差绝对值)", 5.8)
txt(94.5, 64.6, "$s_i=|\\log y_i-\\log \\hat y_i|$", 6.0)
box(83.8, 105.2, 52, 60, "#FFFFFF", C_EDGE, 0.9)
txt(94.5, 57.8, "校准分位数", 5.8)
txt(94.5, 55.2, "$q=\\mathrm{Quantile}_{1-\\alpha}(\\{s_i\\})$", 6.0)
box(83.8, 105.2, 44.6, 50.4, "#FFFFFF", C_EDGE, 0.9)
txt(94.5, 48.5, "预测区间(目标 90%)", 5.8)
txt(94.5, 46.2, "$[\\hat y/e^{q},\\ \\hat y\\,e^{q}]$", 5.8)
txt(94.5, 43.2, "※ 覆盖保证前提:样本可交换", 5.0, color=C_WARN)
arrow(107.4, 60, 109.6, 60)

# --- (5) 输出结果（带可靠性注解）---
box(110, 129, 42, 78, C_ST_F, C_ST)
txt(119.5, 74.6, "(5) 输出结果", 7.2, weight="bold")
txt(119.5, 72.0, "(带可靠性注解)", 5.4)
box(111.4, 127.6, 60, 69.4, "#FFFFFF", C_EDGE, 0.9)
txt(119.5, 67.0, "点估计 ŷ", 5.8)
txt(119.5, 64.4, "乘性半宽 e^q", 5.8)
txt(119.5, 61.8, "窗口覆盖率 cov", 5.8)
box(111.4, 127.6, 44.6, 58, "#FFF8E1", "#E0A800", 1.0)
txt(119.5, 55.6, "cov 为窗口级经验量,", 5.4)
txt(119.5, 53.2, "非逐查询保证", 5.4)
txt(119.5, 50.6, "与目标 90% 比较,", 5.4)
txt(119.5, 48.2, "只作可靠性注解", 5.4)

# ================= 下排：标签驱动的生命周期回环 A→B→C =================
box(3, 21, 13, 36, "#F7F7F7", "#9E9E9E", 0.9)
txt(12, 33.2, "口径", 6.6, weight="bold")
for i, s in enumerate(["1. 覆盖保证只在", "样本可交换时成立",
                       "2. cov 是窗口级经验量,", "非逐查询保证",
                       "3. 漂移下不宣称", "90% 保证;该信号", "不识别漂移原因"]):
    txt(12, 29.6 - 2.2 * i, s, 5.0)

box(24, 57, 13, 36, C_OP_F, C_OP)
txt(40.5, 33.2, "C. 独立审计", 6.6, weight="bold")
txt(40.5, 30.2, "在不参与训练/校准的审计段", 5.4)
txt(40.5, 27.9, "上重算经验覆盖率", 5.4)
txt(40.5, 24.8, "只看覆盖率可被“放宽区间”", 5.0)
txt(40.5, 22.6, "通过,故联合报宽度与点误差", 5.0)
txt(40.5, 19.4, "审计只能发现、不能修复:", 5.0)
txt(40.5, 17.2, "修 q 需要代表性校准数据", 5.0)

box(60, 92, 13, 36, C_OP_F, C_OP)
txt(76, 33.2, "B. 按需取标签 & 修复", 6.6, weight="bold")
txt(76, 30.2, "训练500 + 校准500条标签", 5.4)
txt(76, 27.9, "重定 q;必要时重训 fθ", 5.6)
txt(76, 24.8, "校准段与训练段不相交,", 5.0)
txt(76, 22.6, "避免把校准数据拿去训练", 5.0)
txt(76, 19.4, "两种修复分别评价:", 5.0)
txt(76, 17.2, "检查尾部与稳定查询损害", 5.0)

box(96, 129, 13, 36, C_OP_F, C_OP)
txt(112.5, 33.2, "A. 窗口覆盖率监控", 6.6, weight="bold")
txt(112.5, 30.2, "固定旧 q;按预设规则报警", 5.6)
txt(112.5, 27.9, "→ 记为可靠性失效信号", 5.6)
txt(112.5, 24.8, "(有标签;不识别漂移原因,", 5.0)
txt(112.5, 22.6, "也不是漂移检测器)", 5.0)
txt(112.5, 19.4, "监控与审计各需一笔", 5.0)
txt(112.5, 17.2, "有代表性的标签预算", 5.0)

# 生命周期回环箭头（虚线）：输出 → A → B → C → 回灌 (3)/(4)
dashed_path([(119.5, 42), (119.5, 36.6)])
dashed_path([(96, 24.6), (92.6, 24.6)])
dashed_path([(60, 24.6), (57.6, 24.6)])
dashed_path([(32, 36), (32, 37.6), (64, 37.6), (64, 41.4)])
dashed_path([(40, 36), (40, 40.6), (95, 40.6), (95, 41.4)])
txt(69.5, 39.1, "重训 fθ", 5.2, color=C_OP)
txt(99.5, 39.1, "重定 q", 5.2, color=C_OP)

# ================= 图例 =================
box(3, 129, 0.6, 10, "#FFFFFF", "#CCCCCC", 0.8)
for (x, fc, ec, lab) in [(4.5, C_DB_F, C_DB, "数据库/底模 (DB)"),
                         (27.0, C_ML_F, C_ML, "机器学习模型 (ML)"),
                         (49.5, C_ST_F, C_ST, "共形/可靠性 (统计)"),
                         (71.0, C_OP_F, C_OP, "闭环/运维")]:
    ax.add_patch(Rectangle((x, 6.7), 2.2, 2.2, fc=fc, ec=ec, lw=1.0, zorder=6))
    txt(x + 3.1, 7.8, lab, 5.6, ha="left")
txt(88.0, 7.8, "生命周期回环:监控 → 选择性取标签", 5.6, ha="left")
txt(88.0, 5.6, "→ 修复 → 独立审计", 5.6, ha="left")
arrow(4.5, 4.2, 10.0, 4.2)
txt(10.8, 4.2, "实线:外挂主流程 (1)–(5)", 5.6, ha="left")
dashed_path([(46, 4.2), (52, 4.2)])
txt(52.8, 4.2, "虚线:标签驱动的生命周期回环", 5.6, ha="left")
txt(66, 1.8, "※ 监控信号是有标签可靠性失效信号,不识别漂移原因;漂移破坏可交换性时不宣称 90% 覆盖保证。",
    5.4, color=C_INK)

# ★ 不走 vs.save：那张图的 bbox 会按 rcParams['savefig.bbox']="tight" 收紧，
#   把"固有宽 = 显示宽"这条约定破坏掉（本图 ax 铺满画布，tight 只会多出
#   pad_inches=0.02 的边，MediaBox 变成 7.07in 而非 DCOL 的 7.03in）。
#   这里显式传一个"整幅画布"的 Bbox，MediaBox 才恒等于 figsize。
#   带 CJK 的文本一律写成纯 Unicode（Q、b̂、ŷ、fθ、cov、e^q），绝不与 "$...$" 同串：
#   一行里同时出现 $math$ 与中文时，matplotlib 会把整串交给 STIX，中文全变豆腐块
#   （vs._guard_text 拦的正是这种写法）。纯公式行（如 $q=...$）没有 CJK，仍用 STIX。
vs._guard_text(fig, "arch")
BBOX = Bbox([[0, 0], fig.get_size_inches()])   # 单位 inch，等于整幅画布
for ext, dpi in (("pdf", None), ("png", 400)):
    fig.savefig(os.path.join(FIG, "arch." + ext), dpi=dpi, bbox_inches=BBOX,
                facecolor="white")
print("saved arch.{pdf,png} ->", FIG)
