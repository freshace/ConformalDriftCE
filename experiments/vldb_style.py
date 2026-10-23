# -*- coding: utf-8 -*-
"""论文所有图共用的 VLDB / ACM sigconf 出版级绘图样式。

核心约定（决定了图的可读性，务必遵守）：
    每张图的**固有宽度** = 它在论文里的**显示宽度**。
    单栏图 COL = 3.35 in，双栏图 DCOL = 7.03 in。
    因此 .tex 里一律 `width=\\linewidth`（单栏 figure）或 `width=\\textwidth`（双栏 figure*），
    **不做任何缩放**。若图固有宽度远大于显示宽度，字体会被等比压小到 2~3pt，
    这正是"图太小看不清 / 图例越界"的根因。

用法：
    import vldb_style as vs
    vs.apply()
    fig, ax = vs.subplots(1, 2, width=vs.COL)     # → figsize=(3.35, h)
    ...
    vs.save(fig, "fig_xxx")                        # 写 paper/figures/fig_xxx.{pdf,png}
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------- 尺寸
COL = 3.35    # acmart sigconf 单栏正文宽 ≈ 241.3pt ≈ 3.35in
DCOL = 7.03   # acmart sigconf 双栏正文宽 ≈ 505.9pt ≈ 7.03in

# ---------------------------------------------------------------- 配色
# Okabe–Ito 色盲安全族：低饱和、印刷友好。语义固定，全论文一致。
C_MAIN = "#0072B2"    # 主线 / 我们的方法（蓝）
C_ALT = "#D55E00"     # 对照 / 失效（朱红）
C_THIRD = "#009E73"   # 第三条臂（绿）
C_FOURTH = "#CC79A7"  # 第四条臂（品红）
C_FIFTH = "#8E6C9E"   # 第五条臂（紫）
C_REF = "#666666"     # 参考线 / 注释（灰）
C_LIGHT = "#BBBBBB"   # 背景柱 / 次要元素
C_INK = "#111111"
PALETTE = [C_MAIN, C_ALT, C_THIRD, C_FOURTH, C_FIFTH, C_REF]

_SERIF = ["Times New Roman", "SimSun", "DejaVu Serif"]


def _have_font(name):
    return any(f.name == name for f in font_manager.fontManager.ttflist)


def apply():
    """装载出版级 rcParams。必须在建图之前调用一次。"""
    # ★ 逐字形回退必须把候选族写在 font.family 这个**列表**里。
    #   只设 font.serif 时 matplotlib 会把 "SimSun" 当成路径去解析，CJK 全部丢字形。
    serif = [f for f in _SERIF if f == "DejaVu Serif" or _have_font(f)]
    matplotlib.rcParams.update({
        # ---- 字体：西文 Times，中文宋体（逐字形回退），数学 STIX（与 Times 同族）
        "font.family": serif,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        # ---- 字号：图固有宽度 = 显示宽度，故这里就是论文里的实际磅值
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.6,
        # ---- 线宽 / 刻度：细而清楚
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.minor.size": 1.3,
        "ytick.minor.size": 1.3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.1,
        "lines.markersize": 2.8,
        "patch.linewidth": 0.6,
        # ---- 图例：无框、紧凑
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.1,
        "legend.labelspacing": 0.35,
        "legend.borderaxespad": 0.3,
        # ---- 网格：默认关闭（VLDB 图以留白为主），需要时逐个 ax 打开
        "grid.linewidth": 0.4,
        "grid.alpha": 0.30,
        "grid.linestyle": ":",
        # ---- 导出：矢量文字可编辑 + 高分辨率位图
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.autolayout": False,
    })


def subplots(nr, nc, width=COL, height=None, **kw):
    """按"固有宽度 = 显示宽度"建图。height 缺省时按面板数给一个合理值。"""
    if height is None:
        height = 1.75 if nr == 1 else 1.55 * nr
    fig, axes = plt.subplots(nr, nc, figsize=(width, height), **kw)
    return fig, axes


def figsize(width=COL, height=1.8):
    return (width, height)


def panel_label(ax, s, dx=-0.02, dy=1.06, **kw):
    """左上角 (a)/(b) 面板标号，统一位置与字号。"""
    kw.setdefault("fontsize", 7.5)
    kw.setdefault("fontweight", "bold")
    kw.setdefault("va", "bottom")
    kw.setdefault("ha", "left")
    ax.text(dx, dy, s, transform=ax.transAxes, **kw)


def refline(ax, y, label=None, color=None, **kw):
    """目标水平线（如名义覆盖率 90%）。"""
    color = color or C_REF
    kw.setdefault("lw", 0.7)
    kw.setdefault("ls", (0, (4, 2.5)))
    ax.axhline(y, color=color, **kw)
    if label:
        ax.text(ax.get_xlim()[1], y, label, color=color, fontsize=6.4,
                ha="right", va="bottom")


def _is_cjk(ch):
    return ("　" <= ch <= "鿿") or ("＀" <= ch <= "￯")


def _guard_text(fig, stem):
    """硬门禁：禁止同一字符串里混排 CJK 与内联 $math$。

    含 `$` 的字符串整体交给 mathtext 渲染，其 rm 字体是 STIX（只有西文），
    于是同串里的中文会全部丢字形、被替换成豆腐块——而且 matplotlib 只发
    UserWarning 不会报错。这里改成直接失败，避免悄悄出图。
    需要符号时请直接写 Unicode（α ρ × → ≥ · ±），不要用 $\\alpha$。
    """
    import matplotlib.text as mtext
    bad = []
    for t in fig.findobj(mtext.Text):
        s = t.get_text()
        if "$" in s and any(_is_cjk(c) for c in s):
            bad.append(s)
    if bad:
        for s in bad:
            print(f"  [BAD] 中英混排丢字形: {s!r}")
        sys.exit(f"{stem}: {len(bad)} 个字符串同时含中文与 $math$，"
                 f"请改用 Unicode 符号（α ρ × → ≥ · ±）")


def save(fig, stem, width=None, height=None, outdir=None):
    """写 <repo>/paper/figures/<stem>.{pdf,png}，返回输出目录。"""
    _guard_text(fig, stem)
    here = os.path.dirname(os.path.abspath(__file__))
    figdir = outdir or os.path.join(os.path.dirname(here), "paper", "figures")
    os.makedirs(figdir, exist_ok=True)
    for ext, dpi in (("pdf", None), ("png", 400)):
        p = os.path.join(figdir, f"{stem}.{ext}")
        fig.savefig(p, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"saved {os.path.join(figdir, stem)}.pdf|png")
    return figdir


def ci95(a):
    """重复实验的 mean 与 95% CI 半宽。"""
    import numpy as np
    a = np.asarray(a, dtype=float)
    if a.size < 2:
        return float(a.mean()) if a.size else float("nan"), 0.0
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(a.size))


def fmt_pm(mean, half, nd=2, scale=1.0):
    return f"{mean*scale:.{nd}f}±{half*scale:.{nd}f}"


def check(tag, got, want, tol=1e-6):
    """硬断言：图上取到的数字必须与产物（= 论文数字）一致，否则拒绝出图。"""
    ok = abs(float(got) - float(want)) <= tol
    print(f"  [{'OK ' if ok else 'BAD'}] {tag}: got {got!r}, paper {want!r}")
    return ok


class Assertor:
    """收集所有 check 结果；末尾 assert_all() 一次性判定。"""

    def __init__(self):
        self.fails = []

    def approx(self, tag, got, want, tol=1e-6):
        if not check(tag, got, want, tol):
            self.fails.append(tag)
        return got

    def close(self, tag, got, want, tol=1e-6):
        return self.approx(tag, got, want, tol)

    def between(self, tag, got, lo, hi):
        ok = lo - 1e-9 <= float(got) <= hi + 1e-9
        print(f"  [{'OK ' if ok else 'BAD'}] {tag}: {got!r} in [{lo}, {hi}]")
        if not ok:
            self.fails.append(tag)
        return got

    def assert_all(self, name):
        if self.fails:
            msg = f"{name}: {len(self.fails)} 项与论文数字不符 -> " + "; ".join(self.fails)
            print("FAIL " + msg)
            sys.exit(msg)
        print(f"OK   {name}: 全部数值与论文一致")


# 未调用 apply() 就建图时，给出明确报错而不是画出一张错版图
_orig_subplots = plt.subplots
