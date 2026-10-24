# -*- coding: utf-8 -*-
"""Regenerate every manuscript figure in English with the original drawing code.

Each function below mirrors, statement for statement, the archived Chinese renderer
that produced the figure currently embedded in the manuscript. Only the string
arguments change (Chinese -> English); canvases, subplot rectangles, colours,
markers, bootstrap seeds, statistical definitions and annotation offsets are kept
identical, so the English figure is a pure language variant of the Chinese one.

Sources of the drawing code
---------------------------
* ``figures/scripts/original_style_reference.py``  -> Fig. 2, Fig. 6
* ``paper/2026-10-16/generation_scripts/redraw_layout.py`` -> Fig. 3a/b, Fig. 4, Fig. 7, Fig. 11a/b
* ``paper/2026-10-16/generation_scripts/mk_fig5_wide.py`` -> Fig. 5
* ``figures/scripts/figures.py``                   -> Fig. 8, Fig. 9
* ``figures/scripts/real_cost_cn.py``              -> Fig. 10
* ``paper/2026-10-14/english_restructured/authoring/01_architecture_focus.tex`` -> Fig. 1 (TikZ)

Canvas sizes are the page sizes of the archived Chinese PDFs, so every English
figure keeps the exact aspect ratio of the Chinese figure it replaces.

Run:  python make_figures_en.py
"""

from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[5]          # .../drift_ce
HERE = Path(__file__).resolve().parent
OUT = HERE.parent                                   # .../manuscript_cn/figures_en
REV = ROOT / "provenance/2026-10-14"
PLOT = ROOT / "paper/2026-10-16/plot_data"
TAB = ROOT / "data/tables"
SUP = ROOT / "data/supplement"

os.environ.setdefault("MPLCONFIGDIR", str(HERE / "mplcache"))
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Same rcParams as the archived renderers.
mpl.rcParams.update({
    "font.family": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "axes.unicode_minus": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "mathtext.fontset": "dejavusans",
})

DPI = 600
MANIFEST = []

# Canvas = page size of the archived Chinese PDF (inches).
CANVAS = {
    "02_necessity": (3.650, 2.480),
    "03_failure_modes_a": (3.283, 2.223),
    "03_failure_modes_b": (3.283, 2.387),
    "ch4_width_fan": (3.515, 2.497),
    "ch4_feedback_wide": (7.000, 2.050),
    "06_cross_dataset": (7.000, 2.507),
    "09_data_level_merged": (3.283, 2.390),
    "ch4_detection_compact": (3.107, 3.903),
    "continuous_repair": (6.833, 5.063),
    "real_cost": (7.000, 2.600),
    "ch4_plan_verified_a": (3.247, 2.390),
    "ch4_plan_verified_b": (3.247, 2.390),
}


def ci(values, seed=61014, repeats=10000):
    """Same bootstrap CI as the archived scripts."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    boot = np.random.default_rng(seed).choice(a, (repeats, len(a)), replace=True).mean(1)
    return np.array([a.mean(), *np.quantile(boot, [.025, .975])])


def ci_rows(values, seed=61014, repeats=10000):
    """redraw_layout.ci: means/quantiles along axis 0, for split-by-factor tables."""
    v = np.asarray(values, dtype=float)
    boot = v[np.random.default_rng(seed).integers(0, len(v), (repeats, len(v)))].mean(axis=1)
    low, high = np.quantile(boot, [.025, .975], axis=0)
    return v.mean(axis=0), low, high


def err(ax, x, v, axis="y", **kw):
    m, lo, hi = ci(v)
    delta = np.array([[m - lo], [hi - m]])
    if axis == "y":
        ax.errorbar(x, m, yerr=delta, capsize=2, lw=.8, **kw)
    else:
        ax.errorbar(m, x, xerr=delta, capsize=2, lw=.8, **kw)
    return m


def tag(ax, letter, title, y=1.09, dx=.065, fs=8):
    ax.text(0, y, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")
    ax.text(dx, y, title, transform=ax.transAxes, fontsize=fs, va="bottom")


def save(fig, name):
    for ax in fig.axes:
        if ax.get_xscale() == "log" or ax.get_yscale() == "log":
            ax.minorticks_off()
    fig.canvas.draw()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=DPI, facecolor="white")
    MANIFEST.append({"figure": name, "canvas_in": list(fig.get_size_inches()), "dpi": DPI})
    plt.close(fig)
    print(f"  wrote {name}")


# --------------------------------------------------------------------------- Fig. 2
def fig02_necessity():
    """original_style_reference.motivation()"""
    d = pd.read_csv(REV / "tables/motivation.csv")
    f = plt.figure(figsize=CANVAS["02_necessity"])
    a = f.add_axes([.15, .23, .66, .64])
    b = a.twinx()
    fac = sorted(d.factor.unique())
    cov = np.array([d[d.factor == x].coverage.mean() * 100 for x in fac])
    gm = np.array([d[d.factor == x].gmq.mean() for x in fac])
    a.bar(0, cov[0], color="#AAAAAA", width=.56)
    for i in range(1, len(fac)):
        a.bar(i, cov[i - 1] - cov[i], bottom=cov[i], color="#E78132", width=.62)
        a.plot([i - .65, i + .12], [cov[i - 1]] * 2, color="#888", ls="--", lw=.75)
        a.text(i, 101, f"{cov[i] - cov[i - 1]:.1f}", ha="center", fontsize=8, color="#D55E00")
    a.bar(6, cov[-1], color="#E78132", width=.62)
    a.axhline(cov[0], color="#777", ls=":", lw=.8)
    a.text(-.3, cov[0] + 4, f"{cov[0]:.1f}%", color="#777", fontweight="bold")
    a.text(6, cov[-1] + 3, f"{cov[-1]:.1f}%", ha="center", color="#D55E00", fontweight="bold")
    b.plot(range(6), gm, "o-", color="#20282D", ms=4, lw=1.5)
    for i, x in enumerate(fac):
        err(a, i, d[d.factor == x].coverage * 100, fmt="none", ecolor="#B25F25")
        err(b, i, d[d.factor == x].gmq, fmt="none", ecolor="#20282D")
    for i in [0, 5]:
        b.annotate(f"{gm[i]:.2f}", (i, gm[i]),
                   xytext=(8, -11) if i == 0 else (-5, -28), textcoords="offset points", fontsize=8)
    a.set_ylim(0, 107)
    a.set_yticks([0, 30, 60, 90])
    a.set_xticks(range(7), [f"{v:g}" if v in [1, 2] else f"{v:.2f}" for v in fac] + ["Final"])
    a.set_ylabel("Coverage (%)", color="#D55E00")
    a.set_xlabel("Correlation multiplier\nlabels = coverage loss (pp)")
    b.set_ylim(1, 6.8)
    b.set_ylabel("GMQ (lower is better)")
    b.spines["right"].set_visible(True)
    save(f, "02_necessity")


# --------------------------------------------------------------------------- Fig. 3
ARM_ORDER = ["frozen", "recal_only", "repair", "garbage"]
ARM_COLORS = ["#8A8A8A", "#0072B2", "#009E73", "#E57827"]
ARM_MARKERS = ["o", "s", "^", "D"]
ARM_NAMES = ["Frozen", "Recal. only", "Repaired", "Catastrophic"]


def panel(size, letter, title, left=.19, right=.94, bottom=.23, top=.82):
    """redraw_layout.panel()"""
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    ax.text(0, 1.09, letter, transform=ax.transAxes, weight="bold", va="bottom", fontsize=9)
    ax.text(.065, 1.09, title, transform=ax.transAxes, va="bottom", fontsize=8)
    return fig, ax


def fig03_failure_modes():
    """redraw_layout.mechanisms()"""
    d = pd.read_csv(PLOT / "mechanisms.csv")
    fa, a = panel(CANVAS["03_failure_modes_a"], "a", "Self-test matches target; audit splits")
    fb, b = panel(CANVAS["03_failure_modes_b"], "b", "Interval cost and point error")
    for j, (arm, color, marker, name) in enumerate(zip(ARM_ORDER, ARM_COLORS, ARM_MARKERS, ARM_NAMES)):
        g = d[d.stage == arm]
        assert len(g) == 50
        m = [g.self_coverage.mean() * 100, g.coverage.mean() * 100]
        for row in g.itertuples():
            a.plot([0, 1], [row.self_coverage * 100, row.coverage * 100],
                   color=color, lw=.35, alpha=.12, zorder=1)
            a.scatter([0, 1], [row.self_coverage * 100, row.coverage * 100],
                      color=color, s=3, alpha=.18, zorder=2)
        a.plot([0, 1], m, marker=marker, ms=4, color=color,
               ls="-" if j == 0 else "--", lw=.9, mfc="white")
        for x, key in enumerate(["self_coverage", "coverage"]):
            v, low, high = ci(g[key] * 100)
            a.errorbar(x, v, yerr=[[v - low], [high - v]], fmt="none", ecolor=color, capsize=2, lw=.8)
        a.text(1.14, [65, 97, 80, 87][j], f"{name}\n{m[1]:.1f}%", color=color, fontsize=7, va="center")
        b.scatter(g.width, g.q50, color=color, marker=marker, s=9, alpha=.28, edgecolors="none")
        x, xl, xh = ci(g.width)
        y, yl, yh = ci(g.q50)
        b.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]],
                   fmt=marker, ms=5, color=color, capsize=2, mec="white", mew=.6)
        b.annotate(f"{name}\n{x:.1f} / {y:.2f}", (x, y),
                   xytext=[(5, 15), (14, 17), (12, -3), (-36, -27)][j],
                   textcoords="offset points", fontsize=6.8, color=color)
    a.axhline(90, color="#888", ls=":", lw=.7)
    a.set(xlim=(-.15, 1.75), ylim=(59, 102), ylabel="Coverage (%)")
    a.set_xticks([0, 1], ["Self-test", "Independent audit"])
    b.set(xscale="log", yscale="log", xlim=(5, 3000), ylim=(1, 180),
          xlabel="Mean normalized length W (tighter left)",
          ylabel="Median q-error (lower better)")
    b.set_xticks([10, 100, 1000], ["10", "100", "1000"])
    b.set_yticks([1, 10, 100], ["1", "10", "100"])
    b.minorticks_off()
    save(fa, "03_failure_modes_a")
    save(fb, "03_failure_modes_b")


# --------------------------------------------------------------------------- Fig. 4
def fig04_width_fan():
    """redraw_layout.width_geometry()"""
    z = np.load(PLOT / "residual_sensitivity_raw.npz")
    with mpl.rc_context({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7}):
        fig, (a, b) = plt.subplots(1, 2, figsize=CANVAS["ch4_width_fan"])
        fig.subplots_adjust(left=.17, right=.98, bottom=.20, top=.78, wspace=.72)
        for ax, letter, title in [(a, "a", "Interval geometry"), (b, "b", "Frozen coverage")]:
            ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(0, 10),
                        textcoords="offset points", fontsize=8, weight="bold", va="bottom")
            ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(10, 10),
                        textcoords="offset points", fontsize=7, va="bottom")
        t = np.linspace(0, 1, 120)
        endpoint = 1.5
        x = endpoint * t
        gray, green = "#8A8F98", "#1B8A65"
        for tag_, cov, w, col, style, marker in [
            ("Base", z["base_matched_coverage"] * 100, z["base_matched_width"], gray, "--", "s"),
            ("Residual", z["residual_coverage"] * 100, z["residual_width"], green, "-", "o"),
        ]:
            wm, wl, wh = ci_rows(w, 20261006, 5000)
            cm, _, _ = ci_rows(cov, 20261006, 5000)
            lower, upper = np.exp(-np.log(wm) * t ** 2), np.exp(np.log(wm) * t ** 2)
            a.fill_between(x, lower, upper, color=col, alpha=.13)
            for v in w:
                a.plot(x, np.exp(np.log(v) * t ** 2), color=col, alpha=.20, lw=.5)
                a.plot(x, np.exp(-np.log(v) * t ** 2), color=col, alpha=.20, lw=.5)
            for edge in [lower, upper]:
                a.plot(x, edge, color=col, ls=style, lw=1.3)
            a.scatter(np.full(10, endpoint), w, s=8, color=col, marker=marker, alpha=.4)
            a.scatter(np.full(10, endpoint), 1 / w, s=8, color=col, marker=marker, alpha=.4)
            a.errorbar(endpoint, wm, yerr=[[wm - wl], [wh - wm]], fmt=marker, color=col, capsize=2, ms=4)
            a.errorbar(endpoint, 1 / wm, yerr=[[1 / wm - 1 / wh], [1 / wl - 1 / wm]],
                       fmt=marker, color=col, capsize=2, ms=4)
            a.annotate(f"{tag_}\n{wm:.2f}×\n{cm:.2f}%", (endpoint, wm), xytext=(3, -1),
                       textcoords="offset points", va="top", fontsize=6.8,
                       color="#252525" if tag_ == "Base" else green)
        a.set(yscale="log", ylim=(.0015, 550), xlim=(-.02, 2.25),
              ylabel="Ratio to point prediction", xlabel="Expansion position")
        a.set_yticks([.01, .1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
        a.set_xticks([0, endpoint], ["Center", "Interval end"])
        a.minorticks_off()
        for name, key, col, marker, style in [("Matched Base", "match_cov", gray, "s", "--"),
                                              ("Residual", "res_cov", green, "o", "-")]:
            m, low, high = ci_rows(z[key] * 100, 20261006, 5000)
            b.errorbar(z["dose"][0], m, yerr=np.stack([m - low, high - m]),
                       fmt=marker + style, color=col, capsize=2, lw=1)
        b.axhline(90, color="#252525", ls=":", lw=.7)
        b.set(xscale="log", xlim=(.92, 8.3), ylim=(55, 95),
              xlabel="Drift factor", ylabel="Coverage (%)")
        b.set_xticks([1, 2, 4, 7.39], ["1", "2", "4", "7.39"])
        b.set_yticks([60, 70, 80, 90])
        b.minorticks_off()
        fig.legend(handles=[Line2D([], [], color=gray, ls="--", marker="s", label="Matched Base"),
                            Line2D([], [], color=green, marker="o", label="Residual"),
                            Line2D([], [], color="#252525", ls=":", label="90% target")],
                   loc="upper center", bbox_to_anchor=(.56, .99), ncol=3,
                   fontsize=6.7, columnspacing=.75, handlelength=1.7)
        save(fig, "ch4_width_fan")


# --------------------------------------------------------------------------- Fig. 5
def fig05_feedback_wide():
    """mk_fig5_wide.py"""
    d = pd.read_csv(REV / "tables/feedback_unified.csv")
    d = d[(d.pool == "pool_age.pkl") & (d.state == "age_purge50_posts") & (d.feedback_budget == 60)]
    f, (a, b, c) = plt.subplots(1, 3, figsize=CANVAS["ch4_feedback_wide"])
    f.subplots_adjust(left=.068, right=.985, bottom=.255, top=.755, wspace=.50)
    order = ["explore_00", "explore_05", "explore_10", "explore_20", "uniform"]
    ticks = ["0%", "5%", "10%", "20%", "Uniform"]
    for j, st in enumerate(order):
        g = d[d.strategy == st]
        err(a, j, g.feedback_coverage_bias * 100, fmt="o", color="#252525", ms=4)
        err(b, j, g.zero_inclusive_coverage * 100, fmt="D", color="#16946E", ms=4)
        vals = g.median_normalized_length.to_numpy()
        rng = np.random.default_rng(61014 + j)
        c.scatter(vals, j + rng.uniform(-.13, .13, len(vals)), s=7, color="#159778",
                  alpha=.32, edgecolors="none")
        c.boxplot([vals], positions=[j], orientation="horizontal", widths=.42,
                  patch_artist=True, showfliers=False, manage_ticks=False,
                  boxprops={"facecolor": "#ADDACB", "edgecolor": "#72BFA8", "linewidth": .7},
                  medianprops={"color": "#127E64"}, whiskerprops={"color": "#72BFA8"},
                  capprops={"color": "#72BFA8"})
    for ax in (a, b):
        ax.set_xticks(range(5), ticks)
        ax.set_xlabel("Uniform exploration")
    a.axhline(0, color="#888", ls=":", lw=.7)
    a.set_ylabel("Feedback coverage bias (pp)")
    b.set_ylabel("Coverage (%)")
    b.set_ylim(60, 102)
    b.axhline(90, color="#777", ls=":", lw=.7)
    c.set_xscale("log")
    # keep every decade labelled, as in the Chinese original
    c.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=20))
    c.set_yticks(range(5), ticks)
    c.invert_yaxis()
    c.set_xlabel("Audit median normalized length W")
    c.set_ylabel("Uniform exploration")
    for ax, letter, title in zip((a, b, c), "abc",
                                 ["Feedback coverage bias", "Overall audit coverage",
                                  "Interval-length distribution"]):
        ax.text(0, 1.10, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")
        ax.text(.065, 1.10, title, transform=ax.transAxes, fontsize=8, va="bottom")
    save(f, "ch4_feedback_wide")


# --------------------------------------------------------------------------- Fig. 6
def fig06_cross_dataset():
    """original_style_reference.cross()"""
    d = pd.read_csv(REV / "tables/cross_all.csv")
    val = np.zeros((4, 6))
    texts = {}
    for i, ds in enumerate(["stats", "imdb", "ergastf1", "genome"]):
        for j, cond in enumerate(["corr", "work", "schema"]):
            for k, st in enumerate(["frozen", "repair"]):
                x = ci(d[(d.dataset == ds) & (d.condition == cond) & (d.stage == st)].coverage * 100)
                val[i, 2 * j + k] = x[0]
                texts[i, 2 * j + k] = x
    f, ax = plt.subplots(figsize=CANVAS["06_cross_dataset"])
    f.subplots_adjust(left=.10, right=.91, bottom=.15, top=.83)
    im = ax.imshow(val, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for (i, j), v in texts.items():
        ax.text(j, i - .08, f"{v[0]:.1f}", ha="center", va="center", fontweight="bold",
                fontsize=9, color="white" if v[0] > 75 else "#24333D")
        ax.text(j, i + .22, f"{v[1]:.1f}–{v[2]:.1f}", ha="center", va="center", fontsize=7,
                color="white" if v[0] > 75 else "#24333D")
    ax.set_yticks(range(4), ["stats", "imdb", "ergastf1", "genome"])
    ax.set_xticks(range(6), ["Frozen", "Repaired"] * 3)
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for j, title in enumerate(["Correlation drift", "Workload drift", "Schema drift"]):
        ax.text((2 * j + 1) / 6, 1.16, title, ha="center", transform=ax.transAxes,
                fontweight="bold", fontsize=9)
    f.colorbar(im, ax=ax, fraction=.025, pad=.045, label="Coverage (%)")
    save(f, "06_cross_dataset")


# --------------------------------------------------------------------------- Fig. 7
def fig07_data_level_merged():
    """redraw_layout.merged_realdata()"""
    data = json.loads((PLOT / "realdata.json").read_text(encoding="utf-8"))
    fig, ax = panel(CANVAS["09_data_level_merged"], "", "All queries and the correlation-sensitive subset", top=.71)
    handles = []
    for k, (kind, base, color, title) in enumerate([
        ("", "base_cov", "#0072B2", "All"),
        ("_sens", "base_cov_sens", "#009E73", "Sensitive"),
    ]):
        baseline = np.mean([r["results"][base] for r in data]) * 100
        ax.axhline(baseline, color=color, ls="--", lw=.7, alpha=.8)
        for j, state in enumerate(["f", "r"]):
            values = [np.array([r["results"][str(rho)]["cov_" + state + kind] for r in data]) * 100
                      for rho in [.3, .45, .6, .75, .9]]
            positions = np.array([.3, .45, .6, .75, .9]) + (2 * k + j - 1.5) * .013
            face = "white" if state == "f" else mpl.colors.to_rgba(color, .32)
            ax.boxplot(values, positions=positions, widths=.010, patch_artist=True, showfliers=True,
                       flierprops={"marker": ".", "markersize": 1.6, "markeredgecolor": color,
                                   "markerfacecolor": color, "alpha": .4},
                       manage_ticks=False,
                       boxprops={"facecolor": face, "edgecolor": color, "linewidth": .7},
                       medianprops={"color": color, "linewidth": .7},
                       whiskerprops={"color": color, "linewidth": .7},
                       capprops={"color": color, "linewidth": .7})
            for x, values_at_x in zip(positions, values):
                ax.plot(x, values_at_x.mean(), "o", mfc="white", mec=color, ms=2.6)
            last = values[-1].mean()
            ax.text(.95, last, f"{last:.1f}%", color=color, fontsize=7, va="center")
            handles.append(Patch(facecolor=face, edgecolor=color,
                                 label=title + (" · Frozen" if state == "f" else " · Repaired")))
    ax.axhline(90, color="#888", ls=":", lw=.5)
    ax.set(ylim=(0, 105), xlim=(.255, 1.055), xlabel="Within-group correlation ρ", ylabel="Coverage (%)")
    ax.set_xticks([.3, .45, .6, .75, .9], ["0.30", "0.45", "0.60", "0.75", "0.90"])
    fig.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(.55, 1.0), fontsize=7)
    save(fig, "09_data_level_merged")


# --------------------------------------------------------------------------- Fig. 8
def fig08_detection():
    """figures.figures.detection()"""
    z = np.load(REV / "detection/detection_ci.npz", allow_pickle=True)
    v = np.asarray(z["recs"], float).reshape(50, 44, 7)
    f = plt.figure(figsize=CANVAS["ch4_detection_compact"])
    gs = f.add_gridspec(2, 2, left=.24, right=.95, bottom=.15, top=.78,
                        height_ratios=[1, 1.15], hspace=.75, wspace=.38)
    a = f.add_subplot(gs[0, :])
    b = f.add_subplot(gs[1, 0])
    c = f.add_subplot(gs[1, 1])
    factors = [1., 1.35, 2., 2.72, 4.48, 7.39]
    for det, col, mk, ls, name in [
        (0, "#2166AC", "o", "-", "Coverage (labeled)"),
        (1, "#A65E83", "^", "--", "Residual (labeled)"),
        (3, "#7283AD", "s", ":", "Feature mean (unlabeled)"),
    ]:
        means = []
        for fac in factors:
            ix = np.flatnonzero((v[0, :, 0] == 0) & np.isclose(v[0, :, 1], fac) & (v[0, :, 2] == det))[0]
            arr = v[:, ix, 3] * 100
            err(a, fac, arr, fmt=mk, color=col, ms=3)
            means.append(arr.mean())
        a.plot(factors, means, color=col, ls=ls, lw=1, label=name)
    a.set_ylim(0, 107)
    a.set_xscale("log")
    a.minorticks_off()
    a.set_xticks([1, 2, 4, 7.39], ["1", "2", "4", "7.39"])
    a.set_xlabel("Correlation multiplier")
    a.set_ylabel("Alarm rate (%)")
    a.tick_params(axis="both", labelsize=9.5)
    tag(a, "a", "Monitoring alarms")
    labels = []
    for fam, faclist in [(0, [1.35, 2, 2.72, 4.48, 7.39]), (1, [2, 5, 10, 50])]:
        bi = np.flatnonzero((v[0, :, 0] == fam) & (v[0, :, 1] == 1) & (v[0, :, 2] == 3))[0]
        for fac in faclist:
            ix = np.flatnonzero((v[0, :, 0] == fam) & np.isclose(v[0, :, 1], fac) & (v[0, :, 2] == 3))[0]
            j = len(labels)
            labels.append(("Corr ×" if fam == 0 else "Scale ×") + f"{fac:g}")
            loss = (v[:, bi, 5] - v[:, ix, 5]) * 100
            inc = (v[:, ix, 3] - v[:, bi, 3]) * 100
            for ax, arr, col in [(b, loss, "#D8A477"), (c, inc, "#91A0C4")]:
                m, l, h = ci(arr)
                ax.barh(j, m, color=col, height=.57)
                ax.errorbar(m, j, xerr=[[m - l], [h - m]], fmt="none", ecolor=col, capsize=1.4, lw=.8)
    for ax in [b, c]:
        ax.set_yticks(range(len(labels)), labels if ax is b else [])
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=6.8)
    b.set_xlabel("Loss (pp)")
    c.set_xlabel("Increase (pp)")
    # Match panel a's physical label-to-title offset on the narrower panels.
    title_offset = .065 * a.get_position().width
    tag(b, "b", "Audit coverage", dx=title_offset / b.get_position().width)
    tag(c, "c", "Feature alarms", dx=title_offset / c.get_position().width)
    h, l = a.get_legend_handles_labels()
    f.legend(h, l, loc="upper center", ncol=1, bbox_to_anchor=(.56, .995), fontsize=7)
    save(f, "ch4_detection_compact")


# --------------------------------------------------------------------------- Fig. 9
POL = ["frozen", "frozen_trigger", "current_coverage", "current_residual", "periodic", "random"]
COL = ["#888888", "#D28C45", "#0072B2", "#009E73", "#9466A4", "#AF725B"]
MARK = ["o", "s", "^", "D", "v", "X"]
POL_NAMES = ["Frozen", "Initial ref.", "Current cov.", "Current residual", "Periodic", "Random"]
POL_SHORT = ["Frozen", "Initial", "Coverage", "Residual", "Periodic", "Random"]


def fig09_continuous_repair():
    """figures.figures.continuous()"""
    # stats/imdb live in data/tables, ergastf1/genome in data/continuous.
    wins = pd.concat([pd.read_csv(TAB / "offline_windows.csv"),
                      pd.read_csv(ROOT / "data/continuous/offline_windows.csv")],
                     ignore_index=True)
    runs = pd.concat([pd.read_csv(TAB / "offline_runs.csv"),
                      pd.read_csv(ROOT / "data/continuous/offline_runs.csv")],
                     ignore_index=True)
    dbs = ["stats", "imdb", "ergastf1", "genome"]
    f, axs = plt.subplots(3, 2, figsize=CANVAS["continuous_repair"])
    f.subplots_adjust(left=.09, right=.98, bottom=.16, top=.88, wspace=.35, hspace=.72)
    for ax, ds, letter in zip(axs[:2].ravel(), dbs, "abcd"):
        for pol, cm, mk, nm in zip(POL, COL, MARK, POL_NAMES):
            g = wins[(wins.dataset == ds) & (wins.pattern == "cycles") & (wins.policy == pol)]
            v = g.pivot(index="seed", columns="window", values="coverage").to_numpy() * 100
            ix = np.random.default_rng(61012).integers(0, len(v), (2000, len(v)))
            lo, hi = np.quantile(v[ix].mean(1), [.025, .975], axis=0)
            x = np.arange(1, 81)
            mean = v.mean(0)
            ax.plot(x, mean, color=cm, lw=.8, marker=mk, markevery=8, ms=2.2)
            take = np.arange(0, 80, 8)
            ax.errorbar(x[take], mean[take], yerr=np.array([mean - lo, hi - mean])[:, take],
                        fmt="none", color=cm, capsize=1, lw=.45, alpha=.75)
        for start in [16, 48]:
            ax.axvline(start + .5, color="#999", ls=":", lw=.6)
            ax.axvline(start + 16.5, color="#999", ls=":", lw=.6)
        ax.axhline(90, color="#777", ls=":", lw=.6)
        ax.set_ylim(0, 102)
        ax.set_xticks([1, 16, 32, 48, 64, 80])
        ax.set_xlabel("Window (repair active next window)")
        ax.set_ylabel("Audit coverage (%)")
        tag(ax, letter, ds)
    for ax, subset, letter in zip(axs[2], [dbs[:2], dbs[2:]], "ef"):
        for j, (pol, cm, nm) in enumerate(zip(POL, COL, POL_NAMES)):
            for k, ds in enumerate(subset):
                g = runs[(runs.dataset == ds) & (runs.pattern == "cycles") & (runs.policy == pol)]
                m, lo, hi = ci(g.deployment_labels)
                x = j * 3 + k
                ax.bar(x, 1600, color="#D5D9DF", width=.70)
                ax.bar(x, m - 1600, bottom=1600, color=cm, width=.70)
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=cm, capsize=1.5, lw=.7)
            ax.text(j * 3 + .5, -.16, POL_SHORT[j], ha="right", va="top", rotation=40,
                    transform=ax.get_xaxis_transform(), fontsize=6.5)
        ax.set_xticks([])
        ax.set_ylim(0, 10300)
        ax.set_yticks([0, 4000, 8000])
        ax.set_ylabel("Consumed online labels")
        tag(ax, letter, " / ".join(subset))
    f.legend(handles=[Line2D([], [], color=c, marker=mk, lw=.8, label=n, ms=3)
                      for c, mk, n in zip(COL, MARK, POL_NAMES)],
             ncol=3, loc="upper center", bbox_to_anchor=(.53, .995), fontsize=7)
    save(f, "continuous_repair")


# -------------------------------------------------------------------------- Fig. 10
RC_NAME = dict(zip(POL, ["No update", "Initial reference", "Current coverage",
                         "Current residual", "Periodic", "Random"]))
RC_SHORT = dict(zip(POL, ["No update", "Initial ref.", "Current cov.",
                          "Current resid.", "Periodic", "Random"]))


def fig10_real_cost():
    """figures.scripts.real_cost_cn.py"""
    rc = dict(mpl.rcParams)
    mpl.rcParams.update({"font.family": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    runs = pd.read_csv(SUP / "sql_runs.csv")
    wins = pd.read_csv(SUP / "sql_windows.csv")
    CW = "workload_cycles"
    FOCAL = ["frozen", "current_coverage"]

    def boot_ci(v, seed=61012, n=10000):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        bs = np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))
        m = v[bs].mean(1)
        return v.mean(), np.quantile(m, .025), np.quantile(m, .975)

    fig = plt.figure(figsize=CANVAS["real_cost"])
    gs = fig.add_gridspec(1, 3, left=.112, right=.982, bottom=.235, top=.735,
                          wspace=1.02, width_ratios=[1.18, 1.12, 1.0])
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[0, 2])

    W = int(wins[wins.condition == CW].window.max()) + 1
    x = np.arange(W)
    for j, pol in enumerate(POL):
        g = wins[(wins.condition == CW) & (wins.policy == pol)]
        m = g.groupby("window").coverage.mean().reindex(x).to_numpy() * 100
        focal = pol in FOCAL
        axa.plot(x, m, color=COL[j], lw=1.5 if focal else .9,
                 alpha=1.0 if focal else .45, zorder=4 if focal else 2,
                 marker="o" if focal else None, ms=2.6, mew=0, label=RC_NAME[pol])
    for w in (6, 12, 18, 24):
        axa.axvline(w - .5, color="#9aa3ad", lw=.6, ls=(0, (2, 2.5)), zorder=1)
    axa.text(6, 101.5, "switch", ha="center", va="bottom", fontsize=6.2, color="#6b7480")
    axa.axhline(90, color="#777777", lw=.65, ls=":", zorder=1)
    axa.set_xlim(-.6, W - .4)
    axa.set_ylim(55, 106)
    axa.set_yticks([60, 70, 80, 90, 100])
    axa.set_xlabel("Window")
    axa.set_ylabel("Audit coverage (%)")
    axa.text(0, 1.045, "a", transform=axa.transAxes, fontweight="bold", fontsize=9, va="bottom")
    axa.text(.062, 1.045, "Audit coverage", transform=axa.transAxes, fontsize=8, va="bottom")

    grp = runs[runs.condition == CW].groupby("policy")
    yy = np.arange(len(POL))
    mon = np.asarray([grp.get_group(k).monitor_wall_seconds.mean() for k in POL])
    rep = np.asarray([grp.get_group(k).repair_wall_seconds.mean() for k in POL])
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
    axb.text(mon[j0] / 2 + .8, yy[j0], "mon.", ha="left", va="center", fontsize=6.4, color="#33485c")
    axb.text(mon[j0] + rep[j0] / 2, yy[j0], "rep.", ha="center", va="center", fontsize=6.4, color="white")
    axb.set_yticks(yy, [RC_SHORT[k] for k in POL])
    axb.invert_yaxis()
    axb.tick_params(axis="y", length=0, pad=2)
    axb.set_xlabel("Measured online time (s)")
    axb.set_xlim(0, max(tot) * 1.28)
    axb.text(0, 1.045, "b", transform=axb.transAxes, fontweight="bold", fontsize=9, va="bottom")
    axb.text(.062, 1.045, "Cost breakdown", transform=axb.transAxes, fontsize=8, va="bottom")

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
    axc.set_xlabel("Online labels")
    axc.set_ylabel("Coverage (%)")
    axc.set_ylim(78, 97)
    axc.set_xlim(60, 1300)
    axc.text(.98, .04, "filled: correlation control\nopen: joint change", transform=axc.transAxes,
             ha="right", va="bottom", fontsize=6.4, color="#5b6470", linespacing=1.35)
    axc.text(0, 1.045, "c", transform=axc.transAxes, fontweight="bold", fontsize=9, va="bottom")
    axc.text(.062, 1.045, "Coverage vs labels", transform=axc.transAxes, fontsize=8, va="bottom")

    handles = [Line2D([], [], color=COL[i], marker=MARK[i], lw=1.0, ms=3, label=RC_NAME[POL[i]])
               for i in range(len(POL))]
    fig.legend(handles=handles, ncol=6, loc="upper center", bbox_to_anchor=(.52, .995),
               columnspacing=1.1, handlelength=1.5, handletextpad=.5)
    save(fig, "real_cost")
    mpl.rcParams.update(rc)


# -------------------------------------------------------------------------- Fig. 11
def fig11_plan_verified():
    """redraw_layout.utility()"""
    rows = []
    for condition in ["correlation", "workload"]:
        d = pd.read_csv(PLOT / f"utility_{condition}.csv")
        for state, model in [("before", "frozen"), ("after", "frozen"), ("after", "active")]:
            r = d[(d.state == state) & (d.model == model)].iloc[0].to_dict()
            title = "Corr." if condition == "correlation" else "Joint"
            label = "Baseline" if state == "before" else ("Frozen" if model == "frozen" else "Repaired")
            rows.append(dict(label=title + " / " + label, **r))
    fa, a = panel(CANVAS["ch4_plan_verified_a"], "a", "Queries changing plan", left=.34, right=.97)
    fb, b = panel(CANVAS["ch4_plan_verified_b"], "b", "Paired execution times", left=.34, right=.97)
    for j, r in enumerate(rows):
        y = 5 - j
        v = r["changed_plan_queries"] / r["queries"] * 100
        a.barh(y, v, height=.45, color="#1B8A65", alpha=.7)
        a.text(v + .5, y, f'{int(r["changed_plan_queries"])}/{int(r["queries"])}', va="center", fontsize=7)
        m = (r["geomean_point_over_interval_runtime"] - 1) * 100
        low, high = (r["ci_low"] - 1) * 100, (r["ci_high"] - 1) * 100
        b.errorbar(m, y, xerr=[[m - low], [high - m]], fmt="o", color="#2569A8", capsize=2, ms=4, lw=1)
    for ax in [a, b]:
        ax.set_ylim(-.6, 5.6)
        ax.set_yticks(range(6), [r["label"] for r in rows][::-1])
    a.set(xlim=(0, 22), xlabel="Queries changing plan (%)")
    a.set_xticks([0, 10, 20])
    b.axvline(0, color="#8A8F98", ls=":", lw=.8)
    b.set(xlim=(-5.2, 1.2), xlabel="Point / interval runtime − 1 (%)")
    b.set_xticks([-4, -2, 0])
    save(fa, "ch4_plan_verified_a")
    save(fb, "ch4_plan_verified_b")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"output -> {OUT}")
    for fn in [fig02_necessity, fig03_failure_modes, fig04_width_fan, fig05_feedback_wide,
               fig06_cross_dataset, fig07_data_level_merged, fig08_detection,
               fig09_continuous_repair, fig10_real_cost, fig11_plan_verified]:
        print(fn.__name__)
        fn()
    (HERE / "figure_manifest_en.json").write_text(
        json.dumps(MANIFEST, indent=2, ensure_ascii=False), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
