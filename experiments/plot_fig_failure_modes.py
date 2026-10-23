# -*- coding: utf-8 -*-
"""正文 §4.3 图 fig:failure-modes：三条使覆盖率失去信息量的机制。

三格分别对应 RQ2 的三条机制：
  (a) 证据复用 —— 自测覆盖率是恒等式，独立审计才反映真实质量
  (b) 区间宽度 —— 覆盖对齐时宽区间吸收漂移，使失效不可见
  (c) 反馈代表性 —— β 升高时监控自报覆盖纹丝不动，真实覆盖已崩

★ 本脚本只读既有产物，不训练、不重算任何实验：
     results_repro/verify_audit.npz
     results_repro/experiment_residual_sensitivity.txt
     results_repro/experiment_qstar_bias_dose.txt
  每格绘制的数值都先与产物断言比对，防止"图对了但和正文数字不一致"。
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vldb_style as vs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results_repro")

# Okabe-Ito，与论文其他图一致
C_WIDE = vs.C_ALT     # 宽区间 / 失效
C_NARROW = vs.C_MAIN  # 窄区间 / 正常
C_SELF = vs.C_LIGHT   # 自测 / 监控自报
C_AUDIT = vs.C_THIRD  # 独立审计 / 真实
C_REF = vs.C_REF

FAILURES = []


def need(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILURES.append(msg)


def approx(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def read_res(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        sys.exit("产物缺失: " + p)
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def table_row(txt, title, key="uniform"):
    """从 '表 X' 段落里取 key 行的 5 个单元格（去掉 x / % 后缀）。"""
    try:
        i = txt.index(title)
    except ValueError:
        sys.exit("产物里找不到段落: " + title)
    for ln in txt[i:].splitlines():
        if ln.strip().startswith(key):
            # 表格行以 '|' 结尾，split 后会多出一个空单元格，先剔掉
            cells = [c.strip() for c in ln.split("|")[1:] if c.strip()]
            return [float(re.sub(r"[x%]+$", "", c)) for c in cells]
    sys.exit("段落 %s 里找不到行 %s" % (title, key))


def main():
    vs.apply()
    print("== (a) 证据复用：自测覆盖 vs 独立审计覆盖 ==")
    va = np.load(os.path.join(RES, "verify_audit.npz"))
    arms = ["frozen", "repair", "recal_only", "garbage"]
    self_cov = np.array([va["self_" + a].mean() for a in arms]) * 100
    audit_cov = np.array([va["audit_" + a].mean() for a in arms]) * 100
    width = np.array([va["width_" + a].mean() for a in arms])
    self_n = np.array([va["self_n_" + a].mean() for a in arms])
    need(int(va["R"]) == 50, "verify_audit R=50")
    # 自测覆盖是恒等式 ⌈(n+1)(1−α)⌉/n：冻结臂在 n=3000 上自测、其余三臂在 n=500 上自测，
    # 故前者恒 90.03%、后者恒 90.20%。两值都由 n 决定，与模型质量无关——这正是该格的论点。
    need(approx(self_n[0], 3000, 0.5) and all(approx(v, 500, 0.5) for v in self_n[1:]),
         "自测样本量 n = %s" % self_n.tolist())
    need(approx(self_cov[0], 90.033, 0.01), "自测(冻结,n=3000) = 90.033%%，实测 %.3f" % self_cov[0])
    need(approx(self_cov[1], 90.200, 0.01), "自测(修复) = 90.200%%，实测 %.3f" % self_cov[1])
    need(approx(self_cov[3], 90.200, 0.01), "自测(灾难) = 90.200%%，实测 %.3f" % self_cov[3])
    need(approx(audit_cov[0], 66.087, 0.02), "审计(冻结) ≈ 66.1%%，实测 %.3f" % audit_cov[0])
    need(approx(audit_cov[3], 90.098, 0.02), "审计(灾难) ≈ 90.1%%，实测 %.3f" % audit_cov[3])
    need(approx(width[3], 1254.36, 0.5), "灾难臂宽度 ≈ 1254×，实测 %.2f" % width[3])
    need(approx(va["medqe_garbage"].mean() / va["medqe_repair"].mean(), 97.74 / 2.88, 0.6),
         "灾难臂中位 q-error %.2f vs 修复 %.2f" % (va["medqe_garbage"].mean(), va["medqe_repair"].mean()))

    print("== (b) 区间宽度：覆盖对齐下的宽度与漂移信号 ==")
    revision_res = os.path.join(ROOT, "results_revision_20260930", "figures_sensitivity")
    with open(os.path.join(revision_res, "experiment_residual_sensitivity.txt"), encoding="utf-8") as handle:
        rs = handle.read()
    raw_sensitivity = np.load(os.path.join(revision_res, "residual_sensitivity_raw.npz"))
    head = rs[rs.index("(1) Coverage-matched"):rs.index("(2) Drift dose-response")]
    m_res = re.search(r"Residual\+Conformal \(alpha=0\.10\): coverage=([\d.]+)±([\d.]+)%\s+width=([\d.]+)±([\d.]+)x", head)
    m_base = re.search(r"Base\+Conformal \(alpha matched\):\s+coverage=([\d.]+)±([\d.]+)%\s+width=([\d.]+)±([\d.]+)x", head)
    m_ratio = re.search(r"coverage-matched width ratio Base/Residual = ([\d.]+)±([\d.]+)x", head)
    if not (m_res and m_base and m_ratio):
        sys.exit("experiment_residual_sensitivity.txt 的覆盖对齐段落解析失败")
    cov_n, w_n = float(m_res.group(1)), float(m_res.group(3))
    cov_w, w_w = float(m_base.group(1)), float(m_base.group(3))
    ratio = float(m_ratio.group(1))
    need(approx(w_n, 7.53, 0.01), "窄区间宽 7.53×，实测 %.2f" % w_n)
    need(approx(w_w, raw_sensitivity["base_matched_width"].mean(), 0.01), "独立验证匹配后的宽度 %.2f×" % w_w)
    need(approx(ratio, (raw_sensitivity["base_matched_width"] / raw_sensitivity["residual_width"]).mean(), 0.05),
         "逐划分宽度比 %.1f×" % ratio)
    need(np.all(np.abs(raw_sensitivity["residual_validation_coverage"] -
                       raw_sensitivity["base_matched_validation_coverage"]) < 0.01),
         "仅验证集匹配；最终测试覆盖 %.2f%% vs %.2f%%" % (cov_n, cov_w))

    dose = rs[rs.index("(2) Drift dose-response"):]
    row739 = [ln for ln in dose.splitlines() if ln.strip().startswith("7.39x")]
    if not row739:
        sys.exit("experiment_residual_sensitivity.txt 里找不到 7.39x 剂量行")
    nums = re.findall(r"([\d.]+)±([\d.]+)pp", row739[0])
    sig_base, sig_res = float(nums[0][0]), float(nums[1][0])
    need(approx(sig_base, (0.9 - raw_sensitivity["base_cov"][:, -1]).mean() * 100, 0.05),
         "宽区间漂移信号 %.1fpp" % sig_base)
    need(approx(sig_res, (0.9 - raw_sensitivity["res_cov"][:, -1]).mean() * 100, 0.05),
         "窄区间漂移信号 %.1fpp" % sig_res)

    print("== (c) 反馈代表性：β 剂量-响应 ==")
    txt = read_res("experiment_qstar_bias_dose.txt")
    betas = [0.0, 0.25, 0.5, 0.75, 1.0]
    true_cov = table_row(txt, "表 B")
    mon_cov = table_row(txt, "表 C")
    alarm = table_row(txt, "表 D")
    under = table_row(txt, "表 A")
    need(len(true_cov) == len(betas), "β 网格长度 = %d" % len(true_cov))
    need(approx(true_cov[0], 89.6, 0.05), "β=0 真实覆盖 89.6%%，实测 %.1f" % true_cov[0])
    need(approx(true_cov[-1], 43.8, 0.05), "β=1 真实覆盖 43.8%%，实测 %.1f" % true_cov[-1])
    need(all(approx(v, 90.7, 0.05) for v in mon_cov), "监控自报覆盖恒 90.7%%，实测 %s" % mon_cov)
    need(all(approx(v, 0.0, 0.05) for v in alarm), "报警率恒 0%%，实测 %s" % alarm)
    need(approx(under[0], 1.01, 0.005) and approx(under[-1], 7.94, 0.005),
         "不确定性低估 1.01×→7.94×，实测 %.2f→%.2f" % (under[0], under[-1]))

    if FAILURES:
        sys.exit("\n%d 项数值断言失败，拒绝出图：" % len(FAILURES) + "\n  " + "\n  ".join(FAILURES))

    # ---------------- 出图：双栏宽 = 显示宽，字号即论文实际磅值 ----------------
    fig, axes = vs.subplots(1, 3, width=vs.DCOL, height=2.05)

    ax = axes[0]
    x = np.arange(len(arms))
    ax.bar(x, audit_cov, 0.56, color=C_AUDIT, label="独立审计覆盖")
    ax.plot(x, self_cov, "o", ms=3.4, mfc="white", mew=0.9, color=C_SELF,
            label="自测覆盖（恒等式）", zorder=3)
    ax.set_xticks(x)
    # 四个状态名横排会连成一片，折成两行才留得出字距
    ax.set_xticklabels(["冻结", "修复", "只重\n校准", "灾难性\n错误"], fontsize=6.4)
    # 顶部留白给图例：自测点贴在 90% 附近，"恒等式"三字的空间只能向上要
    ax.set_ylim(60, 116)
    ax.set_ylabel("覆盖率 (%)")
    ax.set_yticks([60, 70, 80, 90, 100])
    for xi, v in zip(x, audit_cov):
        ax.text(xi, v - 5 if v > 85 else v + 1.2, "%.1f" % v, ha="center",
                fontsize=6.4, color="white" if v > 85 else C_AUDIT)
    # 数字直接取自已断言的数组，避免图示与产物各写一份
    ax.text(3, 98.0, "宽 %.0f×\nq50 %.1f" % (width[3], va["medqe_garbage"].mean()),
            ha="center", fontsize=6.0, color=C_WIDE)
    ax.set_title("(a) 证据复用：自测是恒等式", fontsize=7.5, pad=3)
    ax.legend(fontsize=6.0, loc="upper left", ncol=2, handlelength=1.3,
              columnspacing=0.8, borderaxespad=0.2)

    ax = axes[1]
    vals = [w_n, w_w]
    width_ci = [vs.ci95(raw_sensitivity[key])[1] for key in ("residual_width", "base_matched_width")]
    bars = ax.bar([0, 1], vals, 0.5, color=[C_NARROW, C_WIDE], yerr=width_ci,
                  capsize=2, error_kw=dict(lw=0.7, ecolor=vs.C_INK))
    ax.set_yscale("log")
    ax.set_yticks([10, 100, 1000])
    ax.set_yticklabels(["10", "100", "1000"])
    ax.minorticks_off()
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["窄区间\n(残差修正)", "宽区间\n(裸底模)"], fontsize=6.4)
    ax.set_ylabel("区间宽度（倍，对数轴）")
    ax.set_ylim(3, 5000)
    ax.set_xlim(-0.55, 1.55)
    for b, v, h in zip(bars, vals, width_ci):
        ax.text(b.get_x() + b.get_width() / 2, (v + h) * 1.25, "%.1f×" % v, ha="center", fontsize=7)
    # 三条标注按高度分层，各自落在完全空白的区域：
    #   400  比值箭头 —— 抬到宽柱顶 222× 之上，箭身不再压在色块里
    #   900  比值文字
    #   3000 覆盖对齐
    # 漂移信号放在窄柱上方的空档（宽柱标注那个高度已有文字，不能再叠一行）。
    ax.text(0.5, 900, "宽度比 %.1f×" % ratio, ha="center", fontsize=6.4, color=C_REF)
    ax.text(0.5, 3000, "验证匹配；测试 %.1f%% / %.1f%%" % (cov_n, cov_w), ha="center",
            fontsize=6.4, color=C_REF)
    ax.text(0.0, 40, "漂移信号\n%.1fpp vs %.1fpp" % (sig_res, sig_base), ha="center",
            fontsize=6.2, color=C_REF)
    ax.set_title("(b) 宽区间吸收偏移", fontsize=7.5, pad=3)

    ax = axes[2]
    ax.plot(betas, true_cov, "-o", color=C_AUDIT, label="真实覆盖（部署流量）")
    ax.plot(betas, mon_cov, "--s", color=C_SELF, label="监控自报覆盖")
    vs.refline(ax, 90)
    ax.set_xlabel("β：监控样本中\n未漂移查询的比例")
    ax.set_ylabel("覆盖率 (%)")
    ax.set_ylim(35, 103)
    ax.set_xticks(betas)
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.text(0.5, 96.5, "报警率恒 0%", ha="center", fontsize=6.4, color=C_WIDE)
    ax.annotate("", xy=(1.0, 43.8), xytext=(1.0, 90.7),
                arrowprops=dict(arrowstyle="<->", color=C_REF, lw=0.8))
    ax.text(0.18, 49, "不确定性低估 %.2f×" % under[-1], fontsize=6.4, color=C_REF)
    ax.set_title("(c) 监控样本不具代表性", fontsize=7.5, pad=3)
    ax.legend(fontsize=6.0, loc="lower left", borderaxespad=0.3)

    fig.subplots_adjust(wspace=0.34)
    outdir = os.environ.get("DRIFT_FIG_OUT")
    if outdir and os.environ.get("DRIFT_FIG_QA_PATH"):
        sys.path.insert(0, os.environ["DRIFT_FIG_QA_PATH"])
        from audit_panel_alignment import require_matplotlib_panel_alignment
        os.makedirs(outdir, exist_ok=True)
        require_matplotlib_panel_alignment(fig, json_out=os.path.join(outdir, "fig_failure_modes.alignment.json"))
    vs.save(fig, "fig_failure_modes", outdir=outdir)
    print("全部数值断言通过。")


if __name__ == "__main__":
    main()
