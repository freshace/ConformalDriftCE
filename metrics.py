# -*- coding: utf-8 -*-
"""metrics —— 论文主指标 q*(alpha) 的定义与实现（唯一口径）。

背景（为什么不用覆盖率当主指标）
--------------------------------
区间写作 [yhat/e^q, yhat*e^q]，非一致性分数 s = |log y - log yhat|。
于是：
        coverage(q) = P(s <= q) = F_s(q)          —— 覆盖率是"分数分布的 CDF 在 q 处取值"
                                                     它是【输出】，不是一条独立的轴。
        q          = 报告宽度                       —— 可自由选择（能被"买"）
        F_s        = 残差分布                       —— 只由【模型 x 当前数据】决定，不可伪造

因此覆盖率和宽度【不是正交的】：给定模型，它们是同一条曲线上的两个坐标
（古典名称：calibration-sharpness tradeoff；共形里叫 validity-efficiency）。
"用宽度买覆盖率"在静态快照里看不出来，只有时间维能暴露。

本模块定义唯一正确的主指标
--------------------------
    q*(alpha) := 用【当前窗口自身】的分数、按 split conformal 的同一口径
                 校准出来的 q
               = 分数数组中第 ceil((n+1)(1-alpha)) 小的值
               = "在该窗口达到目标覆盖 1-alpha 所需的【最小】报告宽度"

    效率比 rho := q_reported / q*(alpha)
        rho < 1  -> 报告宽度不足，覆盖会跌破目标（静默失效）
        rho ~ 1  -> 诚实（RECAL-MON 按构造成立）
        rho > 1  -> 过度保守，用宽度换覆盖（覆盖率会"看起来"很好）

要点：
  * q* 不是新的估计量 —— 它就是 calibrate_q()，只是【用在正确的数据上】。
    报告 q 用的是旧校准集 -> 过期；用当前窗口 -> 诚实。
  * q* 不能靠调参改小，只能靠模型真的变好（或分布回稳）。
  * 覆盖率可以被买，q* 买不到。所以主指标是 q*（及其效率比 rho），覆盖率降为辅助读数。
  * 【前提】q* 由监控样本估计，故要求监控样本与被评分流量可交换；
    违反时 q* 被系统性低估 -> 又窄又错（见 experiment_recal_sideeffects 的 BIAS 臂）。
"""
import numpy as np

__all__ = ["q_star", "coverage_at_q", "efficiency_ratio", "frontier", "self_check"]


def q_star(scores, alpha):
    """q*(alpha)：与 split conformal 同口径的最小诚实宽度（对数尺度）。

    取第 ceil((n+1)(1-alpha)) 小的分数（与 conformal.calibrate_q 完全一致）。
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return float("nan")
    k = min(int(np.ceil((n + 1) * (1.0 - alpha))) - 1, n - 1)
    return float(s[max(k, 0)])


def coverage_at_q(scores, q):
    """在报告宽度 q 下的实际覆盖率。"""
    s = np.asarray(scores, dtype=float)
    return float(np.mean(s <= q)) if s.size else float("nan")


def efficiency_ratio(q_reported, q_star_value):
    """效率比 rho = 报告宽度 / 最小诚实宽度。"""
    return float(q_reported / q_star_value) if q_star_value and q_star_value > 0 else float("nan")


def frontier(scores, alphas):
    """覆盖-宽度前沿：对每个目标 alpha 给出 (目标覆盖, 最小宽度 q*, 实测覆盖)。"""
    out = []
    for a in alphas:
        q = q_star(scores, a)
        out.append((1.0 - a, q, coverage_at_q(scores, q)))
    return out


def self_check(n=20000, alpha=0.10, seed=0, verbose=True):
    """口径自检：若用 q* 报告，覆盖率应约等于 1-alpha；且 q* <= 任意达到同覆盖的 q。

    这是本模块的【正确性保证】——重报任何实验前先跑它。
    """
    rng = np.random.RandomState(seed)
    # 右偏分布（与 CE 残差同形：重尾）
    scores = np.abs(rng.standard_t(df=3, size=n)) * 0.5 + 0.2
    half = n // 2
    cal, ev = scores[:half], scores[half:]
    qs = q_star(cal, alpha)                 # 用校准集定 q*（拆一半，模拟真实流程）
    cov = coverage_at_q(ev, qs)
    # 最小性：任何更小的 q 覆盖都更低
    qs_small = q_star(cal, alpha * 1.5)
    cov_small = coverage_at_q(ev, qs_small)
    ok_cov = abs(cov - (1 - alpha)) < 0.02
    ok_min = qs_small <= qs and cov_small <= cov + 1e-12
    if verbose:
        print(f"[metrics.self_check] n={n} alpha={alpha}")
        print(f"  q*={qs:.4f} -> 实测覆盖 {cov:.4f} (目标 {1-alpha:.2f})   {'OK' if ok_cov else 'FAIL'}")
        print(f"  更小宽度 {qs_small:.4f} -> 覆盖 {cov_small:.4f}      {'OK(更窄覆盖更低)' if ok_min else 'FAIL'}")
    return bool(ok_cov and ok_min)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for a in (0.05, 0.10, 0.20):
        assert self_check(alpha=a), f"self_check failed at alpha={a}"
    print("\n全部通过：q* 口径可用。")
