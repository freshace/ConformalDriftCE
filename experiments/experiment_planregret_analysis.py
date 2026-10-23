#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§4.11 事后分析：从 F2 的 npz 直接算，不重跑实验。

做三件事：
 1. **跨实验复现（G3）**：把 D5 的口径——`real[default] / min{default, hash, merge, nestloop}`
    ——在【与 F2 完全相同的查询】上重算，从而把 "F2 的 1.03 vs D5 的 1.14" 的差异
    归因到「oracle 集合不同 / 抽样不同」哪一个。
    （F2 的规则集不含 default，故其 oracle 是 3 形状；D5 的 oracle 含 default ⇒ 只可能更低
     ⇒ D5 的 regret 只会更高。这里把两种口径都算出来对照。）
 2. 三规则（点估计 / 稳健 minimax / 区间均值）与参照 default 的实测 regret 汇总 + 配对检验。
 3. 机制量：代价模型排序对基数信念的不变率，以及"排序会变"的子集里稳健是否真的改善。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results_repro")
B = 2000
RNG = np.random.RandomState(20260917)


def boot_ci(x, stat, b=B, rng=RNG):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    n = len(x)
    v = np.array([stat(x[rng.randint(0, n, n)]) for _ in range(b)])
    return float(stat(x)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired_ci(a, b_, b=B, rng=RNG):
    d = np.asarray(a, float) - np.asarray(b_, float)
    return boot_ci(d, np.median, b, rng)


def boot_stat_diff(a, b_, stat, b=B, rng=RNG):
    """配对 bootstrap：统计量之差的 95% 区间。"""
    a = np.asarray(a, float)
    b_ = np.asarray(b_, float)
    n = len(a)
    v = np.empty(b)
    for i in range(b):
        idx = rng.randint(0, n, n)
        v[i] = stat(a[idx]) - stat(b_[idx])
    return float(stat(a) - stat(b_)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def binom_test(k, n, p=0.5):
    """精确二项检验（双侧），不依赖 scipy。"""
    from math import comb
    if n == 0:
        return float("nan")
    pmf = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = pmf[k]
    return float(sum(x for x in pmf if x <= obs + 1e-12))


def report(path, tag):
    if not os.path.exists(path):
        print(f"!! 缺 {path}")
        return None
    d = np.load(path, allow_pickle=True)
    R = d["R"]                                  # point, robust, mean, ref
    cand = [str(s) for s in d["cand"]]          # hash, merge, nestloop, default
    real = d["real"]                            # (n, 4) 与 cand 同序
    j_def = cand.index("default")
    shapes = [i for i, c in enumerate(cand) if c != "default"]

    pt, rb, mn, rf = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
    # D5 口径：oracle 含 default
    oracle_d5 = np.minimum(real[:, j_def], real[:, shapes].min(axis=1))
    R_d5 = real[:, j_def] / oracle_d5
    # F2 口径：oracle 只含 3 个固定形状（这是规则的可选集合）
    R_f2 = rf

    print("=" * 96)
    print(f"§4.11 事后分析  [{tag}]   drift={float(d['drift'])}  q={float(d['q'])}  "
          f"nq={int(d['nq'])}  seed={int(d['seed'])}  reps={int(d['reps'])}")
    print("=" * 96)
    print(f"  有效查询 n = {len(R)}")
    print()
    print("  【G3 跨实验复现】default 计划相对后见之明 oracle 的实测 regret")
    for name, v in (("D5 口径(oracle 含 default, 4 方法)", R_d5),
                    ("F2 口径(oracle 只含 3 固定形状)", R_f2)):
        m, lo, hi = boot_ci(v, np.median)
        p90, plo, phi = boot_ci(v, lambda x: np.percentile(x, 90))
        print(f"    {name:34s} median={m:.3f} [{lo:.3f},{hi:.3f}]   "
              f"p90={p90:.3f} [{plo:.3f},{phi:.3f}]   max={v.max():.2f}")
    print(f"    D5 已发表值(宽-2 分层)            median=1.137 [1.053,1.205]   "
          f"（D5 全宽整体 median=1.150 [1.104,1.184]）")
    print()
    print("  【三规则 + 参照】实测 regret")
    for name, v in (("点估计规则 (belief=D·t)", pt), ("稳健规则 (区间 minimax 后悔)", rb),
                    ("区间均值规则", mn), ("参照: 不强制 default", rf)):
        m, lo, hi = boot_ci(v, np.median)
        p90, plo, phi = boot_ci(v, lambda x: np.percentile(x, 90))
        print(f"    {name:30s} median={m:.3f} [{lo:.3f},{hi:.3f}]   "
              f"p90={p90:.3f} [{plo:.3f},{phi:.3f}]   max={v.max():.2f}")
    print()
    print("  【稳健 vs 点估计】配对")
    dm, dlo, dhi = paired_ci(pt, rb)
    print(f"    median(点估计-稳健) = {dm:+.4f}  95% [{dlo:+.4f},{dhi:+.4f}]"
          f"   {'含 0 ⇒ 无稳健改善证据' if dlo <= 0 <= dhi else '不含 0'}")
    print(f"    严格更好 {np.mean(rb < pt-1e-9)*100:.1f}%  相等 "
          f"{np.mean(np.abs(rb-pt) <= 1e-9)*100:.1f}%  更差 {np.mean(rb > pt+1e-9)*100:.1f}%")
    p90d, p90l, p90h = boot_stat_diff(pt, rb, lambda x: np.percentile(x, 90))
    print(f"    p90 之差 (点估计-稳健) = {p90d:+.4f}  95% [{p90l:+.4f},{p90h:+.4f}]"
          f"   {'含 0' if p90l <= 0 <= p90h else '不含 0'}")
    for thr in (1.2, 1.5, 2.0):
        fd, fl, fh = boot_stat_diff(pt, rb, lambda x, t=thr: float(np.mean(x > t)))
        print(f"    P(regret>{thr}) 之差 = {fd*100:+.2f}pp  95% [{fl*100:+.2f},{fh*100:+.2f}]pp"
              f"   {'含 0' if fl <= 0 <= fh else '不含 0'}")
    for thr in (1.2, 1.5, 2.0):
        print(f"    regret>{thr}: 点估计 {np.mean(pt>thr)*100:.1f}%  稳健 {np.mean(rb>thr)*100:.1f}%"
              f"  参照 {np.mean(rf>thr)*100:.1f}%")
    print()
    if "rank_inv" in d.files:
        ri = d["rank_inv"]
        print("  【机制】代价模型对候选形状的排序是否随基数信念改变")
        print(f"    排序不变（全部情形 argmin 相同）: {ri.sum()}/{len(ri)} = {ri.mean()*100:.1f}%"
              f"  ⇒ 这些查询上 minimax 必与点估计同选")
        if (~ri).any():
            a, b_ = pt[~ri], rb[~ri]
            nbet = int(np.sum(b_ < a - 1e-9))
            nwor = int(np.sum(b_ > a + 1e-9))
            print(f"    排序会变的 {len(a)} 条里: 稳健严格更好 {nbet} 条、"
                  f"相等 {np.sum(np.abs(b_-a) <= 1e-9)} 条、更差 {nwor} 条")
            print(f"      该子集中位 regret: 点估计 {np.median(a):.3f}  稳健 {np.median(b_):.3f}")
            ma, la, ha = boot_ci(a, np.median)
            mb2, lb2, hb2 = boot_ci(b_, np.median)
            print(f"      子集 median 95% CI: 点估计 [{la:.3f},{ha:.3f}]  "
                  f"稳健 [{lb2:.3f},{hb2:.3f}]  （两组独立重采样，重叠与否只能作粗看）")
            dp, dl, dh = paired_ci(a, b_)
            print(f"      子集配对差 median = {dp:+.4f}  95% [{dl:+.4f},{dh:+.4f}]")
            print(f"      符号检验（更好 vs 更差，{nbet}/{nbet+nwor}）双侧 p = "
                  f"{binom_test(nbet, nbet+nwor):.4f}")
            # 子集里的尾部
            for thr in (1.5, 2.0):
                print(f"      子集 regret>{thr}: 点估计 {np.mean(a>thr)*100:.1f}%  "
                      f"稳健 {np.mean(b_>thr)*100:.1f}%")
        else:
            print("    排序会变的查询: 0 条")
    print()
    print(f"  【规模】实测量级: default 实测中位 {np.median(real[:, j_def]):.1f} ms，"
          f"最优形状中位 {np.median(real[:, shapes].min(axis=1)):.1f} ms")
    return dict(n=len(R), R_d5=R_d5, R_f2=R_f2, pt=pt, rb=rb, mn=mn, rf=rf,
                rank_inv=d["rank_inv"] if "rank_inv" in d.files else None)


if __name__ == "__main__":
    out = []
    for tag in sys.argv[1:] or ["D7.39", "D1", "D16q"]:
        r = report(os.path.join(RES, "planregret_native_%s.npz" % tag), tag)
        if r:
            out.append((tag, r))
        print()
    print("=" * 96)
    print("小结（median 实测 regret）")
    print(f"  {'档':>8s} {'n':>4s} {'default(D5口径)':>16s} {'点估计':>8s} {'稳健':>8s} {'参照':>8s}")
    for tag, r in out:
        print(f"  {tag:>8s} {r['n']:>4d} {np.median(r['R_d5']):>16.3f} "
              f"{np.median(r['pt']):>8.3f} {np.median(r['rb']):>8.3f} {np.median(r['rf']):>8.3f}")

    # ---------- 静态代价模型误差 vs 漂移增量 ----------
    # D=1 时 point 规则用的信念恰好 = 真值 t ⇒ argmin_p est[1.0][p] 就是"若基数估计完全正确
    # PG 会选的形状"；其实测 regret 因此度量的是【静态代价模型误差（含残余估计误差）】。
    # D>1 时 point 规则用 belief=D·t ⇒ 度量【漂移后的总误差】。两者之差 = 漂移的增量贡献。
    by = {t: r for t, r in out}
    if "D1" in by and len(out) > 1:
        base = by["D1"]
        m_base = np.median(base["pt"])
        print()
        print("=" * 96)
        print("分解：静态代价模型误差 vs 漂移增量（边际口径，两次运行同种子同总体；")
        print("      有效子集可能略有差异，故按边际中位数比较，不做配对）")
        print("=" * 96)
        mb, lb, hb = boot_ci(base["pt"], np.median)
        print(f"  静态（belief=真值 t，D=1）        median={mb:.3f} [{lb:.3f},{hb:.3f}]"
              f"   ⇒ 即使基数完全正确，PG 选计划仍有这部分 regret（= 代价模型的方法排序误差）")
        for tag, r in out:
            if tag == "D1":
                continue
            m, lo, hi = boot_ci(r["pt"], np.median)
            print(f"  漂移后（D={tag[1:]}）                median={m:.3f} [{lo:.3f},{hi:.3f}]"
                  f"   增量 = {m - m_base:+.3f}")
        print("  ⇒ 若增量接近 0，则下游 regret 主要来自【静态代价模型误差】而非漂移；")
        print("     这也解释了为何只表达基数不确定度的共形区间改不动计划选择。")
