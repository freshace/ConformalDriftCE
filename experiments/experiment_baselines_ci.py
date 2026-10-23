# -*- coding: utf-8 -*-
"""§4.8 Oracle 对比的重复切分版（P4 收尾）：给「500 标签 ≈ 2{,}000 标签」这条结论补 95% CI。

【为什么需要】`experiment_baselines.py` 只有一次 `RandomState(config.SEED)`，
论文 `tab:oracle` 的两行（相关性 62.2/91.0/89.2、工作负载 45.8/90.8/90.6）都是**单次读数**，
而"我们 vs Oracle 差 1.8pp 是否落在噪声内"正是这张表的关键论断。

【复现门】沿用原脚本相同的 rng 消耗顺序（先 permutation(n)，再三次 choice），
故第 1 次重复(seed=42)应逐位复现单次读数。

【产出】results_repro/experiment_baselines_ci.txt + baselines_ci.npz
"""
import os, sys, importlib.util, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import config

_spec = importlib.util.spec_from_file_location("bl", os.path.join(HERE, "experiment_baselines.py"))
bl = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bl)

ROOT = os.path.dirname(HERE)
R = int(os.environ.get("BL_R", "50"))


def ci95(a):
    a = np.asarray(a, dtype=np.float64); a = a[np.isfinite(a)]
    if len(a) < 2:
        return float(a.mean()), float("nan")
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def main():
    out = []
    def log(s):
        out.append(s); print(s, flush=True)

    base, true, feat, nfc, ntab = bl.load(config.DATA_PATH)
    n = len(true)
    log(f"加载 {n} 条查询；R={R} 次独立重复划分")
    log("口径与 experiment_baselines.py 同源（复用其 load/run）⇒ 第 1 次重复(seed=42)应复现单次读数")

    acc = {}
    for tag in ("corr", "work"):
        acc[tag] = {"frozen": [], "ours": [], "oracle": []}

    for r in range(R):
        rng = np.random.RandomState(config.SEED + r)      # 与原脚本一致
        idx = rng.permutation(n)
        tr = idx[:config.N_TRAIN]
        cal = idx[config.N_TRAIN:config.N_TRAIN + config.N_CAL]
        te = idx[config.N_TRAIN + config.N_CAL:config.N_TRAIN + config.N_CAL + config.N_TEST]
        # 1) 相关性漂移
        drift_t = true[te] * np.where(nfc[te] >= 2, np.exp(2.0), 1.0)
        f_, o_, or_ = bl.run("corr", feat[tr], base[tr], true[tr], feat[cal], base[cal], true[cal],
                             feat[te], base[te], drift_t)
        acc["corr"]["frozen"].append(f_); acc["corr"]["ours"].append(o_); acc["corr"]["oracle"].append(or_)
        # 2) 工作负载漂移（同原脚本的三次 choice 顺序）
        tr_s = rng.choice(np.where(ntab <= 3)[0], size=1500, replace=False)
        cal_s = rng.choice(np.setdiff1d(np.where(ntab <= 3)[0], tr_s), size=300, replace=False)
        te_c = rng.choice(np.where(ntab >= 5)[0], size=4000, replace=False)
        f_, o_, or_ = bl.run("work", feat[tr_s], base[tr_s], true[tr_s], feat[cal_s], base[cal_s], true[cal_s],
                             feat[te_c], base[te_c], true[te_c])
        acc["work"]["frozen"].append(f_); acc["work"]["ours"].append(o_); acc["work"]["oracle"].append(or_)
        if (r + 1) % 10 == 0 or r == R - 1:
            log(f"  ... replicate {r+1}/{R}")

    log("")
    log(f"== §4.8 Oracle 对比（R={R} 次独立划分；mean ± 95%CI，方括号为单次读数范围）==")
    log(f"{'漂移':<12} {'冻结':>22} {'我们(500)':>22} {'Oracle(2{,}000)':>22} {'我们−Oracle':>18}")
    log("-" * 100)
    for tag, lab in (("corr", "相关性(7.39x)"), ("work", "工作负载")):
        d = acc[tag]
        def c(k):
            a = np.array(d[k]) * 100; m, h = ci95(a)
            return f"{m:5.1f}±{h:4.1f} [{a.min():.1f},{a.max():.1f}]"
        diff = (np.array(d["ours"]) - np.array(d["oracle"])) * 100
        md, hd = ci95(diff)
        log(f"{lab:<12} {c('frozen'):>22} {c('ours'):>22} {c('oracle'):>22} {md:+6.2f}±{hd:4.2f}pp")
    log("")
    log("== 关键论断核查：「500 标签是否与 2{,}000 标签持平」==")
    for tag, lab in (("corr", "相关性"), ("work", "工作负载")):
        diff = (np.array(acc[tag]["ours"]) - np.array(acc[tag]["oracle"])) * 100
        md, hd = ci95(diff)
        # 95%CI 是否覆盖 0 ⇒ 是否有统计显著差异
        cover0 = (md - hd) <= 0 <= (md + hd)
        verdict = "持平（CI 覆盖 0，无显著差异）" if cover0 else "有显著差异（CI 不含 0）"
        log(f"  {lab:<8} 我们−Oracle = {md:+6.2f}±{hd:4.2f}pp  ⇒ {verdict}")
    log("  (单次读数下论文写的是 +1.8pp / +0.2pp,并据此说'落在噪声内';这里给出直接检验。)")

    log("")
    log("== 复现门：第 1 次重复(seed=42) 应与 tab:oracle 的单次读数一致 ==")
    for tag, lab, tgt in (("corr", "相关性", "62.2/91.0/89.2"), ("work", "工作负载", "45.8/90.8/90.6")):
        d = acc[tag]
        log(f"  {lab:<8} 冻结 {d['frozen'][0]*100:5.1f}% | 我们 {d['ours'][0]*100:5.1f}% | "
            f"Oracle {d['oracle'][0]*100:5.1f}%   (原: {tgt})")

    OUT = os.path.join(ROOT, "results_repro")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "experiment_baselines_ci.txt"), "w", encoding="utf-8") as f:
        f.write("exit=0\n" + "\n".join(out) + "\n")
    np.savez(os.path.join(OUT, "baselines_ci.npz"), R=R,
             **{f"{t}__{k}": np.array(v) for t, d in acc.items() for k, v in d.items()})
    print("\nsaved results_repro/experiment_baselines_ci.txt", flush=True)


if __name__ == "__main__":
    main()
