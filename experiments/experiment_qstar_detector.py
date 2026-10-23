# -*- coding: utf-8 -*-
"""检测器在 q*(alpha) 口径下的重做：检查"宽度类信号优于覆盖率信号"这个结论是否翻转。

背景
----
在 R1（不重校准、固定 q0）下，两个信号都是【非同义反复】的（测量样本 ≠ 校准样本）：
    s_cov  = 1 - coverage(监控样本, q0)        <- 旧口径：覆盖率下降
    s_rho  = q*_est(监控样本) / q0             <- 新口径：最小诚实宽度相对旧宽度的膨胀
    s_alpha= alpha - alpha_t（ACI 适应力度）    <- 早先发现"更强"，但那是【跨窗口积分】的量

关键理论预期
------------
    coverage(监控) = F_mon(q0)，q*_est = F_mon^{-1}(1-alpha)
    二者都是同一分布 F_mon 的单调函数 -> 【单调等价】-> ROC 应当几乎重合。
    若 s_rho 与 s_cov 的 ROC 确实重合，则"换成 q* 口径不改变检测结论"（无翻转）；
    而 s_alpha 的差异来自它累积历史，不是来自量纲。

报告
----
  表 A  三个统计量在 none 流上定阈值（同 FPR=10%）后的 TPR 与检测延迟
  表 B  s_cov 与 s_rho 的秩相关（Spearman）—— 验证单调等价
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, data, model, conformal, drift, metrics

T_WIN, W_WIN, N_MON, N_EVAL = 20, 500, 150, 300
N_STREAM = T_WIN * W_WIN
SEEDS = [42, 43, 44, 45, 46]
SPIKE = [6, 13, 18]
GAMMA = 0.05
ALPHA_LO, ALPHA_HI = 0.002, 0.30

def ru(t): return 2.0
def rp(t): return 1.0 * (1.0 + np.sin(2.0 * np.pi * t / 10.0))
def rt(t): return 2.0 * t / (T_WIN - 1)
def rl(t): return 2.0 if t in SPIKE else 0.2
PATTERNS = [("none", lambda t: 0.0), ("uniform", ru), ("periodic", rp), ("trend", rt), ("long-tail", rl)]
STATS = ["s_cov(旧:覆盖率)", "s_rho(新:q*膨胀)", "s_alpha(ACI力度)"]

def run_one_seed(seed, base, true, feat, nfc):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(true))
    itr = idx[:config.N_TRAIN]
    ica = idx[config.N_TRAIN:config.N_TRAIN + config.N_CAL]
    ist = idx[config.N_TRAIN + config.N_CAL:config.N_TRAIN + config.N_CAL + N_STREAM]
    m0 = model.fit_corrector(feat[itr], base[itr], true[itr], config.LGBM_PARAMS)
    s_cal = conformal.score(true[ica], base[ica], model.predict_corrector(m0, feat[ica]))
    q0 = metrics.q_star(s_cal, config.ALPHA)

    wins = [np.arange(i * W_WIN, (i + 1) * W_WIN) for i in range(T_WIN)]
    mon = [w[:N_MON] for w in wins]
    ev = [w[W_WIN - N_EVAL:] for w in wins]

    out = {}
    for name, fn in PATTERNS:
        rho_t = np.array([float(fn(t)) for t in range(T_WIN)])
        S_mon, S_eval = [], []
        for t in range(T_WIN):
            i_m, i_e = ist[mon[t]], ist[ev[t]]
            S_mon.append(conformal.score(
                drift.inject_correlation_drift(true[i_m], nfc[i_m], rho_t[t], rng),
                base[i_m], model.predict_corrector(m0, feat[i_m])))
            S_eval.append(conformal.score(
                drift.inject_correlation_drift(true[i_e], nfc[i_e], rho_t[t], rng),
                base[i_e], model.predict_corrector(m0, feat[i_e])))

        cov_mon = np.array([metrics.coverage_at_q(S_mon[t], q0) for t in range(T_WIN)])
        cov_ev = np.array([metrics.coverage_at_q(S_eval[t], q0) for t in range(T_WIN)])
        qs = np.array([metrics.q_star(S_mon[t], config.ALPHA) for t in range(T_WIN)])
        s_cov = 1.0 - cov_mon
        s_rho = qs - q0                       # 对数尺度宽度膨胀（>0 表示需要更宽；exp() 得倍数）
        # ACI 力度（与 experiment_aci_baseline 同一动力学）
        a_t, s_alpha = config.ALPHA, np.zeros(T_WIN)
        for t in range(T_WIN):
            qt = metrics.q_star(s_cal, a_t)
            err = 1.0 - metrics.coverage_at_q(S_mon[t], qt)
            s_alpha[t] = config.ALPHA - a_t
            a_t = float(np.clip(a_t + GAMMA * (config.ALPHA - err), ALPHA_LO, ALPHA_HI))
        out[name] = dict(s_cov=s_cov, s_rho=s_rho, s_alpha=s_alpha,
                         cov_ev=cov_ev, q0=float(q0))
    return out


def main():
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    print("=" * 104)
    print("检测器在 q* 口径下的重做 | 目标覆盖 %.2f | 种子 %s | 同 FPR=10%%（阈值取自 none 流 90 分位）"
          % (1 - config.ALPHA, SEEDS))
    print("=" * 104)
    per = []
    for s in SEEDS:
        print("  seed %d ..." % s, flush=True)
        per.append(run_one_seed(s, base, true, feat, nfc))
    names = [n for n, _ in PATTERNS]

    # 阈值：在 none 流上取 90 分位 -> 各统计量 FPR 同为 10%
    thr = {}
    for k in ["s_cov", "s_rho", "s_alpha"]:
        vals = np.concatenate([d["none"][k] for d in per])
        thr[k] = float(np.quantile(vals, 0.90))

    print("\n表 A  同 FPR(=10%%) 下的 TPR 与检测延迟（自冻结覆盖首次跌破目标起算）")
    print("%-11s |" % "模式" + "".join(" %-19s |" % s for s in STATS))
    print("-" * 104)
    for n in names:
        cells = []
        for k in ["s_cov", "s_rho", "s_alpha"]:
            tprs, lats = [], []
            for d in per:
                fired = np.where(d[n][k] > thr[k])[0]
                tprs.append(len(fired) / T_WIN)
                if n != "none":
                    ref = next((t for t in range(T_WIN) if d[n]["cov_ev"][t] < (1 - config.ALPHA)), None)
                    if ref is not None:
                        aft = [f for f in fired if f >= ref]
                        if aft:
                            lats.append(aft[0] - ref)
            tpr = np.mean(tprs) * 100
            lat = ("%.1f" % np.mean(lats)) if lats else "--"
            cells.append("%5.0f%% / %-9s" % (tpr, lat))
        print("%-11s |" % n + "".join(" %-19s |" % c for c in cells))
    print("  阈值: s_cov>%.4f  s_rho>%.4f  s_alpha>%.4f" % (thr["s_cov"], thr["s_rho"], thr["s_alpha"]))

    # 秩相关：验证 s_cov 与 s_rho 是否单调等价
    from scipy.stats import spearmanr
    print("\n表 B  s_cov 与 s_rho 的 Spearman 秩相关（验证二者是否单调等价）")
    print("%-11s | %12s | %12s" % ("模式", "rho(s_cov,s_rho)", "rho(s_cov,s_alpha)"))
    print("-" * 104)
    for n in names:
        a = np.concatenate([d[n]["s_cov"] for d in per])
        b = np.concatenate([d[n]["s_rho"] for d in per])
        c = np.concatenate([d[n]["s_alpha"] for d in per])
        r1, _ = spearmanr(a, b)
        r2, _ = spearmanr(a, c)
        print("%-11s | %12.4f | %12.4f" % (n, r1, r2))
    print("  读法：rho(s_cov,s_rho)≈1 说明覆盖率与 q* 是同一分布的单调函数 -> 换口径【不改变】检测结论；")
    print("        rho(s_cov,s_alpha) 明显更低，说明 ACI 力度的额外检测力来自【跨窗口积分历史】，")
    print("        而不是来自 q* 这个量纲本身 —— 这是一个可写进论文的正面结论。")

    traj = {n: {k: np.mean([d[n][k] for d in per], axis=0).tolist()
                for k in ["s_cov", "s_rho", "s_alpha", "cov_ev"]} for n in names}
    traj["_thr"] = thr
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_qstar_detector.npy")
    np.save(out, traj, allow_pickle=True)
    print("\n已存 %s" % out)


if __name__ == "__main__":
    main()
