# -*- coding: utf-8 -*-
"""A5：前提破坏的【剂量-响应】——把 A4 的极端构造变成一条规律。

设计
----
监控样本按比例 beta 混合：
    beta 份取"廉价查询"（去重过滤列数 <=1；本实验漂移只作用于 nfc>=2，故这批【看不到漂移】）
    1-beta 份随机取（代表性样本）
beta = 0 -> 无偏（A2 的 RECAL 臂）；beta = 1 -> A4 的极端 BIAS 臂。

对每个 beta 报：
    低估倍数 = exp(q*_true - q*_est)     （>1 = 又窄又错）
    真实覆盖 / 监控看到覆盖
    报警率（单侧 95% 二项控制）

预期：随 beta 上升单调恶化；报警率在"真实覆盖已崩"时仍≈0（监控器盲区随 beta 出现）。
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
BETAS = [0.0, 0.25, 0.50, 0.75, 1.00]

def ru(t): return 2.0
def rp(t): return 1.0 * (1.0 + np.sin(2.0 * np.pi * t / 10.0))
def rt(t): return 2.0 * t / (T_WIN - 1)
def rl(t): return 2.0 if t in SPIKE else 0.2
PATTERNS = [("none", lambda t: 0.0), ("uniform", ru), ("periodic", rp), ("trend", rt), ("long-tail", rl)]


def monitor_idx(w, nfc_st, beta, rng):
    """beta 份廉价(nfc<=1) + (1-beta) 份随机，共 N_MON 条。"""
    n_cheap = int(round(beta * N_MON))
    if n_cheap == 0:
        return w[:N_MON]
    cheap_pool = np.array([i for i in w if nfc_st[i] <= 1], dtype=int)
    others = np.array([i for i in w if nfc_st[i] > 1], dtype=int)
    take_c = cheap_pool[:n_cheap] if len(cheap_pool) >= n_cheap else cheap_pool
    need = N_MON - len(take_c)
    take_o = rng.choice(others, size=min(need, len(others)), replace=False) if need > 0 and len(others) else np.array([], dtype=int)
    return np.sort(np.concatenate([take_c, take_o]))


def run_one_seed(seed, base, true, feat, nfc):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(true))
    itr = idx[:config.N_TRAIN]
    ist = idx[config.N_TRAIN + config.N_CAL:config.N_TRAIN + config.N_CAL + N_STREAM]
    m0 = model.fit_corrector(feat[itr], base[itr], true[itr], config.LGBM_PARAMS)
    wins = [np.arange(i * W_WIN, (i + 1) * W_WIN) for i in range(T_WIN)]
    ev = [w[W_WIN - N_EVAL:] for w in wins]

    out = {}
    for name, fn in PATTERNS:
        rho_t = np.array([float(fn(t)) for t in range(T_WIN)])
        S_e = []
        for t in range(T_WIN):
            i_e = ist[ev[t]]
            S_e.append(conformal.score(drift.inject_correlation_drift(true[i_e], nfc[i_e], rho_t[t], rng),
                                       base[i_e], model.predict_corrector(m0, feat[i_e])))
        q_true = np.array([metrics.q_star(S_e[t], config.ALPHA) for t in range(T_WIN)])
        rows = {}
        for beta in BETAS:
            q_est = np.zeros(T_WIN); cov = np.zeros(T_WIN); covmon = np.zeros(T_WIN); alarm = np.zeros(T_WIN)
            for t in range(T_WIN):
                mi = monitor_idx(wins[t], nfc[ist], beta, rng)
                i_m = ist[mi]
                y_m = drift.inject_correlation_drift(true[i_m], nfc[i_m], rho_t[t], rng)
                s_m = conformal.score(y_m, base[i_m], model.predict_corrector(m0, feat[i_m]))
                qe = metrics.q_star(s_m, config.ALPHA)
                q_est[t] = qe
                cov[t] = metrics.coverage_at_q(S_e[t], qe)             # 真实覆盖
                covmon[t] = metrics.coverage_at_q(s_m, qe)             # 监控者看到
                ub = covmon[t] + 1.645 * np.sqrt(max(covmon[t] * (1 - covmon[t]), 1e-9) / len(s_m))
                alarm[t] = 1.0 if ub < (1.0 - config.ALPHA) else 0.0
            rows[beta] = dict(q_est=q_est, cov=cov, covmon=covmon, alarm=alarm)
        out[name] = dict(q_true=q_true, betas=rows, rho=rho_t)
    return out


def main():
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    print("=" * 100)
    print("A5 前提破坏的剂量-响应 | 目标覆盖 %.2f | 种子 %s | beta ∈ %s"
          % (1 - config.ALPHA, SEEDS, BETAS))
    print("beta = 监控样本中『廉价查询(nfc<=1)』的比例；漂移只作用于 nfc>=2 -> beta 越高，监控越看不见漂移")
    print("=" * 100)
    per = []
    for s in SEEDS:
        print("  seed %d ..." % s, flush=True)
        per.append(run_one_seed(s, base, true, feat, nfc))
    names = [n for n, _ in PATTERNS]

    print("\n表 A  ★剂量-响应：不确定性低估倍数 exp(q*_true - q*_est)")
    print("%-11s |" % "模式" + "".join(" %10s |" % ("b=%.2f" % b) for b in BETAS))
    print("-" * 100)
    for n in names:
        cells = []
        for b in BETAS:
            g = float(np.exp(np.mean([d[n]["q_true"].mean() for d in per])
                             - np.mean([d[n]["betas"][b]["q_est"].mean() for d in per])))
            cells.append("%9.2fx" % g)
        print("%-11s |" % n + "".join(" %10s |" % c for c in cells))

    print("\n表 B  真实覆盖率（%d 条评估段）" % N_EVAL)
    print("%-11s |" % "模式" + "".join(" %10s |" % ("b=%.2f" % b) for b in BETAS))
    print("-" * 100)
    for n in names:
        cells = ["%9.1f%%" % (np.mean([d[n]["betas"][b]["cov"].mean() for d in per]) * 100) for b in BETAS]
        print("%-11s |" % n + "".join(" %10s |" % c for c in cells))

    print("\n表 C  监控者看到的覆盖率（它用来判断'一切正常'的量）")
    print("%-11s |" % "模式" + "".join(" %10s |" % ("b=%.2f" % b) for b in BETAS))
    print("-" * 100)
    for n in names:
        cells = ["%9.1f%%" % (np.mean([d[n]["betas"][b]["covmon"].mean() for d in per]) * 100) for b in BETAS]
        print("%-11s |" % n + "".join(" %10s |" % c for c in cells))

    print("\n表 D  ★报警率（真实覆盖已崩却仍不报警 = 盲区）")
    print("%-11s |" % "模式" + "".join(" %10s |" % ("b=%.2f" % b) for b in BETAS))
    print("-" * 100)
    for n in names:
        cells = ["%9.1f%%" % (np.mean([d[n]["betas"][b]["alarm"].mean() for d in per]) * 100) for b in BETAS]
        print("%-11s |" % n + "".join(" %10s |" % c for c in cells))

    traj = {n: {"q_true": np.mean([d[n]["q_true"] for d in per], axis=0).tolist(),
                **{("b%.2f" % b): {k: np.mean([d[n]["betas"][b][k] for d in per], axis=0).tolist()
                                   for k in ["q_est", "cov", "covmon", "alarm"]} for b in BETAS}}
            for n in names}
    traj["_meta"] = dict(betas=BETAS)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_qstar_dose.npy")
    np.save(out, traj, allow_pickle=True)
    print("\n已存 %s" % out)


if __name__ == "__main__":
    main()
