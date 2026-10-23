# -*- coding: utf-8 -*-
"""q* 口径下的【前提检验】：无偏监控 vs 标签延迟 vs 有偏监控。

问题
----
q*(alpha) 由【监控样本】估计（= RECAL-MON 报告的宽度）。这要求监控样本与被评分流量可交换。
A2 已证：无偏时 q*_est 与 q*_true 差 <2%（可靠）。
本实验检验前提被破坏时的后果（贡献 ③ 的定量版本）：
    DELAY1 : 用【上一窗口】的监控标签 -> 监控分布滞后一窗
    BIAS   : 监控只取"廉价查询"（这里取去重过滤列数 <=1 的查询）
             注意：本实验的漂移只作用于 nfc>=2 的查询，故该监控集【完全看不到漂移】

报什么
------
  q*_true       : 评估段自身的诚实最小宽度（基准真值）
  q*_est_arm    : 该臂用监控样本估出的宽度（= 该臂实际报告的宽度）
  缺口 = exp(q*_true - q*_est)  -> >1 表示【低估不确定性】，即"又窄又错"
  覆盖率（辅助读数）
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

def ru(t): return 2.0
def rp(t): return 1.0 * (1.0 + np.sin(2.0 * np.pi * t / 10.0))
def rt(t): return 2.0 * t / (T_WIN - 1)
def rl(t): return 2.0 if t in SPIKE else 0.2
PATTERNS = [("none", lambda t: 0.0), ("uniform", ru), ("periodic", rp), ("trend", rt), ("long-tail", rl)]
ARMS = ["RECAL(无偏)", "DELAY1(滞后)", "BIAS(有偏)"]


def run_one_seed(seed, base, true, feat, nfc):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(true))
    itr = idx[:config.N_TRAIN]
    ist = idx[config.N_TRAIN + config.N_CAL:config.N_TRAIN + config.N_CAL + N_STREAM]
    m0 = model.fit_corrector(feat[itr], base[itr], true[itr], config.LGBM_PARAMS)

    wins = [np.arange(i * W_WIN, (i + 1) * W_WIN) for i in range(T_WIN)]
    mon = [w[:N_MON] for w in wins]
    ev = [w[W_WIN - N_EVAL:] for w in wins]
    # BIAS：监控只取 nfc<=1 的查询（模拟"只标得起廉价查询"）；每窗口取满 N_MON 条
    mon_bias = []
    for w in wins:
        cheap = np.array([i for i in w if nfc[ist[i]] <= 1], dtype=int)
        mon_bias.append(cheap[:N_MON] if len(cheap) >= N_MON else cheap)

    out = {}
    for name, fn in PATTERNS:
        rho_t = np.array([float(fn(t)) for t in range(T_WIN)])
        # 无偏监控 / 评估 的漂移后分数
        S_m, S_e, S_b = [], [], []
        for t in range(T_WIN):
            i_m, i_e = ist[mon[t]], ist[ev[t]]
            i_b = ist[mon_bias[t]]
            S_m.append(conformal.score(drift.inject_correlation_drift(true[i_m], nfc[i_m], rho_t[t], rng),
                                       base[i_m], model.predict_corrector(m0, feat[i_m])))
            S_e.append(conformal.score(drift.inject_correlation_drift(true[i_e], nfc[i_e], rho_t[t], rng),
                                       base[i_e], model.predict_corrector(m0, feat[i_e])))
            S_b.append(conformal.score(drift.inject_correlation_drift(true[i_b], nfc[i_b], rho_t[t], rng),
                                       base[i_b], model.predict_corrector(m0, feat[i_b])))
        q_true = np.array([metrics.q_star(S_e[t], config.ALPHA) for t in range(T_WIN)])
        q_rec = np.array([metrics.q_star(S_m[t], config.ALPHA) for t in range(T_WIN)])
        q_bias = np.array([metrics.q_star(S_b[t], config.ALPHA) for t in range(T_WIN)])
        q_delay = np.concatenate([[q_rec[0]], q_rec[:-1]])      # 用上一窗口的监控结果

        armq = {"RECAL(无偏)": q_rec, "DELAY1(滞后)": q_delay, "BIAS(有偏)": q_bias}
        armS = {"RECAL(无偏)": S_m, "DELAY1(滞后)": S_m, "BIAS(有偏)": S_b}
        rec = dict(rho=rho_t, q_true=q_true, n_bias=int(len(mon_bias[0])))
        for a in ARMS:
            q = armq[a]
            rec["q_" + a] = q
            # 真实覆盖（评估段，检测器不可见）
            rec["cov_" + a] = np.array([metrics.coverage_at_q(S_e[t], q[t]) for t in range(T_WIN)])
            # 监控者【自己看到】的覆盖 + 是否报警（单侧 95% 二项控制，检测器实际可用的信息）
            cm, al = np.zeros(T_WIN), np.zeros(T_WIN)
            for t in range(T_WIN):
                s_m = armS[a][t] if a != "DELAY1(滞后)" else armS[a][max(t - 1, 0)]
                cm[t] = metrics.coverage_at_q(s_m, q[t])
                n_m = len(s_m)
                ub = cm[t] + 1.645 * np.sqrt(max(cm[t] * (1 - cm[t]), 1e-9) / n_m)
                al[t] = 1.0 if ub < (1.0 - config.ALPHA) else 0.0
            rec["covmon_" + a] = cm
            rec["alarm_" + a] = al
        out[name] = rec
    return out


def main():
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    print("=" * 104)
    print("q* 口径下的前提检验 | 目标覆盖 %.2f | 种子 %s | %d 窗口 x %d（监控 %d / 评估 %d）"
          % (1 - config.ALPHA, SEEDS, T_WIN, W_WIN, N_MON, N_EVAL))
    print("=" * 104)
    per = []
    for s in SEEDS:
        print("  seed %d ..." % s, flush=True)
        per.append(run_one_seed(s, base, true, feat, nfc))

    names = [n for n, _ in PATTERNS]
    def m(n, k): return float(np.mean([d[n][k].mean() for d in per]))

    print("\n表 A  各臂估计出的宽度 q*_est  vs  评估段真值 q*_true（对数尺度）")
    print("%-11s | %10s | %14s | %14s | %14s" % ("模式", "q*_true", "RECAL(无偏)", "DELAY1(滞后)", "BIAS(有偏)"))
    print("-" * 104)
    for n in names:
        print("%-11s | %9.3f | %13.3f | %13.3f | %13.3f |" %
              (n, m(n, "q_true"), m(n, "q_RECAL(无偏)"), m(n, "q_DELAY1(滞后)"), m(n, "q_BIAS(有偏)")))

    print("\n表 B  ★不确定性低估倍数 = exp(q*_true - q*_est)  (1.0x=诚实; >1x=又窄又错)")
    print("%-11s | %16s | %16s | %16s" % ("模式", "RECAL(无偏)", "DELAY1(滞后)", "BIAS(有偏)"))
    print("-" * 104)
    for n in names:
        g = [float(np.exp(m(n, "q_true") - m(n, "q_" + a))) for a in ARMS]
        print("%-11s | %15.2fx | %15.2fx | %15.2fx |" % (n, g[0], g[1], g[2]))

    print("\n表 C  覆盖率（辅助读数）：真实覆盖 vs 【监控者自己看到】的覆盖")
    print("%-11s | %22s | %22s | %22s" % ("模式", "RECAL(无偏)", "DELAY1(滞后)", "BIAS(有偏)"))
    print("%-11s | %10s %11s | %10s %11s | %10s %11s" % ("", "真实", "监控看到", "真实", "监控看到", "真实", "监控看到"))
    print("-" * 104)
    for n in names:
        cells = []
        for a in ARMS:
            cells.append("%9.1f%% %10.1f%%" % (m(n, "cov_" + a) * 100, m(n, "covmon_" + a) * 100))
        print("%-11s | %s | %s | %s |" % (n, cells[0], cells[1], cells[2]))

    print("\n表 D  ★监控器盲区：报警率（单侧 95% 二项控制；无漂移流上的报警率即误报率）")
    print("%-11s | %16s | %16s | %16s" % ("模式", "RECAL(无偏)", "DELAY1(滞后)", "BIAS(有偏)"))
    print("-" * 104)
    for n in names:
        a_ = [m(n, "alarm_" + a) * 100 for a in ARMS]
        print("%-11s | %15.1f%% | %15.1f%% | %15.1f%% |" % (n, a_[0], a_[1], a_[2]))
    print("  要点：若某臂在真实覆盖已崩的模式下报警率仍≈误报率，说明【监控器瞎了】——")
    print("        它看到的是被自己做偏的样本，报'一切正常'，而真实可靠性已经崩塌。")

    traj = {n: {k: np.mean([d[n][k] for d in per], axis=0).tolist()
                for k in (["rho", "q_true"] + ["q_" + a for a in ARMS] + ["cov_" + a for a in ARMS]
                          + ["covmon_" + a for a in ARMS] + ["alarm_" + a for a in ARMS])}
            for n in names}
    traj["_meta"] = dict(n_bias=per[0][names[0]]["n_bias"], arms=ARMS)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_qstar_precond.npy")
    np.save(out, traj, allow_pickle=True)
    print("\n已存 %s" % out)


if __name__ == "__main__":
    main()
