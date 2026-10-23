# -*- coding: utf-8 -*-
"""A6：审计必须"独立"【且】"有代表性"——修复方案与价格曲线。

由 A5 得到的两个待检验命题
  P1「独立就够了吗？」 把 150 条有偏监控标签对半分成 75 校准 / 75 审计：
                        审计不再同义反复，但样本仍有偏 -> 预计【仍然瞎】
  P2「代表性审计要多少条？」 校准用 (150-a) 条廉价标签，审计用 a 条【代表性】标签：
                        a 越大越能发现偏差，但标签预算被吃掉 -> 价格曲线

臂（每窗口标签预算恒为 150，便于比较）
  self   : 150 廉价 -> 同批校准 + 同批审计（同义反复；A4/A5 的现状）
  split  : 75 廉价校准 + 75 廉价审计（独立但有偏）
  aud10  : 140 廉价校准 + 10 代表性审计
  aud25  : 125 廉价校准 + 25 代表性审计
  aud50  : 100 廉价校准 + 50 代表性审计
  参考臂 rep150: 150 代表性（= A2 的 RECAL，无偏基准）

报：审计者看到的覆盖 / 真实覆盖 / 报警率 / 低估倍数（BIAS 场景 beta=1）
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
Z = 1.645
ARMS = [("rep150", "全代表性(基准)"), ("self", "同批校准+审计"), ("split", "对半：独立但有偏"),
        ("aud10", "审计10条代表性"), ("aud25", "审计25条代表性"), ("aud50", "审计50条代表性")]

def ru(t): return 2.0
def rp(t): return 1.0 * (1.0 + np.sin(2.0 * np.pi * t / 10.0))
def rt(t): return 2.0 * t / (T_WIN - 1)
def rl(t): return 2.0 if t in SPIKE else 0.2
PATTERNS = [("uniform", ru), ("periodic", rp), ("trend", rt), ("long-tail", rl)]


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
        rec = dict(q_true=q_true)
        for key, _lab in ARMS:
            qe = np.zeros(T_WIN); cov = np.zeros(T_WIN); aud = np.zeros(T_WIN); alm = np.zeros(T_WIN)
            for t in range(T_WIN):
                w = wins[t]
                cheap = np.array([i for i in w if nfc[ist[i]] <= 1], dtype=int)
                noncheap = np.array([i for i in w if nfc[ist[i]] > 1], dtype=int)
                if key == "rep150":                       # 全代表性：整窗随机
                    cal_i = aud_i = w[:N_MON]
                elif key == "self":                       # 150 廉价，同批
                    cal_i = aud_i = cheap[:N_MON]
                elif key == "split":                      # 75 廉价校准 + 75 廉价审计
                    cal_i = cheap[:75]; aud_i = cheap[75:150]
                else:                                     # a 条【整窗随机】代表性审计（与校准集不相交）
                    a = int(key[3:])
                    cal_i = cheap[:N_MON - a]
                    pool = np.setdiff1d(w, cal_i)          # 排除校准用到的行，保证独立
                    aud_i = rng.choice(pool, size=min(a, len(pool)), replace=False) if a and len(pool) else np.array([], dtype=int)
                if len(cal_i) == 0 or len(aud_i) == 0:
                    qe[t] = np.nan; cov[t] = aud[t] = alm[t] = np.nan; continue
                i_c, i_a = ist[cal_i], ist[aud_i]
                s_c = conformal.score(drift.inject_correlation_drift(true[i_c], nfc[i_c], rho_t[t], rng),
                                      base[i_c], model.predict_corrector(m0, feat[i_c]))
                s_a = conformal.score(drift.inject_correlation_drift(true[i_a], nfc[i_a], rho_t[t], rng),
                                      base[i_a], model.predict_corrector(m0, feat[i_a]))
                q = metrics.q_star(s_c, config.ALPHA)
                qe[t] = q
                cov[t] = metrics.coverage_at_q(S_e[t], q)          # 真实覆盖
                aud[t] = metrics.coverage_at_q(s_a, q)             # 审计者看到
                ub = aud[t] + Z * np.sqrt(max(aud[t] * (1 - aud[t]), 1e-9) / max(len(s_a), 1))
                alm[t] = 1.0 if ub < (1.0 - config.ALPHA) else 0.0
            rec[key] = dict(q_est=qe, cov=cov, aud=aud, alarm=alm)
        out[name] = rec
    return out


def main():
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    print("=" * 104)
    print("A6 审计的设计规则：独立 且 有代表性 | 漂移场景=monitor 只取廉价标签（beta=1）| 目标覆盖 %.2f" % (1 - config.ALPHA))
    print("每窗口标签预算恒为 150")
    print("=" * 104)
    per = []
    for s in SEEDS:
        print("  seed %d ..." % s, flush=True)
        per.append(run_one_seed(s, base, true, feat, nfc))
    names = [n for n, _ in PATTERNS]

    def M(mode, key, field):
        return float(np.mean([np.nanmean(d[mode][key][field]) for d in per]))

    for title, field, fmt in [("表 A  审计者看到的覆盖率（它唯一的判断依据）", "aud", "%9.1f%%"),
                              ("表 B  真实覆盖率（评估段）", "cov", "%9.1f%%"),
                              ("表 C  ★报警率（真实覆盖已崩却不报警 = 盲区）", "alarm", "%9.1f%%")]:
        print("\n" + title)
        print("%-11s |" % "模式" + "".join(" %11s |" % lab for _k, lab in ARMS))
        print("-" * 104)
        for n in names:
            cells = [fmt % (M(n, k, field) * 100) for k, _lab in ARMS]
            print("%-11s |" % n + "".join(" %11s |" % c for c in cells))

    print("\n表 D  不确定性低估倍数 exp(q*_true - q*_est)")
    print("%-11s |" % "模式" + "".join(" %11s |" % lab for _k, lab in ARMS))
    print("-" * 104)
    for n in names:
        cells = []
        for k, _lab in ARMS:
            qt = float(np.mean([d[n]["q_true"].mean() for d in per]))
            qe = float(np.mean([np.nanmean(d[n][k]["q_est"]) for d in per]))
            cells.append("%10.2fx" % float(np.exp(qt - qe)))
        print("%-11s |" % n + "".join(" %11s |" % c for c in cells))

    traj = {n: {k: {f: np.nanmean([d[n][k][f] for d in per], axis=0).tolist()
                    for f in ["q_est", "cov", "aud", "alarm"]} for k, _l in ARMS} for n in names}
    traj["_meta"] = dict(arms=[k for k, _ in ARMS])
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_qstar_audit.npy")
    np.save(out, traj, allow_pickle=True)
    print("\n已存 %s" % out)


if __name__ == "__main__":
    main()
