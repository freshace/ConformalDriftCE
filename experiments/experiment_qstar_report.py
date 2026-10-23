# -*- coding: utf-8 -*-
"""按 q*(alpha) 口径重报：FROZEN / RECAL-MON / ACI 三臂（不含重训，故分钟级）。

口径（见 metrics.py，唯一口径）
  q*(alpha)      = 用【当前窗口自身】分数校准出的宽度 = 达到目标覆盖所需的最小宽度
  rho = q_报告/q*   <1 不足(覆盖会掉) ~1 诚实 >1 过度保守(买覆盖)

三臂
  FROZEN : 冻结 m0 + 旧校准集定的 q0        —— 期望 rho<1
  RECAL  : 冻结 m0，每窗口用监控标签重校准  —— 期望 rho~1（按构造成立）
  ACI    : 冻结 m0，alpha_t 自适应（gamma=0.05）—— 期望夹在中间

切分/漂移/窗口/种子与 experiment_aci_baseline.py 完全一致（可复现）。
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, data, model, conformal, drift, metrics

T_WIN, W_WIN, N_MON, N_EVAL = 20, 500, 150, 300
N_STREAM, N_POOL = T_WIN * W_WIN, 8000
SEEDS = [42, 43, 44, 45, 46]
SPIKE = [6, 13, 18]
GAMMA = 0.05
ALPHA_LO, ALPHA_HI = 0.002, 0.30
COOLDOWN, N_REPAIR, W_ABS = 2, 1000, 12.0     # 重训臂：绝对宽度预算 12x + 接受门槛

def ru(t): return 2.0
def rp(t): return 1.0 * (1.0 + np.sin(2.0 * np.pi * t / 10.0))
def rt(t): return 2.0 * t / (T_WIN - 1)
def rl(t): return 2.0 if t in SPIKE else 0.2
PATTERNS = [("none", lambda t: 0.0), ("uniform", ru), ("periodic", rp), ("trend", rt), ("long-tail", rl)]


def run_one_seed(seed, base, true, feat, nfc):
    rng = np.random.RandomState(seed)
    a = config.ALPHA
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
        # 漂移后的监控/评估分数（模型恒为 m0 —— 三臂共用同一模型）
        S_mon, S_eval = [], []
        for t in range(T_WIN):
            i_m, i_e = ist[mon[t]], ist[ev[t]]
            y_m = drift.inject_correlation_drift(true[i_m], nfc[i_m], rho_t[t], rng)
            y_e = drift.inject_correlation_drift(true[i_e], nfc[i_e], rho_t[t], rng)
            S_mon.append(conformal.score(y_m, base[i_m], model.predict_corrector(m0, feat[i_m])))
            S_eval.append(conformal.score(y_e, base[i_e], model.predict_corrector(m0, feat[i_e])))

        qstrue = np.array([metrics.q_star(S_eval[t], config.ALPHA) for t in range(T_WIN)])   # 真实最小诚实宽度
        qsest = np.array([metrics.q_star(S_mon[t], config.ALPHA) for t in range(T_WIN)])    # 监控估计（= RECAL 报告值）

        # FROZEN
        q_fro = np.full(T_WIN, q0)
        # ACI（与 experiment_aci_baseline 同一动力学）
        q_aci = np.zeros(T_WIN); a_t = a
        for t in range(T_WIN):
            qt = metrics.q_star(s_cal, a_t)
            q_aci[t] = qt
            err = 1.0 - metrics.coverage_at_q(S_mon[t], qt)
            a_t = float(np.clip(a_t + GAMMA * (config.ALPHA - err), ALPHA_LO, ALPHA_HI))
        # RECAL
        q_rec = qsest.copy()

        # GUARD_ABS：L1 每窗口免费重校准 + L2 宽度预算触发重训（带接受门槛，采纳后重参考）
        ipool = idx[config.N_TRAIN + config.N_CAL + N_STREAM:
                    config.N_TRAIN + config.N_CAL + N_STREAM + N_POOL]
        m_cur, q_ret = m0, np.zeros(T_WIN)
        cursor, last, n_acc, n_rej = 0, -99, 0, 0
        for t in range(T_WIN):
            i_m = ist[mon[t]]
            y_m = drift.inject_correlation_drift(true[i_m], nfc[i_m], rho_t[t], rng)
            cm = model.predict_corrector(m_cur, feat[i_m])
            q_t = metrics.q_star(conformal.score(y_m, base[i_m], cm), config.ALPHA)
            if (float(np.exp(q_t)) > W_ABS) and (t - last) > COOLDOWN and (cursor + N_REPAIR) <= N_POOL:
                sl = ipool[cursor:cursor + N_REPAIR]; cursor += N_REPAIR
                y_a = drift.inject_correlation_drift(true[sl], nfc[sl], rho_t[t], rng)
                m_new = model.fit_corrector(feat[sl][:500], base[sl][:500], y_a[:500], config.LGBM_PARAMS)
                # 接受门槛：用【同一批监控标签】比较新旧模型
                q_cand = metrics.q_star(
                    conformal.score(y_m, base[i_m], model.predict_corrector(m_new, feat[i_m])), config.ALPHA)
                if q_cand < q_t:
                    m_cur, q_t = m_new, q_cand
                    n_acc += 1
                else:
                    n_rej += 1
                last = t
            q_ret[t] = q_t

        out[name] = dict(
            rho=rho_t, q0=float(q0), qstrue=qstrue, qsest=qsest,
            q_fro=q_fro, q_aci=q_aci, q_rec=q_rec, q_ret=q_ret,
            cov_fro=np.array([metrics.coverage_at_q(S_eval[t], q_fro[t]) for t in range(T_WIN)]),
            cov_aci=np.array([metrics.coverage_at_q(S_eval[t], q_aci[t]) for t in range(T_WIN)]),
            cov_rec=np.array([metrics.coverage_at_q(S_eval[t], q_rec[t]) for t in range(T_WIN)]),
            cov_ret=np.array([metrics.coverage_at_q(S_eval[t], q_ret[t]) for t in range(T_WIN)]),
            labels_ret=(n_acc + n_rej) * N_REPAIR, n_acc=n_acc, n_rej=n_rej,
        )
    return out


def main():
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    print("=" * 108)
    print("q*(alpha) 口径重报：FROZEN / RECAL / ACI | 目标覆盖 1-alpha=%.2f | 种子 %s | %d 窗口 x %d"
          % (1 - config.ALPHA, SEEDS, T_WIN, W_WIN))
    print("口径: q*=当前窗口分数的最小诚实宽度; rho=q报告/q* (<1 不足, ~1 诚实, >1 买覆盖)")
    print("=" * 108)
    per = []
    for s in SEEDS:
        print("  seed %d ..." % s, flush=True)
        per.append(run_one_seed(s, base, true, feat, nfc))

    names = [n for n, _ in PATTERNS]
    def m(n, k): return float(np.mean([d[n][k].mean() for d in per]))
    def mq(n, k): return float(np.mean([d[n][k].mean() for d in per]))

    print("\n表 A  主表：最小诚实宽度 q* vs 各臂报告宽度（对数尺度；5 种子逐窗口均值）")
    print("%-11s | %9s | %9s | %9s | %9s | %9s | %9s" %
          ("模式", "q*_true", "q*_est", "FROZEN", "ACI", "RECAL", "RETRAIN"))
    print("-" * 108)
    for n in names:
        print("%-11s | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f |" %
              (n, mq(n, "qstrue"), mq(n, "qsest"), mq(n, "q_fro"), mq(n, "q_aci"),
               mq(n, "q_rec"), mq(n, "q_ret")))

    print("\n表 B  效率比 rho = 报告宽度 / 最小诚实宽度（<1 静默失效；~1 诚实；>1 买覆盖）")
    print("%-11s | %13s | %13s | %13s | %13s" % ("模式", "FROZEN", "ACI", "RECAL", "RETRAIN"))
    print("-" * 108)
    for n in names:
        r = [mq(n, k) / mq(n, "qstrue") for k in ("q_fro", "q_aci", "q_rec", "q_ret")]
        print("%-11s | %12.3f  | %12.3f  | %12.3f  | %12.3f  |" % (n, *r))

    print("\n表 C  覆盖率（辅助读数）：同一报告宽度下的实际覆盖")
    print("%-11s | %14s | %14s | %14s | %14s" % ("模式", "FROZEN", "ACI", "RECAL", "RETRAIN"))
    print("-" * 108)
    for n in names:
        print("%-11s | %13.1f%% | %13.1f%% | %13.1f%% | %13.1f%% |" %
              (n, m(n, "cov_fro") * 100, m(n, "cov_aci") * 100, m(n, "cov_rec") * 100, m(n, "cov_ret") * 100))

    print("\n表 D  买覆盖的真实价格（宽度倍数 = exp(delta_q)，相对漂移前宽度）")
    print("%-11s | %9s | %16s | %16s | %16s" % ("模式", "q0", "诚实所需宽度倍", "ACI 实付", "RECAL 实付"))
    print("-" * 108)
    for n in names:
        q0 = per[0][n]["q0"]
        w_true = float(np.exp(mq(n, "qstrue") - q0))
        w_aci = float(np.exp(mq(n, "q_aci") - q0))
        w_rec = float(np.exp(mq(n, "q_rec") - q0))
        print("%-11s | %9.3f | %15.2fx | %15.2fx | %15.2fx |" % (n, q0, w_true, w_aci, w_rec))
    print("  读法：FROZEN 报告的仍是 1.00x（旧宽度），而诚实所需已膨胀到 Nx —— 缺口即'不确定性被低估的倍数'。")

    print("\n表 E  宽度缺口 = 诚实所需 / 实际报告（= 1/rho，'模型低估了自己多少倍'）")
    print("%-11s | %16s | %16s | %16s" % ("模式", "FROZEN 缺口", "ACI 缺口", "RECAL 缺口"))
    print("-" * 108)
    for n in names:
        g = [float(np.exp(mq(n, "qstrue") - mq(n, k))) for k in ("q_fro", "q_aci", "q_rec", "q_ret")]
        print("%-11s | %14.2fx | %14.2fx | %14.2fx | %14.2fx |" % (n, *g))

    print("\n表 F  重训臂的标签成本（每次尝试 1000 标签；被拒绝也消耗）")
    print("%-11s | %12s | %12s | %12s | %12s" % ("模式", "标签开销", "接受次数", "拒绝次数", "接受率"))
    print("-" * 108)
    for n in names:
        lab = float(np.mean([d[n]["labels_ret"] for d in per]))
        acc = float(np.mean([d[n]["n_acc"] for d in per]))
        rej = float(np.mean([d[n]["n_rej"] for d in per]))
        rate = acc / (acc + rej) if (acc + rej) > 0 else float("nan")
        rs = ("%.0f%%" % (rate * 100)) if rate == rate else "--"
        print("%-11s | %12.0f | %12.1f | %12.1f | %12s |" % (n, lab, acc, rej, rs))

    traj = {n: {k: np.mean([d[n][k] for d in per], axis=0).tolist()
                for k in ["qstrue", "qsest", "q_fro", "q_aci", "q_rec", "q_ret",
                          "cov_fro", "cov_aci", "cov_rec", "cov_ret", "rho"]} for n in names}
    traj["_q0"] = per[0][names[0]]["q0"]
    traj["_labels_ret"] = {n: float(np.mean([d[n]["labels_ret"] for d in per])) for n in names}
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_qstar.npy")
    np.save(out, traj, allow_pickle=True)
    print("\n已存 %s（逐窗口轨迹，可直接出图）" % out)


if __name__ == "__main__":
    main()
