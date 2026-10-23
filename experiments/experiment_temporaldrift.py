# -*- coding: utf-8 -*-
"""时序可靠性失效监控实验：把查询集当成时间流，漂移强度按 DriftBench 的 4 种时间模式演化，
评估有标签覆盖失效信号、报警时机和修复后的独立评估表现；该信号不能识别漂移原因。

完整化要点（相对初版）
1. 修正检测延迟口径：先定义 onset（冻结覆盖首次 < tau），只把 onset【之后】的首次报警记为延迟；
   早于 onset 的报警单独计为 early alarm（保守报警，不算误报、也不算漏报）。
2. 多个随机种子（5 个），所有指标报 mean ± std。
3. 监控标签预算 L 的敏感性扫描（50/100/150/200）。
4. 落盘逐窗口轨迹（供出图）。

设计
- 切分：train 20000 / cal 3000 / stream 10000 / pool 8000（互不相交）。
- 流：T=20 窗口 × 500 条；每窗口前 L 条(监控标签，检测器可见) + 后 300 条(评估，检测器不可见)。
- 漂移：相关性漂移，rho(t) 按模式演化；标尺与正文一致（rho=2.0 即 7.39x）。
- 检测：单窗口 Wald 正态近似 —— ph + 1.645*sqrt(ph(1-ph)/n) < (1-alpha) 才报警；不作严格 95% 或序贯误报控制声明。
- 修复：报警后从 pool 取【互不重叠】的 1000 条新标签（500 重训 + 500 重校准），冷却 2 窗口。
- 对照：冻结模型（从不修复）。
"""
import os, sys, warnings, json
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, data, model, conformal, drift

T_WIN    = 20
W_WIN    = 500
N_EVAL   = 300
TAU      = 0.85
COOLDOWN = 2
N_REPAIR = 1000
N_STREAM = T_WIN * W_WIN
N_POOL   = 8000
Z        = 1.645          # Wald 单窗口近似的正态临界值，不是序贯控制
SEEDS    = [42, 43, 44, 45, 46]
L_LIST   = [50, 100, 150, 200]
L_MAIN   = 150
SPIKE    = [6, 13, 18]

def rho_uniform(t):  return 2.0
def rho_periodic(t): return 1.0 * (1.0 + np.sin(2.0 * np.pi * t / 10.0))
def rho_trend(t):    return 2.0 * t / (T_WIN - 1)
def rho_longtail(t): return 2.0 if t in SPIKE else 0.2

PATTERNS = [
    ("none",      lambda t: 0.0),
    ("uniform",   rho_uniform),
    ("periodic",  rho_periodic),
    ("trend",     rho_trend),
    ("long-tail", rho_longtail),
]

def alarm_decision(ph, n, target):
    ub = ph + Z * np.sqrt(max(ph * (1.0 - ph), 1e-9) / n)
    return bool(ub < target)

def run_one_seed(seed, base, true, feat, nfc):
    rng = np.random.RandomState(seed)
    target = 1.0 - config.ALPHA
    n_tr, n_ca = config.N_TRAIN, config.N_CAL
    idx = rng.permutation(len(true))
    itr = idx[:n_tr]
    ica = idx[n_tr:n_tr + n_ca]
    ist = idx[n_tr + n_ca:n_tr + n_ca + N_STREAM]
    iad = idx[n_tr + n_ca + N_STREAM:n_tr + n_ca + N_STREAM + N_POOL]
    f_tr, b_tr, y_tr = feat[itr], base[itr], true[itr]
    f_ca, b_ca, y_ca = feat[ica], base[ica], true[ica]
    f_st, b_st, y_st, nfc_st = feat[ist], base[ist], true[ist], nfc[ist]
    f_ad, b_ad, y_ad, nfc_ad = feat[iad], base[iad], true[iad], nfc[iad]

    m0 = model.fit_corrector(f_tr, b_tr, y_tr, config.LGBM_PARAMS)
    q0 = conformal.calibrate_q(
        conformal.score(y_ca, b_ca, model.predict_corrector(m0, f_ca)), config.ALPHA)

    wins = [np.arange(i * W_WIN, (i + 1) * W_WIN) for i in range(T_WIN)]
    ev   = [w[W_WIN - N_EVAL:] for w in wins]          # 固定评估段（后 300）
    mon_all = [w[:max(L_LIST)] for w in wins]           # 前 200 条备用作监控

    res = {}
    for name, fn in PATTERNS:
        rho_t = np.array([float(fn(t)) for t in range(T_WIN)])
        # 冻结轨迹（与 L 无关）
        y_e = [drift.inject_correlation_drift(y_st[ev[t]], nfc_st[ev[t]], rho_t[t], rng)
               for t in range(T_WIN)]
        cov_fro = np.array([conformal.coverage(
            conformal.score(y_e[t], b_st[ev[t]], model.predict_corrector(m0, f_st[ev[t]])), q0)
            for t in range(T_WIN)])
        onset = next((t for t in range(T_WIN) if cov_fro[t] < TAU), None)

        for L in L_LIST:
            m_cur, q_cur = m0, q0
            cursor, last_repair, n_rep = 0, -99, 0
            alarms, cov_our = [], np.zeros(T_WIN)
            for t in range(T_WIN):
                mi = wins[t][:L]
                y_m = drift.inject_correlation_drift(y_st[mi], nfc_st[mi], rho_t[t], rng)
                ph = conformal.coverage(
                    conformal.score(y_m, b_st[mi], model.predict_corrector(m_cur, f_st[mi])), q_cur)
                fired = alarm_decision(ph, L, target) and (t - last_repair) > COOLDOWN
                if fired:
                    s = slice(cursor, cursor + N_REPAIR); cursor += N_REPAIR
                    y_a = drift.inject_correlation_drift(y_ad[s], nfc_ad[s], rho_t[t], rng)
                    m_cur = model.fit_corrector(f_ad[s][:500], b_ad[s][:500], y_a[:500], config.LGBM_PARAMS)
                    q_cur = conformal.calibrate_q(
                        conformal.score(y_a[500:], b_ad[s][500:],
                                        model.predict_corrector(m_cur, f_ad[s][500:])), config.ALPHA)
                    n_rep += 1; last_repair = t; alarms.append(t)
                cov_our[t] = conformal.coverage(
                    conformal.score(y_e[t], b_st[ev[t]], model.predict_corrector(m_cur, f_st[ev[t]])), q_cur)
            lat = None
            if onset is not None:
                after = [a for a in alarms if a >= onset]
                lat = (after[0] - onset) if after else None
            early = sum(1 for a in alarms if onset is None or a < onset)
            res[(name, L)] = dict(
                cov_fro=cov_fro.tolist(), cov_our=cov_our.tolist(), rho=rho_t.tolist(),
                onset=onset, latency=lat, early=early, n_rep=n_rep,
                labels=n_rep * N_REPAIR, n_alarm=len(alarms),
                pass_rate=float(np.mean(cov_our >= TAU)),
                mean_fro=float(cov_fro.mean()), mean_our=float(cov_our.mean()),
                min_our=float(cov_our.min()))
    return res

def main():
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    print("加载 %d 条；种子 %s；监控预算 %s；主配置 L=%d；流 %d 窗口 x %d 条"
          % (len(true), SEEDS, L_LIST, L_MAIN, T_WIN, W_WIN))

    per_seed = []
    for s in SEEDS:
        print("  seed %d ..." % s); sys.stdout.flush()
        per_seed.append(run_one_seed(s, base, true, feat, nfc))

    def agg(pattern, L, key):
        v = np.array([d[(pattern, L)][key] for d in per_seed], dtype=float)
        return v.mean(), v.std()

    print("\n" + "=" * 100)
    print("表 A  主配置（L=%d，5 种子 mean±std）：覆盖与成本" % L_MAIN)
    print("%-12s %16s %16s %12s %10s %12s" %
          ("模式", "冻结覆盖", "我们覆盖", "达标占比", "修复次数", "标签开销"))
    print("-" * 100)
    for name, _ in PATTERNS:
        mf, sf = agg(name, L_MAIN, "mean_fro"); mo, so = agg(name, L_MAIN, "mean_our")
        pr, spr = agg(name, L_MAIN, "pass_rate"); nr, snr = agg(name, L_MAIN, "n_rep")
        print("%-12s %15.1f%% %15.1f%% %11.1f%% %9.1f %11.0f"
              % (name, mf * 100, mo * 100, pr * 100, nr, nr * N_REPAIR))
    print("  注：none 行为对照（无漂移），其修复次数即误报所触发的修复。")

    print("\n表 B  主配置（L=%d）：检测延迟与报警结构" % L_MAIN)
    print("%-12s %10s %12s %14s %14s" % ("模式", "报警总数", "提前报警", "检测延迟(窗口)", "有报警种子数"))
    print("-" * 100)
    for name, _ in PATTERNS:
        na, _s = agg(name, L_MAIN, "n_alarm"); ea, _s2 = agg(name, L_MAIN, "early")
        lat = [d[(name, L_MAIN)]["latency"] for d in per_seed]
        have = sum(1 for x in lat if x is not None)
        latv = [x for x in lat if x is not None]
        latstr = "--" if not latv else "%.1f ± %.1f" % (float(np.mean(latv)), float(np.std(latv)))
        print("%-12s %10.1f %12.1f %14s %14d" % (name, na, ea, latstr, have))

    print("\n表 C  监控标签预算敏感性（5 种子均值；只列会漂移的 4 个模式）")
    print("%-12s %8s %12s %12s %14s %12s" %
          ("模式", "L", "达标占比", "修复次数", "检测延迟", "提前报警"))
    print("-" * 100)
    for name, _ in PATTERNS:
        if name == "none":
            continue
        for L in L_LIST:
            pr, _ = agg(name, L, "pass_rate"); nr, _ = agg(name, L, "n_rep")
            ea, _ = agg(name, L, "early")
            latv = [d[(name, L)]["latency"] for d in per_seed if d[(name, L)]["latency"] is not None]
            latstr = "--" if not latv else "%.1f" % float(np.mean(latv))
            print("%-12s %8d %11.1f%% %12.1f %14s %12.1f" % (name, L, pr * 100, nr, latstr, ea))

    print("\n表 D  误报控制（none 对照流，5 种子 × 20 窗口）")
    for L in L_LIST:
        al, _ = agg("none", L, "n_alarm")
        print("  L=%-4d 平均报警数 = %.2f  (期望 %.1f 次/20 窗口 若门槛恰好 95%%)" % (L, al, 1.0))

    # 落盘：主配置逐窗口轨迹（跨种子均值）+ 汇总
    traj = {}
    for name, _ in PATTERNS:
        cf = np.mean([d[(name, L_MAIN)]["cov_fro"] for d in per_seed], axis=0)
        co = np.mean([d[(name, L_MAIN)]["cov_our"] for d in per_seed], axis=0)
        rho = per_seed[0][(name, L_MAIN)]["rho"]
        traj[name] = dict(cov_fro=cf.tolist(), cov_our=co.tolist(), rho=rho)
    summary = {}
    for name, _ in PATTERNS:
        for L in L_LIST:
            for k in ["pass_rate", "mean_fro", "mean_our", "min_our", "n_rep", "labels", "n_alarm", "early"]:
                m, s = agg(name, L, k)
                summary["%s|%d|%s" % (name, L, k)] = [m, s]
            latv = [d[(name, L)]["latency"] for d in per_seed if d[(name, L)]["latency"] is not None]
            summary["%s|%d|latency" % (name, L)] = [float(np.mean(latv)) if latv else -1.0, 0.0]
    o = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_temporal.npy")
    np.save(o, {"traj": traj, "summary": summary}, allow_pickle=True)
    print("\n已存 results_temporal.npy（traj=主配置跨种子均值轨迹, summary=全指标 mean/std）")

if __name__ == "__main__":
    main()
