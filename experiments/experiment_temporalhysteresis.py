# -*- coding: utf-8 -*-
"""实验 #3：在线修复策略对比 —— single / sustained2(迟滞) / recal_only(仅重校准)。

背景（见 experiment_temporaldrift.py）：single 检测器一报警就【重训+重校准】，
long-tail 下在尖峰上修复后、漂移回落时覆盖反而掉到 81-85%。

本实验比较三种策略：
  single     : 单窗口触发 -> 重训修正器 + 重校准（1000 标签）
  sustained2 : 连续 2 窗口触发 -> 重训 + 重校准（迟滞；假设它更稳）
  recal_only : 单窗口触发 -> 【只重校准】q，不重训修正器（200 标签）

结论（实测）：naive 迟滞是错的修法 —— long-tail 尖峰只持续 1 窗口，sustained2 一次都不触发，
覆盖退回冻结水平，达标 89%->83%；各模式达标也整体下降。
recal_only 反应快、成本低（200 vs 1000 标签），且漂移回落时能再次靠重校准把覆盖拉回。

5 个种子；DriftBench 4 时间模式 + 对照流。
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, data, model, conformal, drift

T_WIN, W_WIN, N_EVAL = 20, 500, 300
TAU, N_STREAM, N_POOL = 0.85, 10000, 8000
Z, L_MON = 1.645, 150
N_RETRAIN, N_RECAL = 1000, 200
SEEDS = [42, 43, 44, 45, 46]
SPIKE = [6, 13, 18]
MODES = ["single", "sustained2", "recal_only"]

def rho_uniform(t):  return 2.0
def rho_periodic(t): return 1.0 * (1.0 + np.sin(2.0 * np.pi * t / 10.0))
def rho_trend(t):    return 2.0 * t / (T_WIN - 1)
def rho_longtail(t): return 2.0 if t in SPIKE else 0.2

PATTERNS = [("none", lambda t: 0.0), ("uniform", rho_uniform),
            ("periodic", rho_periodic), ("trend", rho_trend), ("long-tail", rho_longtail)]

def triggering(ph, n, target):
    ub = ph + Z * np.sqrt(max(ph * (1.0 - ph), 1e-9) / n)
    return bool(ub < target)

def run_seed(seed, base, true, feat, nfc):
    rng = np.random.RandomState(seed)
    target = 1.0 - config.ALPHA
    idx = rng.permutation(len(true))
    itr = idx[:config.N_TRAIN]
    ica = idx[config.N_TRAIN:config.N_TRAIN + config.N_CAL]
    ist = idx[config.N_TRAIN + config.N_CAL:config.N_TRAIN + config.N_CAL + N_STREAM]
    iad = idx[config.N_TRAIN + config.N_CAL + N_STREAM:config.N_TRAIN + config.N_CAL + N_STREAM + N_POOL]
    f_tr, b_tr, y_tr = feat[itr], base[itr], true[itr]
    f_ca, b_ca, y_ca = feat[ica], base[ica], true[ica]
    f_st, b_st, y_st, nfc_st = feat[ist], base[ist], true[ist], nfc[ist]
    f_ad, b_ad, y_ad, nfc_ad = feat[iad], base[iad], true[iad], nfc[iad]

    m0 = model.fit_corrector(f_tr, b_tr, y_tr, config.LGBM_PARAMS)
    q0 = conformal.calibrate_q(conformal.score(y_ca, b_ca, model.predict_corrector(m0, f_ca)), config.ALPHA)

    wins = [np.arange(i * W_WIN, (i + 1) * W_WIN) for i in range(T_WIN)]
    ev = [w[W_WIN - N_EVAL:] for w in wins]
    mon = [w[:L_MON] for w in wins]

    out = {}
    for name, fn in PATTERNS:
        rho_t = np.array([float(fn(t)) for t in range(T_WIN)])
        y_e = [drift.inject_correlation_drift(y_st[ev[t]], nfc_st[ev[t]], rho_t[t], rng) for t in range(T_WIN)]
        cov_fro = np.array([conformal.coverage(
            conformal.score(y_e[t], b_st[ev[t]], model.predict_corrector(m0, f_st[ev[t]])), q0)
            for t in range(T_WIN)])
        onset = next((t for t in range(T_WIN) if cov_fro[t] < TAU), None)

        for mode in MODES:
            m_cur, q_cur = m0, q0
            cursor, last_repair, n_rep, labels = 0, -99, 0, 0
            streak, alarms, cov_our = 0, [], np.zeros(T_WIN)
            for t in range(T_WIN):
                mi = mon[t]
                y_m = drift.inject_correlation_drift(y_st[mi], nfc_st[mi], rho_t[t], rng)
                ph = conformal.coverage(
                    conformal.score(y_m, b_st[mi], model.predict_corrector(m_cur, f_st[mi])), q_cur)
                trig = triggering(ph, L_MON, target)
                streak = streak + 1 if trig else 0
                need = 1 if mode != "sustained2" else 2
                fired = (streak >= need) and (t - last_repair) > 2
                if fired:
                    if mode == "recal_only":
                        s = slice(cursor, cursor + N_RECAL); cursor += N_RECAL
                        y_a = drift.inject_correlation_drift(y_ad[s], nfc_ad[s], rho_t[t], rng)
                        q_cur = conformal.calibrate_q(
                            conformal.score(y_a, b_ad[s], model.predict_corrector(m_cur, f_ad[s])), config.ALPHA)
                        labels += N_RECAL
                    else:
                        s = slice(cursor, cursor + N_RETRAIN); cursor += N_RETRAIN
                        y_a = drift.inject_correlation_drift(y_ad[s], nfc_ad[s], rho_t[t], rng)
                        m_cur = model.fit_corrector(f_ad[s][:500], b_ad[s][:500], y_a[:500], config.LGBM_PARAMS)
                        q_cur = conformal.calibrate_q(
                            conformal.score(y_a[500:], b_ad[s][500:],
                                            model.predict_corrector(m_cur, f_ad[s][500:])), config.ALPHA)
                        labels += N_RETRAIN
                    n_rep += 1; last_repair = t; alarms.append(t); streak = 0
                cov_our[t] = conformal.coverage(
                    conformal.score(y_e[t], b_st[ev[t]], model.predict_corrector(m_cur, f_st[ev[t]])), q_cur)
            lat = None
            if onset is not None:
                after = [a for a in alarms if a >= onset]
                lat = (after[0] - onset) if after else None
            early = sum(1 for a in alarms if onset is None or a < onset)
            out[(name, mode)] = dict(
                pass_rate=float(np.mean(cov_our >= TAU)), mean_fro=float(cov_fro.mean()),
                mean_our=float(cov_our.mean()), min_our=float(cov_our.min()),
                n_rep=n_rep, labels=labels, n_alarm=len(alarms), early=early,
                latency=(-1.0 if lat is None else float(lat)),
                cov_our=cov_our.tolist(), cov_fro=cov_fro.tolist(), rho=rho_t.tolist())
    return out

def main():
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    print("加载 %d 条；策略 %s；种子 %s；监控 L=%d；重训 %d / 仅重校准 %d 标签"
          % (len(true), MODES, SEEDS, L_MON, N_RETRAIN, N_RECAL))
    per = []
    for s in SEEDS:
        print("  seed %d ..." % s); sys.stdout.flush()
        per.append(run_seed(s, base, true, feat, nfc))

    def agg(p, m, k):
        v = np.array([d[(p, m)][k] for d in per], dtype=float)
        return float(v.mean()), float(v.std())

    print("\n" + "=" * 104)
    print("表 1  达标窗口占比（5 种子 mean±std）")
    print("%-12s %16s %16s %16s" % ("模式", "single", "sustained2(迟滞)", "recal_only(仅重校准)"))
    print("-" * 104)
    for name, _ in PATTERNS:
        cells = []
        for m in MODES:
            a, s = agg(name, m, "pass_rate"); cells.append("%.1f%%±%.1f" % (a * 100, s * 100))
        print("%-12s %16s %16s %16s" % (name, cells[0], cells[1], cells[2]))

    print("\n表 2  适应后覆盖 / 标签开销（5 种子均值）")
    print("%-12s %10s %14s %14s %14s %14s %14s %14s" %
          ("模式", "冻结覆盖", "single 覆盖", "sustained 覆盖", "recal 覆盖",
           "single 标签", "sustained 标签", "recal 标签"))
    print("-" * 104)
    for name, _ in PATTERNS:
        mf, _ = agg(name, "single", "mean_fro")
        o1, _ = agg(name, "single", "mean_our"); o2, _ = agg(name, "sustained2", "mean_our"); o3, _ = agg(name, "recal_only", "mean_our")
        l1, _ = agg(name, "single", "labels"); l2, _ = agg(name, "sustained2", "labels"); l3, _ = agg(name, "recal_only", "labels")
        print("%-12s %9.1f%% %13.1f%% %13.1f%% %13.1f%% %13.0f %13.0f %13.0f"
              % (name, mf * 100, o1 * 100, o2 * 100, o3 * 100, l1, l2, l3))

    print("\n表 3  修复次数 / 检测延迟（5 种子均值）")
    print("%-12s %12s %12s %12s %14s %14s %14s" %
          ("模式", "single 修复", "sustained 修复", "recal 修复", "single 延迟", "sustained 延迟", "recal 延迟"))
    print("-" * 104)
    for name, _ in PATTERNS:
        r1, _ = agg(name, "single", "n_rep"); r2, _ = agg(name, "sustained2", "n_rep"); r3, _ = agg(name, "recal_only", "n_rep")
        cells = []
        for m in MODES:
            lat = [d[(name, m)]["latency"] for d in per if d[(name, m)]["latency"] >= 0]
            cells.append("--" if not lat else "%.1f" % float(np.mean(lat)))
        print("%-12s %12.1f %12.1f %12.1f %14s %14s %14s" % (name, r1, r2, r3, cells[0], cells[1], cells[2]))

    traj = {}
    for mode in MODES:
        traj[mode] = dict(
            cov_fro=np.mean([d[("long-tail", mode)]["cov_fro"] for d in per], axis=0).tolist(),
            cov_our=np.mean([d[("long-tail", mode)]["cov_our"] for d in per], axis=0).tolist(),
            rho=per[0][("long-tail", "single")]["rho"])
    summary = {}
    for name, _ in PATTERNS:
        for mode in MODES:
            for k in ["pass_rate", "mean_fro", "mean_our", "min_our", "n_rep", "labels", "n_alarm", "early", "latency"]:
                summary["%s|%s|%s" % (name, mode, k)] = list(agg(name, mode, k))
    o = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_hysteresis.npy")
    np.save(o, {"summary": summary, "traj": traj}, allow_pickle=True)
    print("\n已存 results_hysteresis.npy")

if __name__ == "__main__":
    main()
