# -*- coding: utf-8 -*-
"""ACI 对照实验（Gibbs & Candes 2021, Adaptive Conformal Inference）。

目的（决定性实验）：
  「覆盖下降」到底是漂移的固有现象，还是**固定阈值**的产物？
  如果 ACI（自适应调整分位数水平 alpha_t、不用新标签）能在漂移下守住覆盖，
  那么"覆盖率跌破目标"这个信号就不存在 —— 我们论文的机制被抽掉。
  反过来，如果 ACI 只能靠**牺牲宽度**来守住覆盖，那这件事本身就是：

  角度转换：把「适应力度」当漂移信号（alpha_t 的下调 / 区间宽度的膨胀），
  并论证 ACI 守住了 coverage 却隐藏了「估计已经失效」这个事实。

臂（同一流、同一评估段、同一监控预算）：
  FROZEN      : 冻结 m0 + 固定 q0（现有基线；覆盖下降即现有检测信号）
  ACI         : 冻结 m0；alpha_t 自适应（用漂移前的校准分数），err 来自本窗口监控标签
  RECAL-MON   : 冻结 m0；每个窗口用**本窗口监控标签**重校准 q（零额外标签成本）
  OURS        : 报警(二项控制) → 1000 新标签重训+重校准，冷却 2 窗口（现有方法）

检测分析：三个统计量在同一 FPR 下的 TPR / 检测延迟
  stat_cov    = 1 - 冻结覆盖            （现有信号）
  stat_alpha  = alpha_target - alpha_t  （ACI 的"适应力度"）
  stat_width  = q_t / q0                （区间膨胀倍数）

切分与超参完全沿用 experiment_temporaldrift.py（train 20000 / cal 3000 / stream 10000 / pool 8000；种子 42-46）。
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, data, model, conformal, drift

T_WIN    = 20
W_WIN    = 500
N_MON    = 150      # 监控标签（检测器/自适应可见）
N_EVAL   = 300      # 评估段（不可见）
TAU      = 0.85     # 与现有脚本一致的操作点（用于 onset 口径）
COOLDOWN = 2
N_REPAIR = 1000
N_STREAM = T_WIN * W_WIN
N_POOL   = 8000
Z        = 1.645
SEEDS    = [42, 43, 44, 45, 46]
GAMMA_LIST = [0.01, 0.05, 0.20]          # ACI 步长敏感性
GAMMA_MAIN = 0.05
ALPHA_LO, ALPHA_HI = 0.002, 0.30         # alpha_t 裁剪（防止退化到 0 或无效区间）
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

def q_at(s_sorted, alpha):
    """与 conformal.calibrate_q 同一公式：ceil((n+1)(1-alpha)) 次序统计量。"""
    n = len(s_sorted)
    k = min(int(np.ceil((n + 1) * (1.0 - alpha))) - 1, n - 1)
    return float(s_sorted[max(k, 0)])

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
    s_cal = np.sort(conformal.score(y_ca, b_ca, model.predict_corrector(m0, f_ca)))
    q0 = q_at(s_cal, config.ALPHA)

    wins = [np.arange(i * W_WIN, (i + 1) * W_WIN) for i in range(T_WIN)]
    ev   = [w[W_WIN - N_EVAL:] for w in wins]
    mon  = [w[:N_MON] for w in wins]

    out = {}
    for name, fn in PATTERNS:
        rho_t = np.array([float(fn(t)) for t in range(T_WIN)], dtype=float)
        # 漂移后的监控标签与评估标签（同一 rho(t)，互不重叠）
        y_m = [drift.inject_correlation_drift(y_st[mon[t]], nfc_st[mon[t]], rho_t[t], rng)
               for t in range(T_WIN)]
        y_e = [drift.inject_correlation_drift(y_st[ev[t]], nfc_st[ev[t]], rho_t[t], rng)
               for t in range(T_WIN)]
        f_m, b_m = [f_st[mon[t]] for t in range(T_WIN)], [b_st[mon[t]] for t in range(T_WIN)]
        f_e, b_e = [f_st[ev[t]] for t in range(T_WIN)], [b_st[ev[t]] for t in range(T_WIN)]
        corr_m = [model.predict_corrector(m0, f_m[t]) for t in range(T_WIN)]
        corr_e = [model.predict_corrector(m0, f_e[t]) for t in range(T_WIN)]

        # ---- FROZEN ----
        cov_fro = np.array([conformal.coverage(conformal.score(y_e[t], b_e[t], corr_e[t]), q0)
                            for t in range(T_WIN)])
        onset = next((t for t in range(T_WIN) if cov_fro[t] < TAU), None)
        onset90 = next((t for t in range(T_WIN) if cov_fro[t] < target), None)

        # ---- ACI（对每个 gamma）----
        aci = {}
        for g in GAMMA_LIST:
            a = config.ALPHA
            cov = np.zeros(T_WIN); wid = np.zeros(T_WIN); al = np.zeros(T_WIN); sat = 0
            for t in range(T_WIN):
                q_t = q_at(s_cal, a)
                al[t] = a
                wid[t] = float(np.exp(q_t))
                cov[t] = conformal.coverage(conformal.score(y_e[t], b_e[t], corr_e[t]), q_t)
                err = 1.0 - conformal.coverage(conformal.score(y_m[t], b_m[t], corr_m[t]), q_t)
                a_new = a + g * (config.ALPHA - err)
                a_clip = float(np.clip(a_new, ALPHA_LO, ALPHA_HI))
                if a_clip != a_new:
                    sat += 1
                a = a_clip
            aci[g] = dict(cov=cov, wid=wid, alpha=al, sat=sat)

        # ---- RECAL-MON：零额外标签，用本窗口监控标签重校准 q ----
        cov_rm = np.zeros(T_WIN); wid_rm = np.zeros(T_WIN)
        for t in range(T_WIN):
            q_t = conformal.calibrate_q(conformal.score(y_m[t], b_m[t], corr_m[t]), config.ALPHA)
            wid_rm[t] = float(np.exp(q_t))
            cov_rm[t] = conformal.coverage(conformal.score(y_e[t], b_e[t], corr_e[t]), q_t)

        # ---- OURS：报警→重训+重校准（沿用现有脚本逻辑）----
        m_cur, q_cur = m0, q0
        cursor, last_repair, n_rep = 0, -99, 0
        alarms, cov_our, wid_our = [], np.zeros(T_WIN), np.zeros(T_WIN)
        for t in range(T_WIN):
            ph = conformal.coverage(conformal.score(y_m[t], b_m[t], model.predict_corrector(m_cur, f_m[t])), q_cur)
            fired = alarm_decision(ph, N_MON, target) and (t - last_repair) > COOLDOWN
            if fired:
                s = slice(cursor, cursor + N_REPAIR); cursor += N_REPAIR
                if cursor <= N_POOL:
                    y_a = drift.inject_correlation_drift(y_ad[s], nfc_ad[s], rho_t[t], rng)
                    m_cur = model.fit_corrector(f_ad[s][:500], b_ad[s][:500], y_a[:500], config.LGBM_PARAMS)
                    q_cur = conformal.calibrate_q(
                        conformal.score(y_a[500:], b_ad[s][500:],
                                        model.predict_corrector(m_cur, f_ad[s][500:])), config.ALPHA)
                    n_rep += 1; last_repair = t; alarms.append(t)
            wid_our[t] = float(np.exp(q_cur))
            cov_our[t] = conformal.coverage(
                conformal.score(y_e[t], b_e[t], model.predict_corrector(m_cur, f_e[t])), q_cur)

        out[name] = dict(rho=rho_t, cov_fro=cov_fro, onset=onset, onset90=onset90,
                         aci=aci, cov_rm=cov_rm, wid_rm=wid_rm,
                         cov_our=cov_our, wid_our=wid_our, q0=float(q0),
                         n_rep=n_rep, alarms=alarms, wid_fro=float(np.exp(q0)))
    return out

def main():
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    print("ACI 对照实验：%d 条；种子 %s；流 %d 窗口 x %d（监控 %d / 评估 %d）；gamma %s（主 %.2f）"
          % (len(true), SEEDS, T_WIN, W_WIN, N_MON, N_EVAL, GAMMA_LIST, GAMMA_MAIN))
    print("目标覆盖 %.2f；冻结 q0 宽度见下表" % (1 - config.ALPHA))

    per_seed = []
    for s in SEEDS:
        print("  seed %d ..." % s); sys.stdout.flush()
        per_seed.append(run_one_seed(s, base, true, feat, nfc))

    names = [n for n, _ in PATTERNS]
    gm = GAMMA_MAIN

    def mstd(pattern, key, fn=None):
        if fn is None:
            def fn(d, pattern=pattern, key=key):
                return d[pattern][key]
        v = np.array([fn(d) for d in per_seed], dtype=float)
        return float(v.mean()), float(v.std())

    print("\n" + "=" * 108)
    print("表 A  长期覆盖 / 平均区间宽度（5 种子 mean±std）——核心问题：ACI 能不能不用新标签守住覆盖？")
    print("%-11s | %-17s | %-17s | %-17s | %-17s" %
          ("模式", "FROZEN(覆盖/宽度)", "ACI(g=%.2f)" % gm, "RECAL-MON(零成本)", "OURS(1000标签)"))
    print("-" * 108)
    for n in names:
        cf, sf = mstd(n, "cov_fro")
        wf = float(np.mean([d[n]["wid_fro"] for d in per_seed]))
        ca, sa = mstd(n, "cov", lambda d, n=n: d[n]["aci"][gm]["cov"])
        wa = float(np.mean([d[n]["aci"][gm]["wid"].mean() for d in per_seed]))
        cr, sr = mstd(n, "cov_rm")
        wr = float(np.mean([d[n]["wid_rm"].mean() for d in per_seed]))
        co, so = mstd(n, "cov_our")
        wo = float(np.mean([d[n]["wid_our"].mean() for d in per_seed]))
        print("%-11s | %6.1f%% / %7.1fx | %6.1f%% / %7.1fx | %6.1f%% / %7.1fx | %6.1f%% / %7.1fx"
              % (n, cf * 100, wf, ca * 100, wa, cr * 100, wr, co * 100, wo))
    print("  注：宽度为 exp(q) 的窗口均值（相对宽度的几何含义：真值落在 [yhat/e^q, yhat*e^q]）。")

    print("\n表 B  ACI 的 alpha_t 轨迹与饱和（alpha 目标 %.2f；下调=加宽）" % config.ALPHA)
    print("%-11s | %-13s | %-13s | %-13s | %-9s" %
          ("模式", "alpha_end", "alpha_min", "饱和窗口数", "宽度膨胀"))
    print("-" * 108)
    for n in names:
        ae = float(np.mean([d[n]["aci"][gm]["alpha"][-1] for d in per_seed]))
        am = float(np.mean([d[n]["aci"][gm]["alpha"].min() for d in per_seed]))
        st = float(np.mean([d[n]["aci"][gm]["sat"] for d in per_seed]))
        infl = []
        for d in per_seed:
            w0 = np.exp(d[n]["q0"])
            infl.append(d[n]["aci"][gm]["wid"].max() / w0)
        print("%-11s | %13.4f | %13.4f | %13.1f | %8.1fx" % (n, ae, am, st, float(np.mean(infl))))

    print("\n表 C  检测器对比：同一 FPR(=10%%，阈值取 none 模式的 90 分位) 下的 TPR 与检测延迟")
    print("%-11s | %-15s | %-15s | %-15s" % ("模式", "stat_cov(现有)", "stat_alpha(适应力度)", "stat_width(区间膨胀)"))
    print("%-11s | %-15s | %-15s | %-15s" % ("", "TPR / 延迟", "TPR / 延迟", "TPR / 延迟"))
    print("-" * 108)

    # 统计量：none 模式下收集，用于定阈值
    def stats_for(d, n):
        s_cov = 1.0 - d[n]["cov_fro"]
        s_alp = config.ALPHA - d[n]["aci"][gm]["alpha"]
        s_wid = d[n]["aci"][gm]["wid"] / np.exp(d[n]["q0"])
        return s_cov, s_alp, s_wid

    thr = []
    for k in range(3):
        vals = np.concatenate([stats_for(d, "none")[k] for d in per_seed])
        thr.append(float(np.quantile(vals, 0.90)))

    for n in names:
        cells = []
        for k in range(3):
            tprs, lats = [], []
            for d in per_seed:
                s = stats_for(d, n)[k]
                fired = np.where(s > thr[k])[0]
                tprs.append(len(fired) / T_WIN)
                if n != "none":
                    ref = d[n]["onset90"]
                    if ref is not None:
                        aft = [f for f in fired if f >= ref]
                        if aft:
                            lats.append(aft[0] - ref)
            tpr = float(np.mean(tprs)) * 100
            lat = ("%.1f" % float(np.mean(lats))) if lats else "--"
            cells.append("%5.0f%% / %-6s" % (tpr, lat))
        print("%-11s | %-15s | %-15s | %-15s" % (n, cells[0], cells[1], cells[2]))
    print("  阈值: stat_cov>%.3f, stat_alpha>%.3f, stat_width>%.2f  （均由 none 流 90 分位定，三者 FPR 相同）"
          % (thr[0], thr[1], thr[2]))

    print("\n表 D  ACI 步长 gamma 敏感性（长期覆盖 / 平均宽度）")
    print("%-11s | %-16s | %-16s | %-16s" % ("模式", "g=%.2f" % GAMMA_LIST[0],
                                             "g=%.2f" % GAMMA_LIST[1], "g=%.2f" % GAMMA_LIST[2]))
    print("-" * 108)
    for n in names:
        cs = []
        for g in GAMMA_LIST:
            c = float(np.mean([d[n]["aci"][g]["cov"].mean() for d in per_seed]))
            w = float(np.mean([d[n]["aci"][g]["wid"].mean() for d in per_seed]))
            cs.append("%5.1f%% / %6.1fx" % (c * 100, w))
        print("%-11s | %-16s | %-16s | %-16s" % (n, cs[0], cs[1], cs[2]))

    print("\n表 E  成本与代价（5 种子均值）")
    print("%-11s | %-14s | %-14s | %-14s" % ("模式", "OURS 修复次数", "OURS 额外标签", "RECAL-MON 额外标签"))
    print("-" * 108)
    for n in names:
        nr = float(np.mean([d[n]["n_rep"] for d in per_seed]))
        print("%-11s | %14.1f | %14.0f | %14d" % (n, nr, nr * N_REPAIR, 0))

    # 落盘逐窗口轨迹
    traj = {}
    for n in names:
        traj[n] = dict(
            rho=per_seed[0][n]["rho"].tolist(),
            cov_fro=np.mean([d[n]["cov_fro"] for d in per_seed], axis=0).tolist(),
            cov_aci=np.mean([d[n]["aci"][gm]["cov"] for d in per_seed], axis=0).tolist(),
            cov_rm=np.mean([d[n]["cov_rm"] for d in per_seed], axis=0).tolist(),
            cov_our=np.mean([d[n]["cov_our"] for d in per_seed], axis=0).tolist(),
            wid_fro=float(np.mean([d[n]["wid_fro"] for d in per_seed])),
            wid_aci=np.mean([d[n]["aci"][gm]["wid"] for d in per_seed], axis=0).tolist(),
            wid_rm=np.mean([d[n]["wid_rm"] for d in per_seed], axis=0).tolist(),
            wid_our=np.mean([d[n]["wid_our"] for d in per_seed], axis=0).tolist(),
            alpha_aci=np.mean([d[n]["aci"][gm]["alpha"] for d in per_seed], axis=0).tolist(),
        )
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    np.save(os.path.join(base_dir, "results_aci.npy"), traj, allow_pickle=True)
    print("\n已存 results_aci.npy（逐窗口轨迹：覆盖/宽度/alpha）")

if __name__ == "__main__":
    main()
