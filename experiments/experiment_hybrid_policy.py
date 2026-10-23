# -*- coding: utf-8 -*-
"""混合策略实验：RECAL-MON（免费）+ 宽度预算触发重训。

要回答的问题
------------
上一轮（experiment_aci_baseline）发现：
  - ACI 守不住覆盖（82.6-87.0% < 90%），因为漂移前校准集存在"信息上界"。
  - RECAL-MON（零标签，每窗口用监控标签重校准 q）能守住覆盖（89.6%），
    但代价是宽度膨胀到 11.7-29.4x。
  - 我们的重训（1000 标签，覆盖报警触发）在 periodic/trend/long-tail 上
    宽度反而更差（30.5/22.9/27.3x），未显示优势。

假设：重训没显示出优势，是因为**触发口径错 + 校准过期**——
  它是"覆盖报警"触发，且重训后只校准一次 q；而 rho(t) 在波动，该 q 立刻过期，
  重训换来的锐度被过期的校准吃掉。

本实验：两级策略
  L1（常开，免费）：每窗口用本窗口监控标签重校准 q  -> 任何时刻覆盖都有效
  L2（条件，花标签）：监测宽度 w_t；仅当 w_t > W* 才重训修正器（500 训 + 500 校准），
                     且重训后【继续每窗口重校准】（关键修正）

对照臂
  RECAL_MON      : W* = inf（永不重训）
  HYBRID(W*)     : W* ∈ {8, 12, 18, 30}
  ALARM          : 现有方法（覆盖报警触发重训 + 一次性校准）作为参照

关键诊断
  sharp_gain = w_after / w_before（重训事件上取均值）—— 重训是否真的买到锐度？
  w_before = 用【旧】修正器 + 本窗口监控标签重校准得到的宽度
  w_after  = 用【新】修正器 + 本窗口监控标签重校准得到的宽度
  （两者用同一批监控标签，故只比较相对变化，不构成乐观偏差来源）

切分/漂移/窗口/种子完全沿用 experiment_temporaldrift.py 与 experiment_aci_baseline.py。
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, data, model, conformal, drift

T_WIN, W_WIN, N_MON, N_EVAL = 20, 500, 150, 300
TAU, COOLDOWN, N_REPAIR = 0.85, 2, 1000
N_STREAM = T_WIN * W_WIN
N_POOL = 8000
Z = 1.645
SEEDS = [42, 43, 44, 45, 46]
WSTAR_LIST = [8.0, 12.0, 18.0, 30.0]
SPIKE = [6, 13, 18]

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
    mon = [w[:N_MON] for w in wins]
    ev = [w[W_WIN - N_EVAL:] for w in wins]

    out = {}
    for name, fn in PATTERNS:
        rho_t = np.array([float(fn(t)) for t in range(T_WIN)], dtype=float)
        y_m = [drift.inject_correlation_drift(y_st[mon[t]], nfc_st[mon[t]], rho_t[t], rng) for t in range(T_WIN)]
        y_e = [drift.inject_correlation_drift(y_st[ev[t]], nfc_st[ev[t]], rho_t[t], rng) for t in range(T_WIN)]
        f_m = [f_st[mon[t]] for t in range(T_WIN)]
        b_m = [b_st[mon[t]] for t in range(T_WIN)]
        f_e = [f_st[ev[t]] for t in range(T_WIN)]
        b_e = [b_st[ev[t]] for t in range(T_WIN)]

        # ---------- FROZEN（参考） ----------
        cov_fro = np.array([conformal.coverage(
            conformal.score(y_e[t], b_e[t], model.predict_corrector(m0, f_e[t])), q0) for t in range(T_WIN)])

        # ---------- 两级策略：W* = inf(=RECAL_MON) 与若干预算 ----------
        policy = {}
        for Wstar in [np.inf] + WSTAR_LIST:
            m_cur, cursor, last_repair, n_rep = m0, 0, -99, 0
            cov = np.zeros(T_WIN); wid = np.zeros(T_WIN)
            gain = []          # w_after / w_before（仅重训窗口）
            cov_after = []     # 重训窗口上的覆盖
            for t in range(T_WIN):
                # L1：免费重校准（用当前修正器 + 本窗口监控标签）
                q_t = conformal.calibrate_q(
                    conformal.score(y_m[t], b_m[t], model.predict_corrector(m_cur, f_m[t])), config.ALPHA)
                w_before = float(np.exp(q_t))
                # L2：宽度预算触发重训
                if (w_before > Wstar) and (t - last_repair) > COOLDOWN and (cursor + N_REPAIR) <= N_POOL:
                    s = slice(cursor, cursor + N_REPAIR); cursor += N_REPAIR
                    y_a = drift.inject_correlation_drift(y_ad[s], nfc_ad[s], rho_t[t], rng)
                    m_new = model.fit_corrector(f_ad[s][:500], b_ad[s][:500], y_a[:500], config.LGBM_PARAMS)
                    # 关键修正：重训后【继续用本窗口监控标签重校准】
                    q_new = conformal.calibrate_q(
                        conformal.score(y_m[t], b_m[t], model.predict_corrector(m_new, f_m[t])), config.ALPHA)
                    gain.append(float(np.exp(q_new)) / max(w_before, 1e-9))
                    m_cur, q_t = m_new, q_new
                    n_rep += 1; last_repair = t
                wid[t] = float(np.exp(q_t))
                cov[t] = conformal.coverage(
                    conformal.score(y_e[t], b_e[t], model.predict_corrector(m_cur, f_e[t])), q_t)
                if gain and last_repair == t:
                    cov_after.append(cov[t])
            policy[Wstar] = dict(cov=cov, wid=wid, n_rep=n_rep, labels=n_rep * N_REPAIR,
                                 gain=float(np.mean(gain)) if gain else float("nan"),
                                 n_gain=len(gain),
                                 cov_after=float(np.mean(cov_after)) if cov_after else float("nan"))

        # ---------- ALARM（现有方法：覆盖报警触发 + 一次性校准） ----------
        m_cur, q_cur, cursor, last_repair, n_rep = m0, q0, 0, -99, 0
        cov_a, wid_a = np.zeros(T_WIN), np.zeros(T_WIN)
        for t in range(T_WIN):
            ph = conformal.coverage(
                conformal.score(y_m[t], b_m[t], model.predict_corrector(m_cur, f_m[t])), q_cur)
            if alarm_decision(ph, N_MON, target) and (t - last_repair) > COOLDOWN and (cursor + N_REPAIR) <= N_POOL:
                s = slice(cursor, cursor + N_REPAIR); cursor += N_REPAIR
                y_a = drift.inject_correlation_drift(y_ad[s], nfc_ad[s], rho_t[t], rng)
                m_cur = model.fit_corrector(f_ad[s][:500], b_ad[s][:500], y_a[:500], config.LGBM_PARAMS)
                q_cur = conformal.calibrate_q(
                    conformal.score(y_a[500:], b_ad[s][500:],
                                    model.predict_corrector(m_cur, f_ad[s][500:])), config.ALPHA)
                n_rep += 1; last_repair = t
            wid_a[t] = float(np.exp(q_cur))
            cov_a[t] = conformal.coverage(
                conformal.score(y_e[t], b_e[t], model.predict_corrector(m_cur, f_e[t])), q_cur)

        out[name] = dict(cov_fro=cov_fro, policy=policy,
                         cov_alarm=cov_a, wid_alarm=wid_a, n_rep_alarm=n_rep, wid_fro=float(np.exp(q0)))
    return out

def main():
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    print("混合策略实验（RECAL-MON + 宽度预算触发重训）：%d 条；种子 %s" % (len(true), SEEDS))
    print("窗口 %d x %d（监控 %d / 评估 %d）；W* ∈ %s（inf = 永不重训）；冷却 %d；每次重训 %d 标签"
          % (T_WIN, W_WIN, N_MON, N_EVAL, WSTAR_LIST, COOLDOWN, N_REPAIR))

    per_seed = []
    for s in SEEDS:
        print("  seed %d ..." % s); sys.stdout.flush()
        per_seed.append(run_one_seed(s, base, true, feat, nfc))

    names = [n for n, _ in PATTERNS]

    def mm(pattern, key):
        v = np.array([d[pattern][key] for d in per_seed], dtype=float)
        return float(v.mean()), float(v.std())

    print("\n" + "=" * 112)
    print("表 A  策略前沿：覆盖 / 宽度 / 标签成本（5 种子均值）")
    hdr = "%-11s | %-15s |" % ("模式", "ALARM(现有)") + "".join(
        " %-13s |" % ("RECAL_MON" if W == np.inf else "W*<=%.0fx" % W) for W in [np.inf] + WSTAR_LIST)
    print(hdr)
    print("-" * 112)
    for n in names:
        cells = []
        ca, _ = mm(n, "cov_alarm")
        wa = float(np.mean([d[n]["wid_alarm"].mean() for d in per_seed]))
        nr = float(np.mean([d[n]["n_rep_alarm"] for d in per_seed]))
        cells.append("%5.1f%%/%5.1fx/%4.0f" % (ca * 100, wa, nr * N_REPAIR))
        for W in [np.inf] + WSTAR_LIST:
            c = float(np.mean([d[n]["policy"][W]["cov"].mean() for d in per_seed]))
            w = float(np.mean([d[n]["policy"][W]["wid"].mean() for d in per_seed]))
            lb = float(np.mean([d[n]["policy"][W]["labels"] for d in per_seed]))
            cells.append("%5.1f%%/%5.1fx/%4.0f" % (c * 100, w, lb))
        line = "%-11s |" % n + "".join(" %-15s |" % c for c in cells)
        print(line)
    print("  格式：覆盖 / 平均宽度 / 标签开销（5 种子均值）。FROZEN 参考：", end="")
    print(" ".join("%s=%.1f%%" % (n, float(np.mean([d[n]["cov_fro"].mean() for d in per_seed])) * 100) for n in names))

    print("\n" + "=" * 112)
    print("表 B  关键诊断：重训是否买到【锐度】？(sharp_gain = w_after/w_before，<1 表示重训后同覆盖下更窄)")
    print("%-11s |" % "模式" + "".join(" %-13s |" % ("W*=%.0fx" % W) for W in WSTAR_LIST))
    print("-" * 112)
    for n in names:
        cells = []
        for W in WSTAR_LIST:
            g = [d[n]["policy"][W]["gain"] for d in per_seed if d[n]["policy"][W]["n_gain"] > 0]
            k = sum(d[n]["policy"][W]["n_gain"] for d in per_seed)
            cells.append(("%.2f (n=%d)" % (float(np.mean(g)), k)) if g else "-- (n=0)")
        print("%-11s |" % n + "".join(" %-13s |" % c for c in cells))

    print("\n" + "=" * 112)
    print("表 C  重训窗口上的即时覆盖（cov_after）——重训当下是否立刻恢复有效")
    print("%-11s |" % "模式" + "".join(" %-13s |" % ("W*=%.0fx" % W) for W in WSTAR_LIST))
    print("-" * 112)
    for n in names:
        cells = []
        for W in WSTAR_LIST:
            v = [d[n]["policy"][W]["cov_after"] for d in per_seed if not np.isnan(d[n]["policy"][W]["cov_after"])]
            cells.append("%.1f%%" % (float(np.mean(v)) * 100) if v else "--")
        print("%-11s |" % n + "".join(" %-13s |" % c for c in cells))

    # 落盘
    traj = {}
    for n in names:
        traj[n] = {"rho": None,
                   "cov_fro": np.mean([d[n]["cov_fro"] for d in per_seed], axis=0).tolist(),
                   "cov_alarm": np.mean([d[n]["cov_alarm"] for d in per_seed], axis=0).tolist(),
                   "wid_alarm": np.mean([d[n]["wid_alarm"] for d in per_seed], axis=0).tolist()}
        for W in [np.inf] + WSTAR_LIST:
            k = "inf" if W == np.inf else "%.0f" % W
            traj[n]["cov_W" + k] = np.mean([d[n]["policy"][W]["cov"] for d in per_seed], axis=0).tolist()
            traj[n]["wid_W" + k] = np.mean([d[n]["policy"][W]["wid"] for d in per_seed], axis=0).tolist()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    np.save(os.path.join(base_dir, "results_hybrid.npy"), traj, allow_pickle=True)
    print("\n已存 results_hybrid.npy（各策略逐窗口覆盖/宽度轨迹）")

if __name__ == "__main__":
    main()
