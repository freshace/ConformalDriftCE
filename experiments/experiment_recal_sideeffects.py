# -*- coding: utf-8 -*-
"""RECAL-MON 的副作用验证 + 重训的正确用法（三项一次验证）。

背景结论（前两个实验）
  A. ACI 救不回覆盖（82.6-87.0% < 90%）——漂移前校准集存在信息上界。
  B. RECAL-MON（零额外标签，每窗口用监控标签重校准 q）能守覆盖（89.6%），
     但宽度 8.2x -> 29.4x；且它【抹掉了覆盖信号】。
  C. 上一轮"重训被支配"的结论有 confound：触发用【绝对宽度门槛】，
     且【无条件采纳】新修正器（gain>1 也收）-> 重训级联。

本实验要回答三个问题
  Q1 加【接受门槛】(q_new < q_old 才采纳) + 【漂移相对触发】(w_t > theta*w_ref)
     + 【重参考】(采纳后更新 w_ref)，重训级联是否消失？重训能否真正买到锐度？
  Q2 把【点估计指标】(GMQ / q-error 分位) 纳入评估——重训真正的收益是否在这里？
     （RECAL-MON 只搬分位数，完全不改点估计，所以它在这张表上应当毫无作用。）
  Q3 RECAL-MON 的两个前提有多脆？
     (a) 标签延迟 1 窗口（q 滞后）；(b) 有偏监控（只标注"廉价"的单列查询）。

臂（同一切分 / 同一漂移 / 同一 20x500 窗口 / 同一 5 种子 / 监控 150 / 评估 300）
  FROZEN           : 冻结 m0 + 固定 q0（参照）
  RECAL_MON        : 每窗口用随机监控标签重校准 q；不重训
  RECAL_DELAY1     : 同上，但用【上一窗口】的监控标签（q 滞后一窗）
  RECAL_BIAS       : 同上，但监控标签只取 nfc<=1 的查询（有偏标注）
  ALARM            : 现有方法（覆盖报警触发 + 无条件重训 + 一次性校准）
  GUARD_ABS(12x)   : 绝对宽度门槛 + 接受门槛
  GUARD_REL(2.0x)  : 漂移相对门槛（w_ref 参照）+ 接受门槛 + 采纳后重参考

关键量
  w_t = exp(q_t)（当前覆盖水平下的区间宽度）
  q-error = exp(|log y - log yhat|)；GMQ = exp(mean|log y - log yhat|)
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
SPIKE = [6, 13, 18]
W_ABS = 12.0          # GUARD_ABS 的绝对门槛
THETA = 2.0           # GUARD_REL 的相对门槛（相对 w_ref）

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

def pt_metrics(y, base, corr):
    """点估计指标：GMQ 与 q-error 分位（均基于 yhat = base*exp(corr)）。"""
    s = np.abs(np.log(np.clip(y, 1.0, None)) - np.log(np.clip(base * np.exp(corr), 1.0, None)))
    return (float(np.exp(s.mean())),
            float(np.exp(np.percentile(s, 50))),
            float(np.exp(np.percentile(s, 90))),
            float(np.exp(np.percentile(s, 99))))

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
    w0 = float(np.exp(q0))

    wins = [np.arange(i * W_WIN, (i + 1) * W_WIN) for i in range(T_WIN)]
    ev = [w[W_WIN - N_EVAL:] for w in wins]
    # 监控集：随机（前 N_MON）/ 有偏（nfc<=1 的前 N_MON 条）
    mon_rnd, mon_bias = [], []
    for w in wins:
        mon_rnd.append(w[:N_MON])
        cheap = np.array([i for i in w if nfc_st[i] <= 1], dtype=int)
        mon_bias.append(cheap[:N_MON] if len(cheap) >= N_MON else cheap)

    out = {}
    for name, fn in PATTERNS:
        rho_t = np.array([float(fn(t)) for t in range(T_WIN)], dtype=float)
        # 漂移后的标签：监控（随机/有偏，同一 rho）/ 评估 / 修复池
        y_mr = [drift.inject_correlation_drift(y_st[mon_rnd[t]], nfc_st[mon_rnd[t]], rho_t[t], rng)
                for t in range(T_WIN)]
        y_mb = [drift.inject_correlation_drift(y_st[mon_bias[t]], nfc_st[mon_bias[t]], rho_t[t], rng)
                for t in range(T_WIN)]
        y_e = [drift.inject_correlation_drift(y_st[ev[t]], nfc_st[ev[t]], rho_t[t], rng)
               for t in range(T_WIN)]
        f_mr = [f_st[mon_rnd[t]] for t in range(T_WIN)]
        b_mr = [b_st[mon_rnd[t]] for t in range(T_WIN)]
        f_mb = [f_st[mon_bias[t]] for t in range(T_WIN)]
        b_mb = [b_st[mon_bias[t]] for t in range(T_WIN)]
        f_e = [f_st[ev[t]] for t in range(T_WIN)]
        b_e = [b_st[ev[t]] for t in range(T_WIN)]

        corr_mr0 = [model.predict_corrector(m0, f_mr[t]) for t in range(T_WIN)]
        corr_mb0 = [model.predict_corrector(m0, f_mb[t]) for t in range(T_WIN)]
        corr_e0 = [model.predict_corrector(m0, f_e[t]) for t in range(T_WIN)]

        # ---------- FROZEN ----------
        cov_fro = np.array([conformal.coverage(conformal.score(y_e[t], b_e[t], corr_e0[t]), q0)
                            for t in range(T_WIN)])
        pt_fro = [pt_metrics(y_e[t], b_e[t], corr_e0[t]) for t in range(T_WIN)]

        # ---------- RECAL_MON / DELAY1 / BIAS（纯重校准，不重训） ----------
        def recall_arm(mon_y, mon_f, mon_b, corr_mon, delay=0):
            cov = np.zeros(T_WIN); wid = np.zeros(T_WIN)
            q_prev, q_cur = q0, q0
            for t in range(T_WIN):
                if delay == 1:
                    # 用上一窗口的监控标签定 q（t=0 无上一窗 -> 沿用 q0）
                    if t == 0:
                        q_t = q0
                    else:
                        q_t = conformal.calibrate_q(
                            conformal.score(mon_y[t - 1], mon_b[t - 1], corr_mon[t - 1]), config.ALPHA)
                else:
                    q_t = conformal.calibrate_q(
                        conformal.score(mon_y[t], mon_b[t], corr_mon[t]), config.ALPHA)
                wid[t] = float(np.exp(q_t))
                cov[t] = conformal.coverage(conformal.score(y_e[t], b_e[t], corr_e0[t]), q_t)
            return cov, wid

        cov_rm, wid_rm = recall_arm(y_mr, f_mr, b_mr, corr_mr0, delay=0)
        cov_dl, wid_dl = recall_arm(y_mr, f_mr, b_mr, corr_mr0, delay=1)
        cov_bs, wid_bs = recall_arm(y_mb, f_mb, b_mb, corr_mb0, delay=0)

        # ---------- 重训类：ALARM / GUARD_ABS / GUARD_REL ----------
        def retrain_arm(mode):
            m_cur, q_cur, cursor, last, n_acc, n_rej = m0, q0, 0, -99, 0, 0
            w_ref = w0
            cov = np.zeros(T_WIN); wid = np.zeros(T_WIN)
            for t in range(T_WIN):
                corr_m = model.predict_corrector(m_cur, f_mr[t])
                # L1：免费重校准（始终进行）
                q_t = conformal.calibrate_q(conformal.score(y_mr[t], b_mr[t], corr_m), config.ALPHA)
                w_t = float(np.exp(q_t))
                fired = False
                if mode == "alarm":
                    ph = conformal.coverage(conformal.score(y_mr[t], b_mr[t], corr_m), q_cur)
                    fired = alarm_decision(ph, N_MON, target)
                elif mode == "abs":
                    fired = w_t > W_ABS
                elif mode == "rel":
                    fired = w_t > THETA * w_ref
                if fired and (t - last) > COOLDOWN and (cursor + N_REPAIR) <= N_POOL:
                    s = slice(cursor, cursor + N_REPAIR); cursor += N_REPAIR
                    y_a = drift.inject_correlation_drift(y_ad[s], nfc_ad[s], rho_t[t], rng)
                    m_new = model.fit_corrector(f_ad[s][:500], b_ad[s][:500], y_a[:500], config.LGBM_PARAMS)
                    if mode == "alarm":
                        # 现有方法：用修复池的后 500 条校准（一次性，不重校准）
                        q_new = conformal.calibrate_q(
                            conformal.score(y_a[500:], b_ad[s][500:],
                                            model.predict_corrector(m_new, f_ad[s][500:])), config.ALPHA)
                        m_cur, q_cur = m_new, q_new
                        q_t, w_t = q_new, float(np.exp(q_new))
                        n_acc += 1; last = t
                    else:
                        # 接受门槛：用【同一批监控标签】比较新旧模型
                        corr_new = model.predict_corrector(m_new, f_mr[t])
                        q_cand = conformal.calibrate_q(
                            conformal.score(y_mr[t], b_mr[t], corr_new), config.ALPHA)
                        if q_cand < q_t:
                            m_cur, q_cur = m_new, q_cand
                            q_t, w_t = q_cand, float(np.exp(q_cand))
                            n_acc += 1; last = t
                            if mode == "rel":
                                w_ref = w_t          # 采纳后重参考
                        else:
                            n_rej += 1; last = t     # 拒绝也消耗标签
                wid[t] = w_t
                corr_now = model.predict_corrector(m_cur, f_e[t])
                cov[t] = conformal.coverage(conformal.score(y_e[t], b_e[t], corr_now), q_t)
            return cov, wid, n_acc, n_rej, (n_acc + n_rej) * N_REPAIR

        cov_al, wid_al, acc_al, rej_al, lab_al = retrain_arm("alarm")
        cov_ab, wid_ab, acc_ab, rej_ab, lab_ab = retrain_arm("abs")
        cov_rl, wid_rl, acc_rl, rej_rl, lab_rl = retrain_arm("rel")

        # ---------- 点估计表：只跟【最终模型】有关 ----------
        # RECAL_MON 不改模型 -> 与 FROZEN 完全相同；取最后一窗口实际使用的修正器
        def final_pt(mode):
            # 重跑一遍只为拿到逐窗口点估计（成本低：仅前向）
            m_cur, cursor, last = m0, 0, -99
            gmq = np.zeros(T_WIN); p50 = np.zeros(T_WIN); p90 = np.zeros(T_WIN); p99 = np.zeros(T_WIN)
            w_ref = w0
            for t in range(T_WIN):
                corr_m = model.predict_corrector(m_cur, f_mr[t])
                q_t = conformal.calibrate_q(conformal.score(y_mr[t], b_mr[t], corr_m), config.ALPHA)
                w_t = float(np.exp(q_t))
                fired = (w_t > W_ABS) if mode == "abs" else (w_t > THETA * w_ref)
                if fired and (t - last) > COOLDOWN and (cursor + N_REPAIR) <= N_POOL:
                    s = slice(cursor, cursor + N_REPAIR); cursor += N_REPAIR
                    y_a = drift.inject_correlation_drift(y_ad[s], nfc_ad[s], rho_t[t], rng)
                    m_new = model.fit_corrector(f_ad[s][:500], b_ad[s][:500], y_a[:500], config.LGBM_PARAMS)
                    corr_new = model.predict_corrector(m_new, f_mr[t])
                    q_cand = conformal.calibrate_q(conformal.score(y_mr[t], b_mr[t], corr_new), config.ALPHA)
                    if q_cand < q_t:
                        m_cur, last = m_new, t
                        if mode == "rel":
                            w_ref = float(np.exp(q_cand))
                    else:
                        last = t
                corr_e = model.predict_corrector(m_cur, f_e[t])
                gmq[t], p50[t], p90[t], p99[t] = pt_metrics(y_e[t], b_e[t], corr_e)
            return gmq, p50, p90, p99

        gmq_ab, p50_ab, p90_ab, p99_ab = final_pt("abs")
        gmq_rl, p50_rl, p90_rl, p99_rl = final_pt("rel")

        out[name] = dict(
            rho=rho_t, w0=w0,
            cov_fro=cov_fro, wid_fro=w0,
            pt_fro=np.array(pt_fro).T,   # 与 pt_ab/pt_rl 同为 (4, T_WIN)
            cov_rm=cov_rm, wid_rm=wid_rm,
            cov_dl=cov_dl, wid_dl=wid_dl,
            cov_bs=cov_bs, wid_bs=wid_bs,
            cov_al=cov_al, wid_al=wid_al, lab_al=lab_al, acc_al=acc_al, rej_al=rej_al,
            cov_ab=cov_ab, wid_ab=wid_ab, lab_ab=lab_ab, acc_ab=acc_ab, rej_ab=rej_ab,
            cov_rl=cov_rl, wid_rl=wid_rl, lab_rl=lab_rl, acc_rl=acc_rl, rej_rl=rej_rl,
            pt_ab=np.vstack([gmq_ab, p50_ab, p90_ab, p99_ab]),
            pt_rl=np.vstack([gmq_rl, p50_rl, p90_rl, p99_rl]),
        )
    return out

def main():
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    print("副作用验证 + 重训正确用法：%d 条；种子 %s" % (len(true), SEEDS))
    print("窗口 %d x %d（监控 %d / 评估 %d）；COOLDOWN=%d；每次重训 %d 标签；GUARD_ABS 门槛 %.0fx；GUARD_REL theta=%.1f"
          % (T_WIN, W_WIN, N_MON, N_EVAL, COOLDOWN, N_REPAIR, W_ABS, THETA))

    per_seed = []
    for s in SEEDS:
        print("  seed %d ..." % s); sys.stdout.flush()
        per_seed.append(run_one_seed(s, base, true, feat, nfc))

    names = [n for n, _ in PATTERNS]

    def agg(pattern, key):
        v = np.array([d[pattern][key] for d in per_seed], dtype=float)
        return float(v.mean())

    def wid_mean(pattern, key):
        return float(np.mean([d[pattern][key].mean() for d in per_seed]))

    print("\n" + "=" * 118)
    print("表 A  副作用与重训用法：覆盖 / 平均宽度 / 标签开销（5 种子均值）")
    print("%-11s | %-13s | %-13s | %-13s | %-13s | %-15s | %-15s" %
          ("模式", "FROZEN", "RECAL_MON", "DELAY1", "BIAS(nfc<=1)", "GUARD_ABS(12x)", "GUARD_REL(2x)"))
    print("-" * 118)
    for n in names:
        c0 = agg(n, "cov_fro")
        c1, w1 = agg(n, "cov_rm"), wid_mean(n, "wid_rm")
        c2, w2 = agg(n, "cov_dl"), wid_mean(n, "wid_dl")
        c3, w3 = agg(n, "cov_bs"), wid_mean(n, "wid_bs")
        c4, w4 = agg(n, "cov_ab"), wid_mean(n, "wid_ab")
        c5, w5 = agg(n, "cov_rl"), wid_mean(n, "wid_rl")
        print("%-11s | %5.1f%%/%5.1fx | %5.1f%%/%5.1fx | %5.1f%%/%5.1fx | %5.1f%%/%5.1fx | %5.1f%%/%5.1fx | %5.1f%%/%5.1fx"
              % (n, c0 * 100, per_seed[0][n]["wid_fro"], c1 * 100, w1, c2 * 100, w2,
                 c3 * 100, w3, c4 * 100, w4, c5 * 100, w5))

    print("\n表 B  标签开销与接受/拒绝（5 种子均值）——拒绝也消耗标签")
    print("%-11s | %-26s | %-26s | %-26s" % ("模式", "ALARM(现有)", "GUARD_ABS(12x)", "GUARD_REL(2x)"))
    print("%-11s | %-26s | %-26s | %-26s" % ("", "标签 / 接受 / 拒绝", "标签 / 接受 / 拒绝", "标签 / 接受 / 拒绝"))
    print("-" * 118)
    for n in names:
        cells = []
        for k in ["al", "ab", "rl"]:
            cells.append("%6.0f / %4.1f / %4.1f" % (agg(n, "lab_" + k), agg(n, "acc_" + k), agg(n, "rej_" + k)))
        print("%-11s | %-26s | %-26s | %-26s" % (n, cells[0], cells[1], cells[2]))

    print("\n" + "=" * 118)
    print("表 C  ★点估计指标（GMQ / q-error p50 / p90 / p99）——RECAL_MON 不改点估计，故与 FROZEN 相同")
    print("%-11s | %-24s | %-24s | %-24s" % ("模式", "FROZEN = RECAL_MON", "GUARD_ABS(12x)", "GUARD_REL(2x)"))
    print("-" * 118)
    for n in names:
        pf = np.mean([d[n]["pt_fro"] for d in per_seed], axis=0)
        pa = np.mean([d[n]["pt_ab"] for d in per_seed], axis=0)
        pr = np.mean([d[n]["pt_rl"] for d in per_seed], axis=0)

        def fmt(x):
            return "%6.1f /%6.1f /%7.1f /%9.1f" % (x[0].mean(), x[1].mean(), x[2].mean(), x[3].mean())
        print("%-11s | %-24s | %-24s | %-24s" % (n, fmt(pf), fmt(pa), fmt(pr)))
    print("  注：GMQ 与 q-error 分位为逐窗口值再平均；数值越小越好。")

    print("\n" + "=" * 118)
    print("表 D  RECAL_MON 前提的脆弱性：覆盖 / 宽度（相对无延迟、无偏基准的变化）")
    print("%-11s | %-20s | %-20s | %-20s" % ("模式", "RECAL_MON", "DELAY1 (q 滞后)", "BIAS (只标廉价查询)"))
    print("-" * 118)
    for n in names:
        c1, w1 = agg(n, "cov_rm"), wid_mean(n, "wid_rm")
        c2, w2 = agg(n, "cov_dl"), wid_mean(n, "wid_dl")
        c3, w3 = agg(n, "cov_bs"), wid_mean(n, "wid_bs")
        print("%-11s | %6.1f%% / %6.1fx | %6.1f%% / %6.1fx (%+5.1fpp) | %6.1f%% / %6.1fx (%+5.1fpp)"
              % (n, c1 * 100, w1, c2 * 100, w2, (c2 - c1) * 100, c3 * 100, w3, (c3 - c1) * 100))

    traj = {}
    for n in names:
        traj[n] = {k: (np.mean([d[n][k] for d in per_seed], axis=0).tolist()
                       if hasattr(per_seed[0][n][k], "__len__") else per_seed[0][n][k])
                   for k in ["cov_fro", "cov_rm", "wid_rm", "cov_dl", "wid_dl", "cov_bs", "wid_bs",
                             "cov_al", "wid_al", "cov_ab", "wid_ab", "cov_rl", "wid_rl"]}
        traj[n]["w0"] = per_seed[0][n]["w0"]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    np.save(os.path.join(base_dir, "results_sideeffects.npy"), traj, allow_pickle=True)
    print("\n已存 results_sideeffects.npy（各臂逐窗口覆盖/宽度轨迹）")

if __name__ == "__main__":
    main()
