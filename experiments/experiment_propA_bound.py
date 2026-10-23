# -*- coding: utf-8 -*-
"""命题 A（支持集溢出界）的验证。

命题：设校准分数集 S（取自漂移【前】分布），线上分数服从漂移【后】分布 Q。
     任何只通过选择分位数水平 alpha' 构造的区间（q 必须取 S 的某个次序统计量），
     其可达覆盖率至多
                sup_{q in S} Q(score <= q) = Q(score <= max S) =: B(Q,S)
     故若 B(Q,S) < 1-alpha，则【不存在任何 alpha 调整能恢复 (1-alpha) 覆盖】，
     必须引入新标签（重校准之所以有效，是因为它的校准集取自 Q 而非 P）。

本脚本计算（同一套切分/漂移/窗口/种子，与 experiment_aci_baseline.py 完全一致）：
  - frozen_t      : 用漂移前 q0 的覆盖（现有基线）
  - aci_t         : ACI(gamma=0.05) 实测覆盖（重算，逐种子）
  - bound_t       : B(Q,S) —— 理论可达上限
  - bestalpha_t   : 在 alpha' 网格上取最大覆盖（应与 bound_t 相等，作为一致性检查）
  - overflow_t    : 1 - B(Q,S) —— 漂移到校准支持集【之外】的分数质量
并统计：overflow > alpha 的窗口占比（= 结构性不可能达标的窗口占比）。
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, data, model, conformal, drift

T_WIN, W_WIN, N_MON, N_EVAL = 20, 500, 150, 300
N_STREAM, N_POOL = T_WIN * W_WIN, 8000
SEEDS = [42, 43, 44, 45, 46]
SPIKE = [6, 13, 18]
GAMMA = 0.05
ALPHA_LO, ALPHA_HI = 0.002, 0.30
ALPHA_GRID = [0.0003, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]

def ru(t): return 2.0
def rp(t): return 1.0 * (1.0 + np.sin(2.0 * np.pi * t / 10.0))
def rt(t): return 2.0 * t / (T_WIN - 1)
def rl(t): return 2.0 if t in SPIKE else 0.2
PATTERNS = [("none", lambda t: 0.0), ("uniform", ru), ("periodic", rp), ("trend", rt), ("long-tail", rl)]

def q_at(S, alpha):
    n = len(S)
    k = min(int(np.ceil((n + 1) * (1.0 - alpha))) - 1, n - 1)
    return float(S[max(k, 0)])

def main():
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    print("命题 A 验证：支持集溢出界 B(Q,S) = Q(score <= max S)")
    print("切分/漂移/窗口/种子与 ACI 实验一致；alpha=%.2f；ACI gamma=%.2f" % (config.ALPHA, GAMMA))
    res = {}
    for seed in SEEDS:
        print("  seed %d ..." % seed); sys.stdout.flush()
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(true))
        itr = idx[:config.N_TRAIN]
        ica = idx[config.N_TRAIN:config.N_TRAIN + config.N_CAL]
        ist = idx[config.N_TRAIN + config.N_CAL:config.N_TRAIN + config.N_CAL + N_STREAM]
        m0 = model.fit_corrector(feat[itr], base[itr], true[itr], config.LGBM_PARAMS)
        S = np.sort(conformal.score(true[ica], base[ica], model.predict_corrector(m0, feat[ica])))
        q0 = q_at(S, config.ALPHA)
        wins = [np.arange(i * W_WIN, (i + 1) * W_WIN) for i in range(T_WIN)]
        ev = [w[W_WIN - N_EVAL:] for w in wins]
        mon = [w[:N_MON] for w in wins]
        for name, fn in PATTERNS:
            rho_t = [float(fn(t)) for t in range(T_WIN)]
            sc_m, sc_e = [], []
            for t in range(T_WIN):
                i_m = ist[mon[t]]; i_e = ist[ev[t]]
                y_m = drift.inject_correlation_drift(true[i_m], nfc[i_m], rho_t[t], rng)
                y_e = drift.inject_correlation_drift(true[i_e], nfc[i_e], rho_t[t], rng)
                sc_m.append(conformal.score(y_m, base[i_m], model.predict_corrector(m0, feat[i_m])))
                sc_e.append(conformal.score(y_e, base[i_e], model.predict_corrector(m0, feat[i_e])))
            smax = float(S.max())
            fro = np.zeros(T_WIN); aci = np.zeros(T_WIN); bound = np.zeros(T_WIN)
            besta = np.zeros(T_WIN); ovf = np.zeros(T_WIN)
            a = config.ALPHA
            for t in range(T_WIN):
                q_t = q_at(S, a)
                aci[t] = float(np.mean(sc_e[t] <= q_t))
                if t == 0:
                    pass
                err = 1.0 - float(np.mean(sc_m[t] <= q_t))
                a = float(np.clip(a + GAMMA * (config.ALPHA - err), ALPHA_LO, ALPHA_HI))
                fro[t] = float(np.mean(sc_e[t] <= q0))
                bound[t] = float(np.mean(sc_e[t] <= smax))
                besta[t] = max(float(np.mean(sc_e[t] <= q_at(S, al))) for al in ALPHA_GRID)
                ovf[t] = 1.0 - bound[t]
            res.setdefault(name, []).append(dict(fro=fro, aci=aci, bound=bound, besta=besta, ovf=ovf, smax=smax))

    print("\n" + "=" * 104)
    print("表 A  命题 A 验证（5 种子；逐窗口再平均）")
    print("%-11s | %8s | %8s | %10s | %11s | %10s | %8s" %
          ("模式", "FROZEN", "ACI", "B(Q,S)上界", "best-alpha", "溢出质量", "ACI/上界"))
    print("-" * 104)
    for name, _ in PATTERNS:
        d = res[name]
        f = np.mean([x["fro"].mean() for x in d])
        a_ = np.mean([x["aci"].mean() for x in d])
        b = np.mean([x["bound"].mean() for x in d])
        ba = np.mean([x["besta"].mean() for x in d])
        o = np.mean([x["ovf"].mean() for x in d])
        print("%-11s | %7.1f%% | %7.1f%% | %9.1f%% | %10.1f%% | %9.1f%% | %7.1f%%"
              % (name, f * 100, a_ * 100, b * 100, ba * 100, o * 100, a_ / max(b, 1e-9) * 100))

    print("\n表 B  结构性判定：溢出质量 > alpha(=%.2f) 的窗口占比" % config.ALPHA)
    print("%-11s | %12s | %14s | %14s" % ("模式", "窗口占比", "平均溢出质量", "结论"))
    print("-" * 104)
    for name, _ in PATTERNS:
        d = res[name]
        frac = np.mean([(x["ovf"] > config.ALPHA).mean() for x in d])
        o = np.mean([x["ovf"].mean() for x in d])
        verdict = "任何alpha调整都不可能达标" if frac > 0.5 else ("部分窗口不可能" if frac > 0 else "原理上可达标")
        print("%-11s | %11.0f%% | %13.1f%% | %14s" % (name, frac * 100, o * 100, verdict))

    print("\n表 C  一致性检查：best-alpha（网格）vs B(Q,S)（解析上界）——两者应相等")
    print("%-11s | %14s | %14s | %10s" % ("模式", "网格最大覆盖", "解析上界", "差"))
    print("-" * 104)
    for name, _ in PATTERNS:
        d = res[name]
        b = np.mean([x["bound"].mean() for x in d])
        ba = np.mean([x["besta"].mean() for x in d])
        print("%-11s | %13.2f%% | %13.2f%% | %9.2fpp" % (name, ba * 100, b * 100, (ba - b) * 100))

    out = {n: {k: np.mean([x[k] for x in v], axis=0).tolist() if hasattr(v[0][k], "__len__") else v[0][k]
               for k in ["fro", "aci", "bound", "besta", "ovf"]} for n, v in res.items()}
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    np.save(os.path.join(base_dir, "results_propA.npy"), out, allow_pickle=True)
    print("\n已存 results_propA.npy")

if __name__ == "__main__":
    main()
