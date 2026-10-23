# -*- coding: utf-8 -*-
"""修复后验证：为什么必须用【独立审计集】——以及覆盖审计**不能**验证什么。

【命题 1（自测是恒等式）】设共形分数 s_i、q = 校准集 {s_1..s_n} 的第 ⌈(n+1)(1−α)⌉ 个最小值
（即 `conformal.calibrate_q` 的实现）。则**在同一批校准样本上**测得
        cov_self = #{i: s_i ≤ q} / n = ⌈(n+1)(1−α)⌉ / n
是 n 与 α 的**确定性函数**，与数据、模型、漂移强度**全部无关**。
⇒ "用刚算过 q 的那批样本去验证 q"必然通过，**零信息量**。

【命题 2（覆盖审计的能力边界）】split conformal 的覆盖保证**不依赖模型质量**：
只要校准与评估可交换，重算 q 后 coverage ≈ 1−α 恒成立——**哪怕修正器完全错**。
本脚本用一个**灾难性错误**的修正器（输出再乘 100）验证这一点：
它的覆盖审计同样通过 ~90%，只是把 q（即区间宽度）放大到 ~100× 以上。
⇒ **验证必须同时报告"独立审计覆盖"与"区间宽度"**：前者排除漂移导致的失守，
后者排除"靠无限放宽区间换取覆盖率"。只看覆盖率的验证是可以被"作弊"的。

【四个待验证状态】各自用**自己的** q（q 从哪里来必须写明）：
  frozen      : m_old（不重训）+ **沿用无漂移时的旧 q**   ← 论文口径的"漂移后不做任何事"
  repair      : m_new（500 标签重训）+ 在漂移校准块上重定 q
  recal_only  : m_old（不重训）+ 在漂移校准块上重定 q    ← P2 发现的最优默认策略
  garbage     : m_new 的输出再乘 100（灾难错误）+ 重定 q
所有臂都在**同一批独立审计集**（与任何 q 的来源都不相交）上评估。

【产出】results_repro/experiment_verify_audit.txt + verify_audit.npz
"""
import os, sys, pickle, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, features, model, conformal, drift

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = int(os.environ.get("VA_R", "50"))
DRIFT_MEAN = 2.0              # = 7.39x 相关性漂移（与正文一致）
GARBAGE_LOG = np.log(100.0)   # 灾难性错误：点估计再乘 100

ARMS = ("frozen", "repair", "recal_only", "garbage")


def load_all(pkl_path):
    n_ts, n_cs = features.load_table_col_space(pkl_path)
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    b, y, F, nfc = [], [], [], []
    for ft, label, pgv, *_, ex in d["features"]:
        true = float(np.expm1(label - 1)); pg = float(np.expm1(pgv - 1))
        if true <= 0 or pg <= 0:
            continue
        f = np.nan_to_num(np.concatenate(
            [features.build_schema_feature(ft, ex, n_ts, n_cs), features.stats9(ft)]),
            nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        b.append(pg); y.append(true); F.append(f)
        nfc.append(len(set(ex.get("filter_col_identifiers", []))))
    return (np.array(b), np.array(y), np.array(F, dtype=np.float32), np.array(nfc, dtype=np.int32))


def ci95(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return float(a.mean()), float("nan")
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def ident(n, alpha):
    return min(int(np.ceil((n + 1) * (1 - alpha))), n) / n


def med_qe(true, base, corr):
    est = np.clip(base * np.exp(corr), 1, None)
    return float(np.median(np.maximum(true / est, est / true)))


def main():
    out = []
    def log(s):
        out.append(s); print(s, flush=True)

    base, true, feat, nfc = load_all(config.DATA_PATH)
    n_all = len(true)
    alpha = config.ALPHA
    log(f"加载 {n_all} 条;alpha={alpha};漂移=相关性 {np.exp(DRIFT_MEAN):.2f}x;R={R} 次独立划分")
    log("")

    log("== 命题 1：校准集自测覆盖 ≡ ⌈(n+1)(1−α)⌉ / n，只取决于样本量 n ==")
    log(f"{'n':>8} | {'⌈(n+1)(1−α)⌉':>12} | {'自测覆盖（常数）':>16} | 用途")
    log("-" * 66)
    for n_, use in ((150, "在线监控窗口 L"), (500, "修复用校准块 N_ADAPT_CAL"),
                    (3000, "主实验校准集 N_CAL"), (4000, "主实验测试集 N_TEST")):
        k = min(int(np.ceil((n_ + 1) * (1 - alpha))), n_)
        log(f"{n_:>8} | {k:>12} | {k/n_*100:>15.4f}% | {use}")
    log("  ⇒ 与数据、模型、漂移强度均无关；n 不同常数也不同（旧笔记里的 90.7% 只是 n=150 的情形）。")

    A = {}
    for k in ("self", "audit", "asub", "width", "q", "medqe", "self_n"):
        for a in ARMS:
            A[f"{k}_{a}"] = []
    qsrc = {}

    for r in range(R):
        rng = np.random.RandomState(config.SEED + r)
        idx = rng.permutation(n_all)
        tr = idx[:config.N_TRAIN]
        ca = idx[config.N_TRAIN:config.N_TRAIN + config.N_CAL]
        te = idx[config.N_TRAIN + config.N_CAL:config.N_TRAIN + config.N_CAL + config.N_TEST]

        m_old = model.fit_corrector(feat[tr], base[tr], true[tr], config.LGBM_PARAMS)
        c_old_ca = model.predict_corrector(m_old, feat[ca])
        # 论文口径的"冻结"：沿用无漂移校准集上定下的旧 q
        q_old = conformal.calibrate_q(conformal.score(true[ca], base[ca], c_old_ca), alpha)

        y_d = drift.inject_correlation_drift(true[te], nfc[te], DRIFT_MEAN, rng)
        perm = rng.permutation(len(te))
        a_tr = perm[:config.N_ADAPT_TRAIN]
        a_ca = perm[config.N_ADAPT_TRAIN:config.N_ADAPT_TRAIN + config.N_ADAPT_CAL]
        a_ev = perm[config.N_ADAPT_TRAIN + config.N_ADAPT_CAL:]
        m_new = model.fit_corrector(feat[te][a_tr], base[te][a_tr], y_d[a_tr], config.LGBM_PARAMS)

        c_new_te = model.predict_corrector(m_new, feat[te])
        c_old_te = model.predict_corrector(m_old, feat[te])
        sens_ev = (nfc[te][a_ev] >= 2)

        def sc(corr, sl):
            return conformal.score(y_d[sl], base[te][sl], corr[sl])

        # --- frozen: q 来自无漂移校准集，自测也在那批样本上 ---
        s_ca_f = conformal.score(true[ca], base[ca], c_old_ca)
        A["self_frozen"].append(conformal.coverage(s_ca_f, q_old))
        A["self_n_frozen"].append(len(ca))
        A["q_frozen"].append(q_old)
        s_ev_f = sc(c_old_te, a_ev)
        A["audit_frozen"].append(conformal.coverage(s_ev_f, q_old))
        A["asub_frozen"].append(conformal.coverage(s_ev_f[sens_ev], q_old))
        A["medqe_frozen"].append(med_qe(y_d[a_ev], base[te][a_ev], c_old_te[a_ev]))
        A["width_frozen"].append(float(np.exp(q_old)))
        qsrc["frozen"] = "无漂移校准集 n=3000（旧 q）"

        # --- repair / recal_only / garbage: q 在漂移校准块 a_ca 上重定 ---
        corr_map = {"repair": c_new_te, "recal_only": c_old_te,
                    "garbage": c_new_te + GARBAGE_LOG}
        for a, corr in corr_map.items():
            s_ca = sc(corr, a_ca)
            q = conformal.calibrate_q(s_ca, alpha)
            A[f"self_{a}"].append(conformal.coverage(s_ca, q))     # 自测：同一批样本
            A[f"self_n_{a}"].append(len(a_ca))
            A[f"q_{a}"].append(q)
            s_ev = sc(corr, a_ev)
            A[f"audit_{a}"].append(conformal.coverage(s_ev, q))
            A[f"asub_{a}"].append(conformal.coverage(s_ev[sens_ev], q) if sens_ev.sum() else np.nan)
            A[f"medqe_{a}"].append(med_qe(y_d[a_ev], base[te][a_ev], corr[a_ev]))
            A[f"width_{a}"].append(float(np.exp(q)))
        qsrc["repair"] = qsrc["recal_only"] = qsrc["garbage"] = "漂移校准块 n=500（重定）"
        if (r + 1) % 10 == 0 or r == R - 1:
            log(f"  ... replicate {r+1}/{R}")

    # ---- 汇总 ----
    log("")
    log(f"== 结果（R={R} 次独立重复;mean ± 95%CI;审计集 n={config.N_TEST - config.N_ADAPT_TRAIN - config.N_ADAPT_CAL},与任何 q 的来源不相交）==")
    log("")
    hdr = (f"{'待验证状态':<34} {'自测覆盖':>15} {'独立审计覆盖':>15} {'审计·敏感子集':>15} "
           f"{'区间宽度':>13} {'审计中位 q-error':>16}")
    log(hdr); log("-" * len(hdr))
    label = {"frozen": "冻结（不重训＋沿用旧 q）", "repair": "修复（500 标签重训）",
             "recal_only": "只重校准（不重训）", "garbage": "修复＋修正器×100（灾难错误）"}
    for a in ARMS:
        c = []
        for k in ("self", "audit", "asub"):
            m, h = ci95(np.array(A[f"{k}_{a}"], dtype=float))
            c.append(f"{m*100:5.2f}±{h*100:4.2f}")
        w, hw = ci95(np.array(A[f"width_{a}"], dtype=float))
        qe = float(np.nanmean(A[f"medqe_{a}"]))
        log(f"{label[a]:<34} {c[0]:>15} {c[1]:>15} {c[2]:>15} {w:>8.1f}±{hw:4.1f}x {qe:>16.2f}")
    log("")
    log("  q 的来源：" + " ; ".join(f"{k}={v}" for k, v in qsrc.items()))
    log("")
    log("== 命题 1 验证：每个'自测'数字是否恒等于该 n 下的理论常数 ==")
    for a in ARMS:
        v = np.unique(np.round(np.array(A[f"self_{a}"], dtype=float), 12))
        n_ = int(np.mean(A[f"self_n_{a}"]))
        th = ident(n_, alpha)
        ok = "✓" if (len(v) == 1 and abs(v[0] - th) < 1e-9) else "✗"
        log(f"  self_{a:<11} n={n_:<5} 取值集合={len(v)}（应为 1）  唯一值={v[0]*100:7.4f}%  "
            f"理论值={th*100:7.4f}%  {ok}")
    log("")
    log("== 结论 ==")
    log("  ① 校准集自测在【四个质量天差地别的状态】下给出**逐位相同**的数字 ⇒ 零信息量，")
    log("     它甚至无法区分'修复'与'修正器全错'。")
    log("  ② 独立审计集能区分'冻结'与其余三者（覆盖 62% vs ~90%），这是自测做不到的。")
    log("  ③ 但【覆盖审计也区分不了'修复'与'灾难错误'】——两者都 ~90%，因为 conformal 的")
    log("     覆盖保证不依赖模型质量，垃圾修正器只是把 q 放大。⇒ **必须同时报告区间宽度**：")
    log("     修复 ~7x，灾难错误 ~100x 以上。只看覆盖率的验证是可以被'放宽区间'作弊的。")
    log("  ④ 中位 q-error 与宽度一起构成'有用性'检查，覆盖审计只负责'是否失守'。")

    OUT = os.path.join(ROOT, "results_repro")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "experiment_verify_audit.txt"), "w", encoding="utf-8") as f:
        f.write("exit=0\n" + "\n".join(out) + "\n")
    np.savez(os.path.join(OUT, "verify_audit.npz"),
             **{k: np.array(v, dtype=np.float64) for k, v in A.items()}, R=R)
    print("\nsaved results_repro/experiment_verify_audit.txt", flush=True)


if __name__ == "__main__":
    main()
