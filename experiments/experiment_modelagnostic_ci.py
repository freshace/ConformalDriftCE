# -*- coding: utf-8 -*-
"""§4.9 通用性(跨底模)的重复切分版（P4/P1 替代）：给 PG/ALECE/PRICE 三底模的
"无漂移 / 漂移后冻结 / 修复后" 补 95% CI。

【为什么是"P1 的替代"】P1 原计划把跨底模从 3 个扩到 5 个(+MSCN/DeepDB)。
经全仓勘察（.tools/_p1_probe*.py），这条**从现有产物不可行**：
  - AdaCE 备份(results_backup_15dim_20260909)是**另一条线**：15 维特征、2749 条分区，
    且 meta 是特征矩阵而非索引 ⇒ 与 DriftCE 的 50000 条 622 维工作负载**无法对齐**
    （用 (真值, PG估计) 配对匹配：命中 0 条）；
  - ETGE/results_expert_oracle 只含 PG / SchemaBase / LGBM / ALECE / Neural / PRICE 六个，
    **没有 MSCN / DeepDB / NeuroCard**；
  - 该 oracle npz 又是另一个 10000 条分区。
⇒ 要加 MSCN/DeepDB 必须**在 stats 工作负载上真跑那两个估计器**（工程量大且 model-agnostic
   已非本文新颖性，见 ICDE'22）。故本脚本改为把现有三底模结论**补上 CI**——这同样落在 P4 范围内。

【复现门】口径与 `experiment_modelagnostic.py` 同源（复用其 load_bases；同一 rng 顺序），
第 1 次重复(seed=42)应复现其单次读数。

【产出】results_repro/experiment_modelagnostic_ci.txt + modelagnostic_ci.npz
"""
import os, sys, importlib.util, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import config, model, conformal

_spec = importlib.util.spec_from_file_location("ma", os.path.join(HERE, "experiment_modelagnostic.py"))
ma = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ma)

ROOT = os.path.dirname(HERE)
R = int(os.environ.get("MA_R", "50"))


def ci95(a):
    a = np.asarray(a, dtype=np.float64); a = a[np.isfinite(a)]
    if len(a) < 2:
        return float(a.mean()), float("nan")
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def main():
    out = []
    def log(s):
        out.append(s); print(s, flush=True)

    base, true, sf, nfc, idx = ma.load_bases(config.DATA_PATH)
    alece = np.load(os.path.join(ma.ORACLE_DIR, "alece_stats.npy"))[idx]
    price = np.load(os.path.join(ma.ORACLE_DIR, "price_stats.npy"))[idx]
    log(f"加载 {len(true)} 条；R={R} 次独立重复划分（三底模共用同一划分，配对比较）")
    log("口径与 experiment_modelagnostic.py 同源 ⇒ 第 1 次重复(seed=42)应复现其单次读数")

    BASES = [("PG", base), ("ALECE", alece), ("PRICE", price)]
    acc = {n: {"nodrift": [], "frozen": [], "fixed": [], "wq": [], "w0": []} for n, _ in BASES}

    for r in range(R):
        # 原脚本在 wrap_exp **内部**新建 RandomState(SEED)，故 PG/ALECE/PRICE 三个底模
        # 共用同一划分**与同一适应置换**（配对比较）。这里对每个底模各新建一份
        # RandomState(SEED+r)，才能与单次读数逐位对齐——若共用一个 rng，
        # 后两个底模的置换会被前一个推进，复现门就会失败（实测确实如此）。
        n = len(true)
        idxp = np.random.RandomState(config.SEED + r).permutation(n)
        tr = idxp[:config.N_TRAIN]
        cal = idxp[config.N_TRAIN:config.N_TRAIN + config.N_CAL]
        te = idxp[config.N_TRAIN + config.N_CAL:config.N_TRAIN + config.N_CAL + config.N_TEST]
        for name, bp in BASES:
            rng = np.random.RandomState(config.SEED + r)
            rng.permutation(n)          # 与原脚本一致：先消耗掉划分那次抽样
            F = np.hstack([sf, np.log(np.clip(bp, 1, None))[:, None]]).astype(np.float32)
            m_old = model.fit_corrector(F[tr], bp[tr], true[tr], config.LGBM_PARAMS)
            q_old = conformal.calibrate_q(
                conformal.score(true[cal], bp[cal], model.predict_corrector(m_old, F[cal])), config.ALPHA)
            acc[name]["nodrift"].append(conformal.coverage(
                conformal.score(true[te], bp[te], model.predict_corrector(m_old, F[te])), q_old))
            dt = true[te] * np.where(nfc[te] >= 2, np.exp(2.0), 1.0)
            acc[name]["frozen"].append(conformal.coverage(
                conformal.score(dt, bp[te], model.predict_corrector(m_old, F[te])), q_old))
            acc[name]["w0"].append(float(np.exp(q_old)))
            perm = rng.permutation(len(te))
            a_tr = te[perm[:config.N_ADAPT_TRAIN]]
            a_cal = te[perm[config.N_ADAPT_TRAIN:config.N_ADAPT_TRAIN + config.N_ADAPT_CAL]]
            a_ev = te[perm[config.N_ADAPT_TRAIN + config.N_ADAPT_CAL:]]
            m_new = model.fit_corrector(F[a_tr], bp[a_tr], dt[perm[:config.N_ADAPT_TRAIN]], config.LGBM_PARAMS)
            q_new = conformal.calibrate_q(
                conformal.score(dt[perm[config.N_ADAPT_TRAIN:config.N_ADAPT_TRAIN + config.N_ADAPT_CAL]],
                                bp[a_cal], model.predict_corrector(m_new, F[a_cal])), config.ALPHA)
            acc[name]["fixed"].append(conformal.coverage(
                conformal.score(dt[perm[config.N_ADAPT_TRAIN + config.N_ADAPT_CAL:]],
                                bp[a_ev], model.predict_corrector(m_new, F[a_ev])), q_new))
            acc[name]["wq"].append(float(np.exp(q_new)))
        if (r + 1) % 10 == 0 or r == R - 1:
            log(f"  ... replicate {r+1}/{R}")

    log("")
    log(f"== §4.9 跨底模通用性（R={R} 次独立重复；mean ± 95%CI，方括号为单次读数范围）==")
    log(f"{'底模':<8} {'无漂移':>24} {'漂移后(冻结)':>24} {'修复后':>24} {'检测→修复':>12}")
    log("-" * 96)
    for name, _ in BASES:
        d = acc[name]
        def cell(k):
            a = np.array(d[k]) * 100; m, h = ci95(a)
            return f"{m:5.1f}±{h:4.1f} [{a.min():.1f},{a.max():.1f}]"
        gain = np.array(d["fixed"]) - np.array(d["frozen"])
        mg, hg = ci95(gain * 100)
        log(f"{name:<8} {cell('nodrift'):>24} {cell('frozen'):>24} {cell('fixed'):>24} "
            f"{mg:+5.1f}±{hg:4.1f}pp")

    log("")
    log("== 区间宽度 ==")
    for name, _ in BASES:
        d = acc[name]
        a0, h0 = ci95(d["w0"]); a1, h1 = ci95(d["wq"])
        log(f"  {name:<8} 冻结 {a0:8.2f}±{h0:6.2f}x  →  修复后 {a1:8.2f}±{h1:6.2f}x")

    log("")
    log("== 复现门：第 1 次重复(seed=42) 应与 experiment_modelagnostic.py 的单次读数一致 ==")
    for name, _ in BASES:
        d = acc[name]
        log(f"  {name:<8} 无漂移 {d['nodrift'][0]*100:6.1f}% | 漂移后(冻) {d['frozen'][0]*100:6.1f}% | "
            f"修复后 {d['fixed'][0]*100:6.1f}% | 检-修 {(d['fixed'][0]-d['frozen'][0])*100:+6.1f}%")

    OUT = os.path.join(ROOT, "results_repro")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "experiment_modelagnostic_ci.txt"), "w", encoding="utf-8") as f:
        f.write("exit=0\n" + "\n".join(out) + "\n")
    np.savez(os.path.join(OUT, "modelagnostic_ci.npz"),
             names=np.array([n for n, _ in BASES]), R=R,
             **{f"{k}__{m}": np.array(v) for m, dd in acc.items() for k, v in dd.items()})
    print("\nsaved results_repro/experiment_modelagnostic_ci.txt", flush=True)


if __name__ == "__main__":
    main()
