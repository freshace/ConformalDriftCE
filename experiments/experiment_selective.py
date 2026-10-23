# -*- coding: utf-8 -*-
"""实验 #2：有保证的选择性预测 (risk-coverage)。

两条规则，按同一个可观测信号拒绝部分查询：
  R1 【我们 / 有保证】用 Mondrian 分组的 q_g 作信任分：接受 q_g <= lam 的组。
     因为每个组都单独满足了 >= 1-alpha 的共形覆盖（组内可交换），
     接受集是若干组的并，其名义覆盖仍 >= 1-alpha —— 与 lam 无关的【硬保证】。
  R2 【启发式 / 无保证】用 K 个 bootstrap LGBM 的预测标准差作不确定度
     （对标 Faper 的 MC-Dropout 方差、Fauce 的深度集成）：接受 std <= lam。【无任何保证】。

设置：A 无漂移(校准 3000) 与 C 相关性 7.39x 适应后(校准 500)。
指标：保留率 / 接受集覆盖 / 接受集平均宽度 / 接受集 q-error>10 占比 / 接受集平均 q-error。
"""
import os, sys, pickle, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, features, model, conformal, drift

MIN_N = 50
K_BOOT = 5
BAD_Q = 10.0     # q-error > 10 记为好坏界

def load_all(pkl_path):
    n_ts, n_cs = features.load_table_col_space(pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    b, y, F, nfc, ntab = [], [], [], [], []
    for ft, label, pgv, *_ , ex in data["features"]:
        true = float(np.expm1(label - 1)); pg = float(np.expm1(pgv - 1))
        if true <= 0 or pg <= 0:
            continue
        f = np.nan_to_num(np.concatenate(
            [features.build_schema_feature(ft, ex, n_ts, n_cs), features.stats9(ft)]),
            nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        f = np.hstack([f, np.log(np.clip(pg, 1.0, None))]).astype(np.float32)
        b.append(pg); y.append(true); F.append(f)
        nfc.append(len(set(ex.get("filter_col_identifiers", []))))
        ntab.append(len(ex.get("table_indices", [])))
    return (np.array(b), np.array(y), np.array(F, dtype=np.float32),
            np.array(nfc, dtype=np.int32), np.array(ntab, dtype=np.int32))

def qerr(true, est):
    est = np.clip(est, 1e-9, None)
    return np.maximum(true / est, est / true)

def group_qs(scores, groups, alpha, min_n):
    gl = conformal.calibrate_q(scores, alpha)
    qs = {}
    for g in np.unique(groups):
        s = scores[groups == g]
        if len(s) >= min_n:
            qs[int(g)] = conformal.calibrate_q(s, alpha)
    return gl, qs

def boot_std(feat_tr, base_tr, y_tr, feat_ev, rng):
    """K 个 bootstrap LGBM 修正器的预测标准差（对数空间），作为启发式不确定度。"""
    preds = []
    n = len(y_tr)
    for k in range(K_BOOT):
        sel = rng.randint(0, n, size=n)
        mk = model.fit_corrector(feat_tr[sel], base_tr[sel], y_tr[sel], config.LGBM_PARAMS)
        preds.append(model.predict_corrector(mk, feat_ev))
    return np.std(np.vstack(preds), axis=0)

def sweep_rule(label, score, accept_fn, thresholds, y_ev, est_ev, q_used, cov_fn):
    """按阈值扫描；每个阈值给保留率/覆盖(+95% 二项半宽)/宽度/坏例占比/平均 q-error。"""
    rows = []
    for lam in thresholds:
        acc = accept_fn(score, lam)
        if acc.sum() < 20:
            continue
        cov = float(np.mean(cov_fn(acc)))
        nacc = int(acc.sum())
        ci = 1.96 * float(np.sqrt(max(cov * (1.0 - cov), 1e-9) / nacc))   # 二项 95% 半宽
        w = float(np.exp(np.mean(q_used[acc])))
        qe = qerr(y_ev[acc], est_ev[acc])
        rows.append(dict(lam=float(lam), ret=float(acc.mean()), cov=cov, ci=ci,
                         width=w, bad=float(np.mean(qe > BAD_Q)), mean_qe=float(np.mean(qe))))
    return rows

def main():
    alpha = config.ALPHA
    rng = np.random.RandomState(config.SEED)
    base, true, feat, nfc, ntab = load_all(config.DATA_PATH)
    print("加载 %d 条 (K_BOOT=%d, MIN_N=%d, alpha=%.2f)" % (len(true), K_BOOT, MIN_N, alpha))

    idx = rng.permutation(len(true))
    tr = idx[:config.N_TRAIN]
    ca = idx[config.N_TRAIN:config.N_TRAIN + config.N_CAL]
    te = idx[config.N_TRAIN + config.N_CAL:config.N_TRAIN + config.N_CAL + config.N_TEST]
    m_old = model.fit_corrector(feat[tr], base[tr], true[tr], config.LGBM_PARAMS)
    y_drift = drift.inject_correlation_drift(true[te], nfc[te], 2.0, rng)
    perm = rng.permutation(len(te))
    a_tr = perm[:config.N_ADAPT_TRAIN]
    a_ca = perm[config.N_ADAPT_TRAIN:config.N_ADAPT_TRAIN + config.N_ADAPT_CAL]
    a_ev = perm[config.N_ADAPT_TRAIN + config.N_ADAPT_CAL:]
    m_new = model.fit_corrector(feat[te][a_tr], base[te][a_tr], y_drift[a_tr], config.LGBM_PARAMS)

    # ---- 设置 A：无漂移 ----
    sA_cal = conformal.score(true[ca], base[ca], model.predict_corrector(m_old, feat[ca]))
    sA_ev  = conformal.score(true[te], base[te], model.predict_corrector(m_old, feat[te]))
    estA   = base[te] * np.exp(model.predict_corrector(m_old, feat[te]))
    # ---- 设置 C：相关性 7.39x 适应后 ----
    sC_cal = conformal.score(y_drift[a_ca], base[te][a_ca], model.predict_corrector(m_new, feat[te][a_ca]))
    sC_ev  = conformal.score(y_drift[a_ev], base[te][a_ev], model.predict_corrector(m_new, feat[te][a_ev]))
    estC   = base[te][a_ev] * np.exp(model.predict_corrector(m_new, feat[te][a_ev]))

    settings = [
        dict(name="A 无漂移", s_cal=sA_cal, g_cal=ntab[ca], s_ev=sA_ev, g_ev=ntab[te],
             y_ev=true[te], est_ev=estA,
             f_bt=feat[ca][:2000], b_bt=base[ca][:2000], y_bt=true[ca][:2000], f_be=feat[te]),
        dict(name="C 7.39x(适应)", s_cal=sC_cal, g_cal=ntab[te][a_ca], s_ev=sC_ev, g_ev=ntab[te][a_ev],
             y_ev=y_drift[a_ev], est_ev=estC,
             f_bt=feat[te][a_tr], b_bt=base[te][a_tr], y_bt=y_drift[a_tr], f_be=feat[te][a_ev]),
    ]

    save = {}
    for S in settings:
        gl, qs = group_qs(S["s_cal"], S["g_cal"], alpha, MIN_N)
        q_ev = np.array([qs.get(int(g), gl) for g in S["g_ev"]], dtype=float)
        cov_ev = (S["s_ev"] <= q_ev)
        thr = sorted(set(np.round(q_ev, 6).tolist()))
        rows1 = sweep_rule("R1", q_ev, lambda s, l: s <= l + 1e-12, thr, S["y_ev"], S["est_ev"], q_ev, lambda a: cov_ev[a])
        std = boot_std(S["f_bt"], S["b_bt"], S["y_bt"], S["f_be"], np.random.RandomState(config.SEED + 7))
        thr2 = np.quantile(std, np.linspace(0.05, 0.95, 19))
        rows2 = sweep_rule("R2", std, lambda s, l: s <= l, thr2, S["y_ev"], S["est_ev"], q_ev, lambda a: cov_ev[a])
        save[(S["name"], "R1")] = rows1
        save[(S["name"], "R2")] = rows2

        print("\n" + "=" * 96)
        print("设置 %s   全局 q=%.3f (宽 %.2fx)  组内 q: %s" %
              (S["name"], gl, np.exp(gl), ", ".join("%d:%.2f" % (k, v) for k, v in sorted(qs.items()))))
        for tag, rows, note in [("R1 我们(有保证, 按组 q)", rows1, "覆盖应始终 >= 90%"),
                                ("R2 启发式(bootstrap 方差)", rows2, "无保证, 覆盖可跌破 90%")]:
            print("\n  %s   —— %s" % (tag, note))
            print("  %10s %9s %14s %10s %12s %12s" %
                  ("阈值", "保留率", "接受集覆盖", "平均宽度", "坏例(qe>10)", "平均 q-error"))
            for r in rows:
                print("  %10.4f %8.1f%% %8.1f%%±%.1f %9.2fx %11.1f%% %12.2f"
                      % (r["lam"], r["ret"] * 100, r["cov"] * 100, r["ci"] * 100,
                         r["width"], r["bad"] * 100, r["mean_qe"]))
            cmin = min(r["cov"] for r in rows); cmax = max(r["cov"] for r in rows)
            print("  -> 接受集覆盖范围 %.1f%% ~ %.1f%%；保留率 %.0f%%~%.0f%%；宽度 %.2fx~%.2fx"
                  % (cmin * 100, cmax * 100, min(r["ret"] for r in rows) * 100,
                     max(r["ret"] for r in rows) * 100,
                     min(r["width"] for r in rows), max(r["width"] for r in rows)))

    # ---- 多种子控制检查：R1 的接受集覆盖是否在多次抽样下稳定贴着 90% ----
    print("\n" + "=" * 96)
    print("多种子控制检查（R1，取最接近 50% 的阈值；受组粒度限制实际约 62%，5 个种子）")
    print("  %6s %8s %10s %14s %12s %10s" % ("种子", "保留率", "组数", "接受集覆盖", "95%半宽", "平均宽度"))
    ctl = []
    for sd in [42, 43, 44, 45, 46]:
        r2 = np.random.RandomState(sd)
        ix = r2.permutation(len(true))
        tr2 = ix[:config.N_TRAIN]
        ca2 = ix[config.N_TRAIN:config.N_TRAIN + config.N_CAL]
        te2 = ix[config.N_TRAIN + config.N_CAL:config.N_TRAIN + config.N_CAL + config.N_TEST]
        m2 = model.fit_corrector(feat[tr2], base[tr2], true[tr2], config.LGBM_PARAMS)
        sc = conformal.score(true[ca2], base[ca2], model.predict_corrector(m2, feat[ca2]))
        se = conformal.score(true[te2], base[te2], model.predict_corrector(m2, feat[te2]))
        gl2, qs2 = group_qs(sc, ntab[ca2], alpha, MIN_N)
        qe2 = np.array([qs2.get(int(g), gl2) for g in ntab[te2]], dtype=float)
        # 选一个阈值使保留率尽量接近 50%
        cand = sorted(set(np.round(qe2, 6).tolist()))
        best = min(cand, key=lambda l: abs(float(np.mean(qe2 <= l + 1e-12)) - 0.5))
        acc = qe2 <= best + 1e-12
        cov = float(np.mean(se[acc] <= qe2[acc]))
        ci = 1.96 * float(np.sqrt(max(cov * (1 - cov), 1e-9) / max(int(acc.sum()), 1)))
        w = float(np.exp(np.mean(qe2[acc])))
        ctl.append(dict(seed=sd, ret=float(acc.mean()), groups=len(qs2), cov=cov, ci=ci, width=w))
        print("  %6d %7.1f%% %10d %9.1f%%±%.1f %11s %9.2fx"
              % (sd, acc.mean() * 100, len(qs2), cov * 100, ci * 100, "", w))
    cv = np.array([c["cov"] for c in ctl])
    print("  -> 5 种子接受集覆盖：均值 %.1f%%，标准差 %.1fpp，范围 %.1f%%~%.1f%%（名义目标 90%%）"
          % (cv.mean() * 100, cv.std() * 100, cv.min() * 100, cv.max() * 100))
    save[("control_R1", "5seeds")] = ctl

    o = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_selective.npy")
    np.save(o, save, allow_pickle=True)
    print("\n已存 results_selective.npy")

if __name__ == "__main__":
    main()
