# -*- coding: utf-8 -*-
"""实验 #1：分组(条件)覆盖 —— Mondrian conformal。

问题：split conformal 只给【边际】覆盖，子群会系统性欠覆盖。
做法：按【join 宽度(表数)】或【过滤列数】分组，每组内各自校准 q_g
    （每组只用本组校准样本；在组内交换且样本充足时才有 Mondrian 的组内有限样本保证）；样本 < MIN_N 的组回退全局 q。
对照：全局 q（现行） vs 分组 q（Mondrian）。
指标：每组覆盖、整体覆盖、组间极差(max-min)、最差组、平均区间宽度。

三设置：A 无漂移；B 相关性 7.39x（冻结）；C 相关性 7.39x（适应 500 标签）。
注意：校准集与其分组元数据必须来自【同一批查询】，否则索引错位（初版即此 bug）。
"""
import os, sys, pickle, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, features, model, conformal, drift

MIN_N = 50
MIN_TEST = 30
GROUP_KEYS = [("join 宽度(表数)", "ntab"), ("过滤列数", "nfc")]

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

def calib_groups(scores, groups, alpha, min_n):
    assert len(scores) == len(groups), "校准分数与分组长度不一致"
    gl = conformal.calibrate_q(scores, alpha)
    qs = {}
    for g in np.unique(groups):
        s = scores[groups == g]
        if len(s) >= min_n:
            qs[int(g)] = conformal.calibrate_q(s, alpha)
    return gl, qs

def summarize(scores, groups, gl, qs):
    assert len(scores) == len(groups), "评估分数与分组长度不一致"
    qg = np.array([qs.get(int(g), gl) for g in groups], dtype=float)
    rows = []
    for g in np.unique(groups):
        m = groups == g
        if int(m.sum()) < MIN_TEST:
            continue
        rows.append(dict(g=int(g), n=int(m.sum()),
                         cov_glob=float(np.mean(scores[m] <= gl)),
                         cov_grp=float(np.mean(scores[m] <= qg[m])),
                         w_glob=float(np.exp(gl)), w_grp=float(np.exp(qg[m].mean())),
                         q_grp=float(qg[m].mean()), fallback=bool(int(g) not in qs)))
    cg = [r["cov_glob"] for r in rows]; cp = [r["cov_grp"] for r in rows]
    return dict(rows=rows, n=len(scores),
                cov_all_glob=float(np.mean(scores <= gl)),
                cov_all_grp=float(np.mean(scores <= qg)),
                spread_glob=(max(cg) - min(cg)) if cg else 0.0,
                spread_grp=(max(cp) - min(cp)) if cp else 0.0,
                min_glob=(min(cg) if cg else 1.0), min_grp=(min(cp) if cp else 1.0),
                w_all_glob=float(np.exp(gl)), w_all_grp=float(np.exp(qg.mean())))

def main():
    alpha = config.ALPHA
    rng = np.random.RandomState(config.SEED)
    base, true, feat, nfc, ntab = load_all(config.DATA_PATH)
    print("加载 %d 条 (MIN_N=%d, MIN_TEST=%d, alpha=%.2f)" % (len(true), MIN_N, MIN_TEST, alpha))

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

    def mk(title, s_cal, gcal_nfc, gcal_ntab, s_ev, gev_nfc, gev_ntab):
        return dict(title=title, s_cal=s_cal,
                    gcal={"nfc": np.asarray(gcal_nfc), "ntab": np.asarray(gcal_ntab)},
                    s_ev=s_ev, gev={"nfc": np.asarray(gev_nfc), "ntab": np.asarray(gev_ntab)})

    s_cal_ab = conformal.score(true[ca], base[ca], model.predict_corrector(m_old, feat[ca]))
    s_ev_a  = conformal.score(true[te], base[te], model.predict_corrector(m_old, feat[te]))
    s_ev_b  = conformal.score(y_drift, base[te], model.predict_corrector(m_old, feat[te]))
    s_cal_c = conformal.score(y_drift[a_ca], base[te][a_ca], model.predict_corrector(m_new, feat[te][a_ca]))
    s_ev_c  = conformal.score(y_drift[a_ev], base[te][a_ev], model.predict_corrector(m_new, feat[te][a_ev]))

    settings = [
        mk("A 无漂移", s_cal_ab, nfc[ca], ntab[ca], s_ev_a, nfc[te], ntab[te]),
        mk("B 相关性7.39x(冻结)", s_cal_ab, nfc[ca], ntab[ca], s_ev_b, nfc[te], ntab[te]),
        mk("C 相关性7.39x(适应)", s_cal_c, nfc[te][a_ca], ntab[te][a_ca], s_ev_c, nfc[te][a_ev], ntab[te][a_ev]),
    ]

    save = {}
    for S in settings:
        print("\n" + "=" * 90)
        print("设置 %s   (校准 n=%d, 评估 n=%d)" % (S["title"], len(S["s_cal"]), len(S["s_ev"])))
        for label, key in GROUP_KEYS:
            gl, qs = calib_groups(S["s_cal"], S["gcal"][key], alpha, MIN_N)
            R = summarize(S["s_ev"], S["gev"][key], gl, qs)
            save[(S["title"], key)] = dict(gl=gl, qs=qs, rows=R["rows"])
            print("\n  [按 %s]  全局 q=%.3f (宽 %.2fx)  组内 q: %s"
                  % (label, gl, np.exp(gl), ", ".join("%d:%.3f" % (k, v) for k, v in sorted(qs.items()))))
            print("  %6s %6s %12s %12s %10s %10s %6s" %
                  ("组", "n", "全局q覆盖", "分组q覆盖", "全局宽", "分组宽", "回退"))
            for r in R["rows"]:
                print("  %6d %6d %11.1f%% %11.1f%% %9.2fx %9.2fx %6s"
                      % (r["g"], r["n"], r["cov_glob"] * 100, r["cov_grp"] * 100,
                         r["w_glob"], r["w_grp"], "是" if r["fallback"] else ""))
            print("  " + "-" * 78)
            print("  整体 %.1f%% -> %.1f%% | 组间极差 %.1fpp -> %.1fpp | 最差组 %.1f%% -> %.1f%% | 平均宽 %.2fx -> %.2fx"
                  % (R["cov_all_glob"] * 100, R["cov_all_grp"] * 100,
                     R["spread_glob"] * 100, R["spread_grp"] * 100,
                     R["min_glob"] * 100, R["min_grp"] * 100, R["w_all_glob"], R["w_all_grp"]))

    o = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_groupcov.npy")
    np.save(o, save, allow_pickle=True)
    print("\n已存 results_groupcov.npy")

if __name__ == "__main__":
    main()
