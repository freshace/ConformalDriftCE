# -*- coding: utf-8 -*-
"""闭环决策价值实验（#5 + #1）：时序流上修复策略的 coverage+width+GMQ 联合比较。

回答两个遗留缺口（审查表 🟡 项）：
  #5 recal-only vs retrain 的【联合】比较：以前只比 coverage/达标/标签；
     本实验每个策略同时报 coverage、width(=exp(q))、GMQ(部署点估计)与标签开销。
  #1 triggered repair vs unconditional repair：监控是否真的承担决策作用。
     对照：冻结 / 无监控每窗口盲修(200x20) / 无监控按日程重训 / 触发式(监控+按需修)。

流协议与 experiment_temporalhysteresis.py 完全一致（同切分/同窗口/同阈值/同冷却）：
  train 20000 / cal 3000 / stream 10000(20 窗×500) / pool 8000 互不相交；
  每窗口前 L 条作监控标签、后 300 条作评估(检测器不可见)；
  漂移为相关性漂移(rho 按模式演化)；报警 = ph+1.645*SE < 0.9(Wald 单窗口近似)；
  修复标签取自 pool 递增段，冷却 2 窗口。

策略：
  frozen           冻结（监控都不做），labels=0
  trig_recal_L50   监控 L=50；触发 -> 只重校准 200 标签
  trig_recal_L150  监控 L=150；触发 -> 只重校准 200
  trig_retrain_L150 监控 L=150；触发 -> 重训+重校准 1000
  trig_hybrid_L150 监控 L=150；触发 -> 先只重校准 200；若在重校准之后再次报警
                   (问题持续) -> 升级重训 1000（廉价快响应 + 持续时才升级）
  uncond_recal     无监控；每窗口盲重校准 200（20 窗 = 4000 标签）
  sched_retrain    无监控；按固定日程(t=5,12)盲重训 1000（2000 标签）

说明：inject_correlation_drift 是确定性逐元素位移，因此监控/评估/修复的漂移
标签对所有策略逐位相同；策略间唯一差异是【何时、花多少标签、做什么动作】。
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import data
import drift
import model
import conformal

T_WIN, W_WIN, N_EVAL = 20, 500, 300
TAU, N_STREAM, N_POOL = 0.85, 10000, 8000
Z, L_MAIN = 1.645, 150
N_RETRAIN, N_RECAL = 1000, 200
SEEDS = [42, 43, 44, 45, 46]
SPIKE = [6, 13, 18]
SCHED_WINDOWS = (5, 12)
COOLDOWN = 2
POLICIES = ["frozen", "trig_recal_L50", "trig_recal_L150",
            "trig_retrain_L150", "trig_hybrid_L150", "uncond_recal", "sched_retrain"]


def rho_uniform(t):
    return 2.0


def rho_periodic(t):
    return 1.0 * (1.0 + np.sin(2.0 * np.pi * t / 10.0))


def rho_trend(t):
    return 2.0 * t / (T_WIN - 1)


def rho_longtail(t):
    return 2.0 if t in SPIKE else 0.2


PATTERNS = [("none", lambda t: 0.0), ("uniform", rho_uniform),
            ("periodic", rho_periodic), ("trend", rho_trend),
            ("long-tail", rho_longtail)]


def triggering(ph, n, target):
    ub = ph + Z * np.sqrt(max(ph * (1.0 - ph), 1e-9) / n)
    return bool(ub < target)


def gmq_of(y, est):
    ratio = np.maximum(y, est) / np.maximum(np.minimum(y, est), 1e-12)
    return float(np.exp(np.mean(np.abs(np.log(ratio)))))


import io as _io_mod  # noqa: E402

LOG_BUF = _io_mod.StringIO()


def log(msg):
    print(msg, flush=True)
    LOG_BUF.write(str(msg) + "\n")

def _do_repair(action, cursor, pool, t, rho, m_cur, f_ad, b_ad):
    """从 pool 的递增段取标签执行修复；返回 (m_cur, q_cur, n_labels, new_cursor)。"""
    n_take = N_RECAL if action == "recal" else N_RETRAIN
    s = slice(cursor, cursor + n_take)
    cursor += n_take
    y_a = drift.inject_correlation_drift(pool["y"][s], pool["nfc"][s], rho[t], None)
    if action == "recal":
        q_cur = conformal.calibrate_q(
            conformal.score(y_a, b_ad[s], model.predict_corrector(m_cur, f_ad[s])),
            config.ALPHA)
        return m_cur, q_cur, n_take, cursor
    half = n_take // 2
    m_new = model.fit_corrector(f_ad[s][:half], b_ad[s][:half], y_a[:half], config.LGBM_PARAMS)
    q_cur = conformal.calibrate_q(
        conformal.score(y_a[half:], b_ad[s][half:], model.predict_corrector(m_new, f_ad[s][half:])),
        config.ALPHA)
    return m_new, q_cur, n_take, cursor


def run_policy(policy, rho_t, y_e, y_m_all, st, ad, m0, q0):
    """单条流上跑一个策略；返回逐窗口轨迹与汇总。流标签对所有策略逐位相同。"""
    b_st, f_st = st["b"], st["f"]
    f_ad, b_ad = ad["f"], ad["b"]
    pool = {"y": ad["y"], "nfc": ad["nfc"]}
    target = 1.0 - config.ALPHA

    m_cur, q_cur = m0, q0
    cursor, last_repair, last_action = 0, -99, None
    mon_L = {"trig_recal_L50": 50, "trig_recal_L150": L_MAIN,
             "trig_retrain_L150": L_MAIN, "trig_hybrid_L150": L_MAIN}.get(policy, 0)
    cov = np.zeros(T_WIN)
    width = np.zeros(T_WIN)
    abslog_sum, abslog_n = 0.0, 0
    n_rep, labels_rep, alarms = 0, 0, []
    trig_hist = []

    for t in range(T_WIN):
        # ---- 决策（只有监控类策略花监控标签；uncond/sched 无监控）----
        fired = False
        if policy.startswith("trig"):
            # 监控标签取自本窗口前 mon_L 条（预注入的漂移标签 y_m_all[t]）
            y_m = y_m_all[t][:mon_L]
            ph = conformal.coverage(
                conformal.score(y_m, b_st[t][:mon_L],
                                model.predict_corrector(m_cur, f_st[t][:mon_L])), q_cur)
            trig = triggering(ph, mon_L, target)
            trig_hist.append(trig)
            if trig:
                alarms.append(t)
            since = t - last_repair
            if policy == "trig_hybrid_L150":
                escalate = (last_action == "recal" and last_repair >= 0
                            and any(trig_hist[last_repair + 1:t]))
                action = "retrain" if escalate else "recal"
            elif policy == "trig_retrain_L150":
                action = "retrain"
            else:
                action = "recal"
            fired = trig and since > COOLDOWN
        elif policy == "uncond_recal":
            action, fired = "recal", True   # 每窗口无条件盲修
        elif policy == "sched_retrain":
            action, fired = ("retrain", True) if t in SCHED_WINDOWS else (None, False)

        if fired and action is not None:
            m_cur, q_cur, n_take, cursor = _do_repair(
                action, cursor, pool, t, rho_t, m_cur, f_ad, b_ad)
            n_rep += 1
            labels_rep += n_take
            last_repair, last_action = t, action

        # ---- 评估（检测器不可见的后 300 条）----
        corr_ev = model.predict_corrector(m_cur, f_st[t][W_WIN - N_EVAL:])
        est_ev = b_st[t][W_WIN - N_EVAL:] * np.exp(np.clip(corr_ev, -30.0, 30.0))
        cov[t] = conformal.coverage(
            conformal.score(y_e[t], b_st[t][W_WIN - N_EVAL:], corr_ev), q_cur)
        width[t] = float(np.exp(q_cur))
        ratio = np.maximum(y_e[t], est_ev) / np.maximum(np.minimum(y_e[t], est_ev), 1e-12)
        abslog_sum += float(np.sum(np.abs(np.log(ratio))))
        abslog_n += len(y_e[t])

    mon_cost = mon_L * T_WIN if mon_L else 0
    return {
        "cov": cov.tolist(), "width": width.tolist(),
        "pass_rate": float(np.mean(cov >= TAU)),
        "mean_cov": float(cov.mean()), "min_cov": float(cov.min()),
        "mean_width": float(width.mean()),
        "gmq": float(np.exp(abslog_sum / abslog_n)),
        "n_rep": n_rep, "labels_rep": labels_rep, "labels_mon": mon_cost,
        "labels_total": labels_rep + mon_cost,
        "n_alarm": len(alarms),
    }


def run_seed(seed, base, true, feat, nfc):
    """复用时序协议：同切分/同窗口/同阈值；返回 {(pattern, policy): metrics}。"""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(true))
    itr = idx[:config.N_TRAIN]
    ica = idx[config.N_TRAIN:config.N_TRAIN + config.N_CAL]
    ist = idx[config.N_TRAIN + config.N_CAL:config.N_TRAIN + config.N_CAL + N_STREAM]
    iad = idx[config.N_TRAIN + config.N_CAL + N_STREAM:
              config.N_TRAIN + config.N_CAL + N_STREAM + N_POOL]
    f_tr, b_tr, y_tr = feat[itr], base[itr], true[itr]
    f_ca, b_ca, y_ca = feat[ica], base[ica], true[ica]
    f_st_all, b_st_all, y_st_all, nfc_st_all = feat[ist], base[ist], true[ist], nfc[ist]
    f_ad, b_ad, y_ad, nfc_ad = feat[iad], base[iad], true[iad], nfc[iad]

    m0 = model.fit_corrector(f_tr, b_tr, y_tr, config.LGBM_PARAMS)
    q0 = conformal.calibrate_q(
        conformal.score(y_ca, b_ca, model.predict_corrector(m0, f_ca)), config.ALPHA)

    wins = [np.arange(i * W_WIN, (i + 1) * W_WIN) for i in range(T_WIN)]
    ev = [w[W_WIN - N_EVAL:] for w in wins]
    mon = [w[:200] for w in wins]

    out = {}
    for name, fn in PATTERNS:
        rho_t = np.array([float(fn(t)) for t in range(T_WIN)])
        # 漂移注入是确定性逐元素位移：对每个 pattern 只注入一次，所有策略共享同一流
        y_e = [drift.inject_correlation_drift(y_st_all[e], nfc_st_all[e], rho_t[t], None)
               for t, e in enumerate(ev)]
        y_m_all = [drift.inject_correlation_drift(y_st_all[m], nfc_st_all[m], rho_t[t], None)
                   for t, m in enumerate(mon)]
        st = {"b": [b_st_all[w] for w in wins], "f": [f_st_all[w] for w in wins]}
        ad = {"f": f_ad, "b": b_ad, "y": y_ad, "nfc": nfc_ad}

        # 冻结轨迹（参照）
        cov_fro = np.array([conformal.coverage(
            conformal.score(y_e[t], b_st_all[ev[t]],
                            model.predict_corrector(m0, f_st_all[ev[t]])), q0)
            for t in range(T_WIN)])

        for policy in POLICIES:
            r = run_policy(policy, rho_t, y_e, y_m_all, st, ad, m0, q0)
            r["mean_fro"] = float(cov_fro.mean())
            r["cov_fro"] = cov_fro.tolist()
            r["rho"] = rho_t.tolist()
            out[(name, policy)] = r
    return out


def main():
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    log("加载 %d 条；种子 %s；策略 %s" % (len(true), SEEDS, POLICIES))
    per = []
    for s in SEEDS:
        log("  seed %d ..." % s)
        per.append(run_seed(s, base, true, feat, nfc))

    def agg(p, m, k):
        v = np.array([d[(p, m)][k] for d in per], dtype=float)
        return float(v.mean()), float(v.std())

    def cell(p, m, k, s=1.0):
        a, b = agg(p, m, k)
        return "%.1f±%.1f" % (a * s, b * s)

    J = ["trig_recal_L150", "trig_retrain_L150", "trig_hybrid_L150"]
    D = ["frozen", "uncond_recal", "trig_recal_L50", "sched_retrain"]

    log("\n" + "=" * 118)
    log("表 1a  (Q#5) 覆盖与标签：frozen / trig_recal / trig_retrain / trig_hybrid（5 种子 mean±std）")
    log("%-11s %18s %18s %18s %20s" % ("模式", "策略", "覆盖", "达标占比", "修复/监控/总标签"))
    log("-" * 118)
    for name, _ in PATTERNS:
        for m in ["frozen"] + J:
            c = cell(name, m, "mean_cov", 100)
            p = cell(name, m, "pass_rate", 100)
            lr = "%.0f/%.0f/%.0f" % tuple(agg(name, m, k)[0] for k in
                                          ("labels_rep", "labels_mon", "labels_total"))
            # ★ 2026-09-17 修：数据行原用 print，只进 stdout，落盘 txt 里只剩表头
            #   （txt 由 LOG_BUF 写出）⇒ 表格看似"空"。全部表格行必须走 log()。
            log("%-11s %18s %18s %18s %20s" % (name, m, c, p, lr))

    log("\n" + "=" * 118)
    log("表 1b  (Q#5) 联合口径：区间宽度与部署点估计 GMQ（5 种子 mean±std）")
    log("%-11s %26s %26s %26s" % ("模式", "trig_recal 宽/GMQ", "trig_retrain 宽/GMQ",
                                    "trig_hybrid 宽/GMQ"))
    log("-" * 118)
    for name, _ in PATTERNS:
        cells = []
        for m in J:
            w = cell(name, m, "mean_width")
            g = cell(name, m, "gmq")
            cells.append("%s / %s" % (w, g))
        log("%-11s %26s %26s %26s" % (name, cells[0], cells[1], cells[2]))
    gf = [float(np.exp(np.mean(np.log([d[(name, "frozen")]["gmq"] for d in per]))))
          for name, _ in PATTERNS]
    log("  frozen GMQ（跨种子几何均值）: " +
          ", ".join("%s=%.2f" % (n, g) for (n, _), g in zip(PATTERNS, gf)))

    log("\n" + "=" * 118)
    log("表 2  (Q#1) 监控的决策价值：triggered vs unconditional（同流同漂移，5 种子 mean±std）")
    log("%-11s %18s %18s %16s %12s %12s" % ("模式", "策略", "覆盖", "达标占比", "总标签", "修复次数"))
    log("-" * 118)
    for name, _ in PATTERNS:
        for m in D:
            c = cell(name, m, "mean_cov", 100)
            p = cell(name, m, "pass_rate", 100)
            lt = "%.0f" % agg(name, m, "labels_total")[0]
            nr = "%.1f" % agg(name, m, "n_rep")[0]
            log("%-11s %18s %18s %16s %12s %12s" % (name, m, c, p, lt, nr))
    log("  注：标签=监控标签+修复标签；uncond/sched 无监控成本。none 行的修复即误报代价。")

    summary = {}
    for name, _ in PATTERNS:
        for m in POLICIES:
            for k in ["pass_rate", "mean_cov", "min_cov", "mean_width", "gmq",
                      "n_rep", "labels_rep", "labels_mon", "labels_total", "n_alarm",
                      "mean_fro"]:
                a, b = agg(name, m, k)
                summary["%s|%s|%s" % (name, m, k)] = [a, b]
    traj = {}
    for name, _ in PATTERNS:
        for m in ["frozen", "trig_recal_L150", "trig_retrain_L150",
                  "trig_hybrid_L150", "uncond_recal", "sched_retrain"]:
            traj["%s|%s" % (name, m)] = dict(
                cov=np.mean([d[(name, m)]["cov"] for d in per], axis=0).tolist(),
                width=np.mean([d[(name, m)]["width"] for d in per], axis=0).tolist(),
                rho=per[0][(name, "frozen")]["rho"])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt_path = os.path.join(root, "results_repro", "experiment_loop_decision.txt")
    with open(txt_path, "w", encoding="utf-8") as h:
        h.write("exit=0\n" + LOG_BUF.getvalue() + "\n")
    np.save(os.path.join(root, "results_loop_decision.npy"),
            {"summary": summary, "traj": traj}, allow_pickle=True)
    # ★ 自检门（2026-09-17）：① 重开落盘 txt 数数据行；② 论文引用的每个 cell 从
    #   summary 重算比对。结果补写进 txt，任何失败 ⇒ 非零退出（run_all 才能看见）。
    _mark = len(LOG_BUF.getvalue())
    fails = self_check(txt_path, summary)
    with open(txt_path, "a", encoding="utf-8") as h:
        h.write("\n[自检门]\n" + LOG_BUF.getvalue()[_mark:])
    log("\nsaved results_repro/experiment_loop_decision.txt + results_loop_decision.npy")
    if fails:
        raise SystemExit(1)


# ★ 2026-09-17 自检门（防静默回归 + 把论文数字钉在产物上）。
# 背景：本脚本三张表的**数据行**原用 print() 写出，而落盘 txt 由 LOG_BUF 生成
# （表头用 log()）⇒ txt 里只剩表头，表格"看起来是空的"，且 exit=0 静默通过。
# 这个 bug 骗过了一次人眼复核，故补两道门：
#   (a) 落盘后**重开 txt** 数数据行（print 类回归立刻暴露）；
#   (b) 把 experiments.tex 与 conclusion.tex 引用到的每个 cell 从 summary 重算比对。
def self_check(txt_path, summary):
    fails = []
    saved = _io_mod.open(txt_path, encoding="utf-8").read()
    n_data = sum(1 for l in saved.splitlines() if l.count("\u00b1") >= 2)
    # 表1a 5x4=20 行 + 表1b 5 行 + 表2 5x4=20 行
    if n_data < 45:
        fails.append("落盘 txt 数据行仅 %d 行（期望 >=45：表1a 20 + 表1b 5 + 表2 20）" % n_data)
    log("  [格式] 落盘 txt 数据行 %d 行（期望 >=45）" % n_data)

    def g(p, m, k, s=1.0):
        key = "%s|%s|%s" % (p, m, k)
        if key not in summary:      # ★ 缺键 ⇒ NaN ⇒ 比对必然 FAIL（绝不静默、也不崩）
            return float("nan")
        return summary[key][0] * s

    DRIFT_MODES = [n for n, _ in PATTERNS if n != "none"]   # 宽度区间只在这 4 个模式上

    # 论文引用的 cell（容差 0.051 = 论文取一位小数的舍入上限）
    want = [
        ("none|trig_recal_L50 覆盖",      g("none", "trig_recal_L50", "mean_cov", 100), 89.5),
        ("none|uncond_recal 覆盖",        g("none", "uncond_recal", "mean_cov", 100), 89.8),
        ("uniform|trig_recal_L50 覆盖",   g("uniform", "trig_recal_L50", "mean_cov", 100), 88.6),
        ("uniform|uncond_recal 覆盖",     g("uniform", "uncond_recal", "mean_cov", 100), 89.4),
        ("uniform|sched_retrain 覆盖",    g("uniform", "sched_retrain", "mean_cov", 100), 83.0),
        ("periodic|trig_recal_L50 覆盖",  g("periodic", "trig_recal_L50", "mean_cov", 100), 91.3),
        ("periodic|uncond_recal 覆盖",    g("periodic", "uncond_recal", "mean_cov", 100), 89.5),
        ("periodic|sched_retrain 覆盖",   g("periodic", "sched_retrain", "mean_cov", 100), 84.0),
        ("long-tail|trig_recal_L50 覆盖", g("long-tail", "trig_recal_L50", "mean_cov", 100), 93.0),
        ("long-tail|uncond_recal 覆盖",   g("long-tail", "uncond_recal", "mean_cov", 100), 89.4),
        ("trend|trig_recal_L50 覆盖",     g("trend", "trig_recal_L50", "mean_cov", 100), 87.3),
        ("trend|uncond_recal 覆盖",       g("trend", "uncond_recal", "mean_cov", 100), 89.4),
        # trend 诚实例外：重训式覆盖最好
        ("trend|trig_retrain_L150 覆盖",  g("trend", "trig_retrain_L150", "mean_cov", 100), 89.3),
        ("trend|trig_retrain L150 达标",  g("trend", "trig_retrain_L150", "pass_rate", 100), 99.0),
        ("trend|trig_recal_L150 覆盖",    g("trend", "trig_recal_L150", "mean_cov", 100), 88.2),
        ("trend|trig_recal L150 达标",    g("trend", "trig_recal_L150", "pass_rate", 100), 87.0),
        # 标签预算
        ("L=50 触发式总标签 min",         min(g(p, "trig_recal_L50", "labels_total")
                                              for p, _ in PATTERNS), 1160.0),
        ("L=50 触发式总标签 max",         max(g(p, "trig_recal_L50", "labels_total")
                                              for p, _ in PATTERNS), 1600.0),
        ("uncond_recal 总标签",           g("uniform", "uncond_recal", "labels_total"), 4000.0),
        # GMQ：recal 恒等于冻结；retrain 在 periodic/long-tail 恶化、uniform 改善
        ("uniform frozen GMQ",            g("uniform", "frozen", "gmq"), 5.24),
        ("uniform retrain GMQ",           g("uniform", "trig_retrain_L150", "gmq"), 4.22),
        ("periodic frozen GMQ",           g("periodic", "frozen", "gmq"), 3.34),
        ("periodic retrain GMQ",          g("periodic", "trig_retrain_L150", "gmq"), 4.60),
        ("long-tail frozen GMQ",          g("long-tail", "frozen", "gmq"), 2.74),
        ("long-tail retrain GMQ",         g("long-tail", "trig_retrain_L150", "gmq"), 4.29),
        # 区间宽度端点：★ 限定在**四种漂移模式**上（排除 none）——
        #   论文写 "recal 12.6--26.2x vs retrain 22.9--30.5x" 时，none 行的
        #   7.4x / 13.4x 是无漂移基线，不属于"漂移下修复策略"的比较范围。
        #   （首次跑这道门时报 FAIL，正是它把这条口径边界钉出来。）
        ("recal 宽度 min(漂移4模式)",
         min(g(p, "trig_recal_L150", "mean_width") for p in DRIFT_MODES), 12.6),
        ("recal 宽度 max(漂移4模式)",
         max(g(p, "trig_recal_L150", "mean_width") for p in DRIFT_MODES), 26.2),
        ("retrain 宽度 min(漂移4模式)",
         min(g(p, "trig_retrain_L150", "mean_width") for p in DRIFT_MODES), 22.9),
        ("retrain 宽度 max(漂移4模式)",
         max(g(p, "trig_retrain_L150", "mean_width") for p in DRIFT_MODES), 30.5),
        # none 行（无漂移）单独钉住，防止上面排除后没人管
        ("none|trig_recal_L150 宽度",     g("none", "trig_recal_L150", "mean_width"), 7.4),
        ("none|trig_retrain_L150 宽度",   g("none", "trig_retrain_L150", "mean_width"), 13.4),
    ]
    for label, got, exp in want:
        ok = (got == got) and abs(got - exp) <= 0.051     # got==got 排除 NaN
        log("  %s  %-32s 产物=%8s  论文=%s"
            % ("PASS" if ok else "FAIL", label,
               "%.2f" % got if got == got else "缺键/NaN", exp))
        if not ok:
            fails.append("%s: 产物=%s 论文=%s"
                         % (label, "%.3f" % got if got == got else "缺键/NaN", exp))
    log("  [自检] %s（%d 项论文数字 + 数据行完整性）"
        % ("PASS" if not fails else "FAIL: " + "; ".join(fails), len(want)))
    return fails


if __name__ == "__main__":
    main()
