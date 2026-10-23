# -*- coding: utf-8 -*-
"""一键可复现：跑到全部漂移实验，输出到 results_repro/。固定种子(SEED=42)+n_jobs=1，确定性。"""
import subprocess, os, sys

DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DIR, "results_repro")
os.makedirs(OUT, exist_ok=True)

EXPS = [
    # ---- 原有：覆盖率口径 ----
    "experiment_correlation.py",
    # ★ 2026-09-16 扶正：数据级相关性漂移（真改数据、边际严格不变、改完重算真值）。
    #   此前是孤儿实验——论文 §4.3 用了它，但它不在 run_all 也不在 RESULTS_MAP，
    #   导致论文里最硬的数据漂移证据无法一键复现。
    "experiment_correlation_realdata.py",
    "experiment_scaledrift.py",
    "experiment_workloaddrift.py",
    "experiment_mixeddrift.py",
    "experiment_schemadrift.py",
    "experiment_baselines.py",
    "experiment_modelagnostic.py",
    "analyze_consistency.py",
    "experiment_ablation.py",
    "experiment_residual_necessity.py",
    "experiment_residual_sensitivity.py",
    "experiment_loop_decision.py",
    "experiment_crossdataset.py",
    "experiment_alphatradeoff.py",
    # ★ 2026-09-16 补入：审计发现它在 RESULTS_MAP 里却不在 EXPS，同样无法一键复现
    "experiment_qerror_sideeffect.py",
"experiment_verify_audit.py",
    "experiment_detection.py",
"experiment_detection_ci.py",
"experiment_necessity_ci.py",
    # ★ 2026-09-17 P4：把仍为单次读数的关键结论补重复切分 CI（各有复现门）
    "experiment_correlation_ci.py",      # §4.2 相关性漂移
    "experiment_ablation_ci.py",         # §4.10 消融 C/D（及 A/B）
    "experiment_modelagnostic_ci.py",    # §4.9 跨底模（P1 的可行替代）
    "experiment_driftspectrum_ci.py",    # §4.6 尺度 / 混合 / schema
    "experiment_baselines_ci.py",        # §4.8 Oracle 对比
    "experiment_crossdataset_ci.py",     # §4.9 跨数据集（4 库 × 3 漂移）
    "experiment_temporaldrift.py",
    "experiment_groupcoverage.py",
"experiment_groupcoverage_ci.py",
    "experiment_selective.py",
    "experiment_temporalhysteresis.py",
    # ---- 本次新增：策略与副作用（覆盖率口径）----
    "experiment_aci_baseline.py",
    "experiment_hybrid_policy.py",
    "experiment_recal_sideeffects.py",
    "experiment_propA_bound.py",
    # ---- 本次新增：q*(alpha) 主指标口径（论文新主表）----
    "experiment_qstar_report.py",        # 主表：FROZEN/ACI/RECAL/RETRAIN 的 rho 与宽度缺口
    "experiment_qstar_precondition.py",  # 前提：无偏/滞后/有偏 + 监控器盲区
    "experiment_qstar_bias_dose.py",     # 剂量-响应
    "experiment_qstar_audit_design.py",  # 审计设计规则（独立 且 有代表性）
    "experiment_qstar_detector.py",      # 检测器：覆盖率 vs q* vs ACI力度（结论是否翻转）
    # ★ 2026-09-16 新增：上面那个数据级实验的【重复切分 + CI】版本。
    #   原脚本只有一次随机实现；本项目已实测单次读数极差可达 28.6pp（workload drift），
    #   故补 R=50 次独立重复（每次同时重抽数据实现与查询/划分）。
    "experiment_correlation_realdata_ci.py",
    # ★ 2026-09-17 §4.11 下游 plan regret 的**事后分析**：只读 npz，不接 PostgreSQL，故可进快速集。
    #   原产物不可用（见下方 §4.11 说明块），重跑脚本需真 PG，故只在 SLOW 说明里给命令。
    "experiment_planregret_analysis.py",
]

# ★ 2026-09-17 §4.11「论文数字 ↔ 产物」核对门禁（只读 npz；**必须每轮重报前跑**）。
#   它把 §4.11 / conclusion / 图注引用的每个数字从 npz 重算并逐条比对；
#   本轮它抓到一处真实转写错误（把窄档 D=16 子集的 51.4%→31.4% 误挂在共形自洽档上）。
#   放在 EXPS 之外单独执行，便于在改完论文后立刻复跑。
# ★ 2026-09-17 修正：路径前缀原写作 `tools/`，但仓库里是 **`.tools/`**（带点）。
#   后果很隐蔽——`run_all` 只会打印 `[MISSING tools/_g1_verify_paper_numbers.py]`，
#   而 MISSING **不计入失败**，于是"run_all 0 失败"里其实**从未包含这道门禁**。
#   凡新增跨目录脚本，务必先确认目录名（本仓库：`.tools/`）。
VERIFY_GATES = [
    (".tools/_g1_verify_paper_numbers.py", 900),   # 相对仓库根（PRICE-EnhancedBoost）
    # ★ 2026-09-17 新增版面门禁：`xelatex exit 0` ≠ 版面没坏。双栏下过宽的表格会被排到
    #   栏外并与右栏文字**实际重叠**，日志里只有一条 Overfull（肉眼在缩略图里看不出来）。
    #   本门禁按 PDF 字形框实测碰撞——首次运行即查出 9 处真实碰撞（最严重 33.7pt），
    #   全部是先前就存在的表格。需要 drift_ce/paper/main.pdf 已编译；缺 PDF 时跳过。
    (".tools/_g2_pdf_layout_check.py", 600),
]

# ★ 2026-09-17 §4.9 步骤②「漂移闭环 × 可插拔底模」——不在 experiments/ 下，
#   故用独立列表跑；**eval 段不接 PostgreSQL**（只读已入库的池 pkl + 特征缓存 npz），
#   因此可以进一键复现。
#   前置产物（均已入库）：drift_ce/pool_imdb766.pkl（漂移态真值）+
#   drift_ce/results_repro/_e1_cache.npz（三态静态特征 250 维、MSCN/DeepDB 预测）。
#   口径与 §4.4 的 imdb 臂一致：T=100 / R=30 次独立划分 / 均值±95%CI。
#   臂定义与论文 §4.9 的"冻结 / 只重校准 / 修复"分类法逐字对齐：
#     裸底模 / 冻结(旧 q) / 只重校准 / 修复 C_drift / 修复 C_fresh
#   ★ 若只改了论文里的 ② 数字，重跑这一条 + 门禁即可（约 0.4 分钟）。
#   ★ 重建前置产物需要 PostgreSQL：
#     1) & postgres.exe -D <datadir> -p 55432 ...（**必须提权**，见 §4.11 说明）
#     2) python .tools/_c3_imdb_pool.py --workload "datas/datas/workloads/test/imdb/workloads.sql" \
#             --limit 300 --out drift_ce/pool_imdb766.pkl
#     3) python .tools/_e1_basemodel_drift_arms.py --stage prep   （特征 + MSCN/DeepDB 预测，不需 PG）
EXTRA_OFFLINE = [
    (".tools/_e1_basemodel_drift_arms.py",
     ["--stage", "eval", "--n-t", "100", "--reps", "30"], 1800),
]

# 每个实验的超时（秒）。CI 实验要跑 50 次重复，需要更长时间。
TIMEOUTS = {
    "experiment_correlation_realdata_ci.py": 2400,
    "experiment_detection_ci.py": 2400,
    "experiment_necessity_ci.py": 1800,
    "experiment_groupcoverage_ci.py": 2400,
    "experiment_correlation_ci.py": 2400,
    "experiment_ablation_ci.py": 3600,
    "experiment_modelagnostic_ci.py": 3600,
    "experiment_driftspectrum_ci.py": 3600,
    "experiment_baselines_ci.py": 2400,
    "experiment_crossdataset_ci.py": 5400,
    "experiment_crossdataset.py": 1800,
    "experiment_temporaldrift.py": 1800,
    "experiment_residual_necessity.py": 3600,
    "experiment_residual_sensitivity.py": 1800,
    "experiment_loop_decision.py": 1800,
}
DEFAULT_TIMEOUT = 900

# 需要真实数据、耗时较长（不默认跑）：experiment_realdata_drift.py
SLOW = ["experiment_realdata_drift.py"]

# ============================================================
# §4.4 真实数据端到端（B 系列）—— **不在本脚本内自动跑**
# ------------------------------------------------------------
# 它需要一台挂着真实 StackExchange `stats` 库的 PostgreSQL，
# 并且要真正删数据 + ANALYZE（b01 一步就要 ~13 分钟），不适合放进一键复现。
#
# 复现步骤（缺一不可）：
#   1) 起 PostgreSQL（本项目用 16.6，端口 55432），加载 stats 库；
#   2) python drift_ce/pg_drift_pool.py --only baseline,age_purge50_posts ...
#      ⇒ 产出 pool_age.pkl（**已入库**，所以第 3 步及之后可离线复现）；
#   3) python .tools/_b01_build_state.py / _b02_gen_features.py / _b03_sensitivity.py
#   4) python .tools/_b04_arms.py     → results_repro/b2_arms.npz（其余 b05–b08 读它）
#   5) python .tools/_b05_diag.py / _b06_repeat.py / _b07_degrade.py / _b08_why.py
#
# ★ 2026-09-16：b04–b08 原先默认读 <tmp>/（不在仓库里，
#   已失效）；现默认改为仓库内 drift_ce/pool_age.pkl，故 **b04–b08 可离线复现**。
#   已核实：_b05_diag.py 重跑输出与存档 b3_diag.txt 逐位一致。
# ============================================================

# ============================================================
# §4.11 下游 plan regret —— **不在本脚本内自动跑**（需真实 PostgreSQL）
# ------------------------------------------------------------
# 【为什么原产物不能用】论文 §4.11 的"稳健"一列来自
# `pg_experiment/experiment_plan_regret_expected.py`：
#     cost_true   = min(枚举 4 种 join 方法的实测耗时)
#     cost_robust = min(【重新枚举同一批】的实测耗时)   ⇒ 与 cost_true 是同一个量，比值恒为 1
#   即"稳健 median=1.00 / p90=1.05 / max=1.9"只是计时噪声。
#   （其"点估计"一列 median=1.08/p90=1.68/max=13.7 是**有效**测量，故只废掉稳健一列。）
#   详见 PLAN_PAPER_COMPLETION.md §11（发现）与 §12（修正后重跑）。
#
# 【正确设计的重跑】`.tools/_f2_planregret_realcost.py` —— 选择与评估**分离**：
#   * 选择：用【注入基数信念】下的 EXPLAIN 估计代价（点估计规则 / 区间 minimax 相对后悔）
#   * 评估：用【真数据】上的 EXPLAIN ANALYZE 实测耗时（GUC 强制计划形状）
#   注入机制**不需要 pg_hint_plan**（本机 PG16.6 上 Cybertec 的 REL16 1.6.1/1.6.2 两个 Windows
#   构建都不可用：单表 hint 静默失效、2 表 join 上必崩 0xC0000005）：
#     (a) UPDATE pg_statistic SET stadistinct=<绝对值>   把比例型 ndistinct 固定，否则 join 估计饱和
#     (b) UPDATE pg_class    SET reltuples=base*f        join 估计对 f 严格线性（实测 0.01×–100×）
#     (c) SET enable_hashjoin/enable_mergejoin/enable_nestloop   强制计划形状
#   候选集必须只用**固定形状**；"不强制(default)"不能进规则候选集（它是"让 PG 自己决定"，
#   在任何信念下都不劣于任何被强制形状 ⇒ 会让三条规则恒等）。
#
# 复现步骤：
#   1) 起 PostgreSQL（本项目 16.6，端口 55432），加载 stats 库；
#      ★ 必须提权运行：沙箱禁止 postmaster 给 backend 发信号，崩溃恢复会死于
#        "could not signal for checkpoint: Operation not permitted"。
#   2) & ".venv\Scripts\python.exe" -u ".tools\_f2_planregret_realcost.py" --nq 300 --reps 2 --drift 7.39
#      （另跑 --drift 1 作跨实验复现门禁；敏感性档用**共形自洽**的 --drift 16 --q 2.7726(=log 16)：
#        q < log D 时区间不覆盖真值，只能当对照，不能当口径；
#        --dump 6 可额外导出代价矩阵供机制图用）
#   3) 画图：python drift_ce/experiments/plot_fig12_planregret.py（仅画图不需要 PostgreSQL）
#   4) 汇总：python drift_ce/experiments/experiment_planregret_analysis.py（已在 EXPS，只读 npz）
# ============================================================

print("=" * 64)
print("DriftCE 可复现实验（固定种子 42，n_jobs=1）")
print(f"  快速集 {len(EXPS)} 个；慢集（需真实数据，不默认跑）{len(SLOW)} 个：{SLOW}")
print("=" * 64)

# 0) 主指标口径自检——任何重报之前必须先过这一关
print("\n########## metrics.py (q* 口径自检) ##########")
try:
    m = subprocess.run([sys.executable, os.path.join(DIR, "metrics.py")],
                       cwd=DIR, capture_output=True, text=True, timeout=120,
                       encoding="utf-8", errors="replace")
    print(m.stdout[-1200:])
    if m.returncode != 0:
        print(f"[FAILED self_check exit={m.returncode}]\n{m.stderr[-800:]}")
except Exception as e:
    print(f"[ERROR metrics.py: {str(e)[:160]}]")

# ★ 2026-09-19 修正：本节此前只打印 [FAILED]/[TIMEOUT]/[ERROR] 而**从不记录**，
#   且 SUITE_FAILS 是在这个循环**之后**才初始化的，末尾再汇总。
#   后果：快速集里任一脚本崩溃（退出码非 0）或脚本缺失时，run_all 末尾仍打印
#   "✅ 离线实验与全部门禁通过"——与 2026-09-17 修掉的 MISSING 静默失效同类。
SUITE_FAILS = []
for exp in EXPS:
    path = os.path.join(DIR, "experiments", exp)
    tmo = TIMEOUTS.get(exp, DEFAULT_TIMEOUT)
    print(f"\n########## {exp} ##########")
    if not os.path.exists(path):
        print(f"[MISSING {exp}]")
        SUITE_FAILS.append(exp + " (MISSING)")
        continue
    try:
        r = subprocess.run([sys.executable, path], cwd=DIR, capture_output=True, text=True, timeout=tmo, encoding="utf-8", errors="replace")
        txt = r.stdout + ("\n[STDERR Last]\n" + r.stderr[-1200:] if r.stderr.strip() else "")
        with open(os.path.join(OUT, exp.replace(".py", ".txt")), "w", encoding="utf-8") as f:
            f.write(f"exit={r.returncode}\n" + txt)
        print(r.stdout[-2200:])
        if r.returncode != 0:
            print(f"[FAILED exit={r.returncode}]")
            SUITE_FAILS.append(exp + f" (exit={r.returncode})")
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT {exp}]")
        SUITE_FAILS.append(exp + " (TIMEOUT)")
    except Exception as e:
        print(f"[ERROR {exp}: {str(e)[:120]}]")
        SUITE_FAILS.append(exp + " (ERROR)")

print("\n" + "=" * 64)
print(f"输出已存到 {OUT}/")

# 0.5) 不在 experiments/ 下的离线可复现实验（如 §4.9 步骤②）
REPO = os.path.dirname(DIR)
# ★ MISSING 也算失败：2026-09-17 发现"路径写成 tools/ 而实际是 .tools/"时，
#   run_all 只打印 MISSING 就继续，末尾仍然报"0 失败" ⇒ 门禁静默失效。
for rel, argv, tmo in EXTRA_OFFLINE:
    path = os.path.join(REPO, rel)
    print(f"\n########## [离线] {rel} {' '.join(argv)} ##########")
    if not os.path.exists(path):
        print(f"[MISSING {rel}]")
        SUITE_FAILS.append(rel + " (MISSING)")
        continue
    try:
        r = subprocess.run([sys.executable, path] + argv, cwd=REPO, capture_output=True,
                           text=True, timeout=tmo, encoding="utf-8", errors="replace")
        print(r.stdout[-2500:])
        if r.returncode != 0:
            print(f"[FAILED exit={r.returncode}]\n{r.stderr[-800:]}")
            SUITE_FAILS.append(rel + f" (exit={r.returncode})")
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT {rel}]")
        SUITE_FAILS.append(rel + " (TIMEOUT)")
    except Exception as e:
        print(f"[ERROR {rel}: {str(e)[:160]}]")
        SUITE_FAILS.append(rel + " (ERROR)")

# 1) 论文数字 ↔ 产物 核对门禁（只读 npz）
for rel, tmo in VERIFY_GATES:
    path = os.path.join(REPO, rel)
    print(f"\n########## [门禁] {rel} ##########")
    if not os.path.exists(path):
        print(f"[MISSING {rel}]")
        SUITE_FAILS.append(rel + " (MISSING)")
        continue
    try:
        r = subprocess.run([sys.executable, path], cwd=REPO, capture_output=True, text=True,
                           timeout=tmo, encoding="utf-8", errors="replace")
        print(r.stdout[-2500:])
        if r.returncode != 0:
            print(f"[GATE FAILED exit={r.returncode}]\n{r.stderr[-800:]}")
            SUITE_FAILS.append(rel + f" (exit={r.returncode})")
        else:
            print("[GATE PASSED]")
    except Exception as e:
        print(f"[ERROR {rel}: {str(e)[:160]}]")
        SUITE_FAILS.append(rel + " (ERROR)")

print("\n" + "=" * 64)
if SUITE_FAILS:
    print(f"❌ 离线实验/门禁失败 {len(SUITE_FAILS)} 项：{SUITE_FAILS}")
else:
    print("✅ 离线实验与全部门禁通过")
print("=" * 64)
