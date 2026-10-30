#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""云上运行：生成【真实数据漂移池】——可携带回本地做 q*(alpha) 分析。

设计原则
--------
1. **只做贵的事**：把"造数据 + 重算真值"一次性做完，落盘成池；策略/检测实验全部回本地跑。
2. **漂移必须是真的**：用 SQL 真改数据（值置换 / 行删除），再 ANALYZE，再重新执行查询取真值。
   不是"对真值乘常数"。
3. **漂移对准强关联列对**：保边际的值置换【只】会改变"多列合取"查询的真值，
   幅度取决于两列原本有多相关。配置表按本地算出的关联强度排序（见下）。
4. **可重复**：`setseed()` 固定随机置换；每个配置前用建库脚本整库回滚。

用法（云实例上）
----------------
  python3 pg_drift_pool.py --data-dir ~/stats_data \
      --workload stats_multi_table.tsv --limit 2000 --out pool_stats.pkl
  # 只看配置与 SQL，不跑：
  python3 pg_drift_pool.py --dry-run

依赖：psycopg2（cloud_setup.sh 已装）。不需要 numpy。
本地产物消费：把 pool_*.pkl 拷回本地，用 drift_ce 的 q* 框架分析。
"""
import argparse, os, re, subprocess, sys, time, pickle

# 统一 UTF-8 输出（与 drift_ce 其它脚本一致；Windows 控制台/管道下中文不乱码）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# psycopg2 延迟导入：使 --dry-run 与查询解析可在本地（无 psycopg2）验证

# ----------------------------------------------------------------- 漂移配置
# 关联强度来自本地分析（关联强度 x 共同查询数 排序）：
#   posts: answercount + viewcount        关联 1.541, 共同查询 20   <- 最强
#   posts: creationdate + score           关联 0.715, 共同查询 19
#   posthistory: creationdate + typeid    关联 0.185, 共同查询 27
# ------------------------------------------------------------------ 统计量口径
# ★ 为什么必须把 default_statistics_target 调高（本轮两次云端冒烟后定位到的真因）
#
#   现象：三个 reanalyze=False 的置换状态，pg_est 仍有 19/20 条变化（中位 3-15%）。
#   真因：每个状态开头的 [restore] 会跑 reload 脚本，而该脚本【每张表 \copy 之后都跟着
#         analyze】。default_statistics_target=100 时 ANALYZE 只抽样 30000 行 ->
#           baseline : reload 的 ANALYZE(样本1) + 显式 ANALYZE(样本2)  -> 用样本2
#           perm_*   : 只有 reload 的 ANALYZE(样本1)                  -> 用样本1
#         两个状态用【不同的随机样本】，差出来的就是采样噪声。
#         本地实测 target=100 时 43 条统计量里【35 条不一致】、Plan Rows 7/8 条变化。
#
#   修法：把 target 提到 300*target >= 表行数，ANALYZE 退化为【全表扫描】-> 统计量确定性。
#         最大表 votes = 328064 行，300*2000 = 600000 覆盖全部表。
#         本地实测 target=2000 时 43 条统计量【0 条不一致】、Plan Rows 0/8 条变化。
#         保边际置换不改边缘分布，所以全表扫描得到的统计量必然逐位相同。
#
#   注意：这个 GUC 必须同时作用于【两处】——
#     1) pool 自己的连接（prep()），作用域内会话里的显式 ANALYZE；
#     2) restore() 拉起的那个 psql 子进程（用 PGOPTIONS），作用域是 reload 脚本自带的 analyze。
#   只设一处都不够（这正是第一版修复没生效的原因之一）。
STATS_TARGET = int(os.environ.get("PG_STATS_TARGET", "2000"))

PERM_TMPL = """
UPDATE {table} SET {col} = s.{col}
FROM (SELECT {pk}, row_number() OVER (ORDER BY {pk}) AS rn
      FROM {table} WHERE {where}) r
JOIN (SELECT row_number() OVER (ORDER BY random()) AS rn, {col}
      FROM {table} WHERE {where}) s
  ON r.rn = s.rn
WHERE {table}.{pk} = r.{pk};
"""

def perm_sql(table, col, where="TRUE", pk="id"):
    return PERM_TMPL.format(table=table, col=col, where=where, pk=pk)


DRIFT_CONFIGS = [
    # (名称, 漂移 SQL, 特征是否仍有效, 是否重新 ANALYZE, 是否 VACUUM FULL)
    #
    #   保边际的值置换【不改统计量(直方图/size/fanout)】-> 622 维特征逐位不变 -> 可直接复用本地特征
    #   尺度漂移【改统计量】-> 特征必须重算；而本地 datas/datas/statistics/finetune/stats/ 缺
    #     schema_graph/col_semantic/table_semantic/col_stats_vec/table_meta 五个文件，
    #     无法本地重跑 Sql2Feature -> 该状态标记为特征无效（仅做"无需特征"的诊断：真值变化分布 + q*）
    #
    #   ★ reanalyze：保边际置换后 PG 的统计量【在数学上不变】（每列边缘分布逐位保留），
    #     此时再 ANALYZE 只会引入【采样噪声】，把"统计量不变 => 特征可复用"这个前提污染掉。
    #     所以置换状态不重新 ANALYZE，使唯一允许改变的东西就是真值本身。
    #
    #   ★ vacuum_full（本轮云端冒烟后新增，修掉一个真实存在的混淆）：
    #     只保证 pg_statistic 不变【还不够】。规划器的行数估计还依赖 pg_class.relpages/reltuples
    #     与表【当前的物理块数】—— estimate_rel_size() 用"上次 ANALYZE 的密度 x 当前文件块数"外推。
    #     而 UPDATE 是原地重写：留下大量死元组、把文件撑大 -> 估算行数跟着翻倍。
    #     实测：云端冒烟里三个 reanalyze=False 的状态，pg_est 中位变化 3-15%、最大 4 倍
    #     —— 这些变化【不可能来自漂移】。另 VACUUM FULL 还会把 AUTOANALYZE 的触发计数清零。
    #     因此非 baseline 状态一律 VACUUM FULL：表体回到紧凑、pg_class 被实测更新、
    #     而 pg_statistic 仍然【一个字节都不动】。
    #   ★ reanalyze：每个状态开头都会 [restore]，而 reload 脚本【每张表都自带 analyze】，
    #     所以所有状态的统计量都来自 reload 那一次。置换保边缘分布 -> 那次 analyze 的结果
    #     对置换后的表依然有效（且因为 STATS_TARGET 走全表扫描而【逐位相同】），
    #     故 baseline 与置换状态都不需要再 ANALYZE；只有 scale_del 真的改了分布，才必须重算。
    #
    #   ★ vacuum_tables：UPDATE/DELETE 是原地写，留死元组把文件撑大，而规划器用
    #     「上次 ANALYZE 的密度 x 当前文件块数」外推行数 -> 估算会凭空涨（实测 +44%）。
    #     VACUUM FULL 把表体重写紧凑并实测更新 pg_class，但【不触碰 pg_statistic】。
    #     ★ 只 VACUUM【被漂移改动的那张表】，不要全库 VACUUM：
    #       全库 VACUUM 会把未改动的表也重写一遍，凭空引入与漂移无关的差异；
    #       只动被改的那张，其余表就与 baseline 逐字节相同（同一份 \copy 的产物）。
    ("baseline", [], True, False, []),
    ("perm_posts_answercount", [perm_sql("posts", "answercount", "posttypeid = 1")], True, False, ["posts"]),
    ("perm_posts_score", [perm_sql("posts", "score", "posttypeid = 1")], True, False, ["posts"]),
    ("perm_ph_typeid", [perm_sql("posthistory", "posthistorytypeid", "TRUE")], True, False, ["posthistory"]),
    #   ★ perm_creationdate（本轮新增）：独立审计发现前三个目标列只被 3~11% 的查询引用，
    #     效应被严重稀释。creationdate 被 265/500 条查询引用（53%），且 posts.creationdate
    #     与物理顺序的相关性高达 -0.9858，置换会破坏极强的结构。
    #     仍是【保边际】置换 -> histogram40 不变 -> 622 维特征逐位不变 -> 可直接复用
    #     （因此【不需要】那 5 个缺失的统计量文件）。
    #     预期：受影响查询从 5~11% 升到约 50%，用于检验"总量 = 受影响比例 x 单条损伤"。
    ("perm_creationdate",
     [perm_sql(t, "creationdate", "TRUE") for t in
      ("comments", "posthistory", "postlinks", "posts", "users", "votes")],
     True, False,
     ["comments", "posthistory", "postlinks", "posts", "users", "votes"]),
    ("scale_del50_posts", ["DELETE FROM posts WHERE random() < 0.5;"], False, True, ["posts"]),
    # ============================ B 实验新增（形状改变漂移） ============================
    #   ★ 为什么要这个状态：A 实验的三种漂移都【没有真正改变任何一列的边缘形状】——
    #       perm_*      保边缘分布（数学上逐位不变）
    #       scale_del50 随机删 50%（等比缩放，形状不变）
    #     于是 drift_ce 真正使用的 622 维特征里，数据依赖的那部分几乎没动，
    #     「特征陈旧」这个变量从未被触发。
    #
    #   ★ age_purge50：删【最老的】⌊n/2⌋ 行（按 (creationdate, id) 全序）。
    #      删除比例与 scale_del50 完全相同（50%），但边缘形状大幅改变：
    #        creationdate 形状 L1 = 0.4994（随机删只有 0.0142 -> 35.2x）
    #        answercount  形状 L1 = 1.1374（随机删      0.4477 ->  2.5x）
    #      两者配对 -> 差值直接隔离出「形状改变」这一分量。
    #
    #   ★ 确定性：排序键 (creationdate, id) 是全序，无并列歧义、不用随机种子。
    #      本地 pandas 侧用同一条规则（sort_values(['creationdate','id']) 后丢前 n//2 行），
    #      实测 posts.csv 行数 91976 -> 删 45988 -> 留 45988，与 PG 完全一致
    #      （阈值处恰有 1 行列并列，PG 侧 creationdate < 1366544339 删 45987、
    #        再按 id 删 1 行补足，与 pandas 的全序结果相同）。
    #
    #   ★ 特征必须重算（feature_valid=False）：本地已用【原版 gen_*.py】
    #      对漂移后的 stats_age 数据集重跑了 histogram40/fanout40/summary40/size，
    #      产物在 drift_ce/_mirror_age/。实测 194/300 条查询的 622 维特征发生改变，
    #      其中 fan_max 最大偏移 1.9765e+03 —— 这正是 B 要研究的变量。
    #
    #   ★ reanalyze=True + VACUUM FULL posts：删了 50% 的行，分布与 relpages 都真的变了，
    #      必须让 PG 自己的统计量刷新（这是「ANALYZE 便宜、所以给 PG 估算器应有的待遇」
    #      这一前提），才能把「特征陈旧」单独隔离出来。
    ("age_purge50_posts",
     ["DELETE FROM posts WHERE id IN ("
      "SELECT id FROM posts ORDER BY creationdate, id "
      "LIMIT (SELECT count(*) / 2 FROM posts));"],
     False, True, ["posts"]),
]


# ----------------------------------------------------------------- 工具
def load_queries(path, limit=None):
    """返回 [(line_idx, sql)]，line_idx 为【非空行】的 0-based 序号。

    ⚠️ 与本地特征的对齐依据：已本地验证
        `unpublished internal prototype` 的 features40_s3a.pkl 条目 i
        与 `datas/datas/workloads/finetune/stats/workloads.sql` 第 i 行（非空行序号）【精确一一对应】
        （50000/50000 逐条比对，0 不一致）。
    因此云上只要按同一个文件、同样的非空行序号输出，本地就能直接取到 622 维特征。
    支持两种格式：`sql || true [|| pg_est || tag]`（finetune/workloads.sql，4 列）
                 或 `sql<TAB>true`（stats_multi_table.tsv）。
    """
    out, idx = [], 0
    with open(os.path.expanduser(path), encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("--"):
                continue
            parts = line.split("\t") if "\t" in line else line.split("||")
            sql = re.sub(r"^/\*[^*]*\*/\s*", "", parts[0]).strip().rstrip(";").strip()
            if not sql:
                continue
            out.append((idx, sql))
            idx += 1
            if limit and len(out) >= limit:
                break
    return out


def connect(a):
    try:
        import psycopg2
    except ImportError:
        print("缺少 psycopg2：pip3 install --user psycopg2-binary", file=sys.stderr)
        raise
    c = psycopg2.connect(dbname=a.db, user=a.user, password=a.password,
                         host=a.host, port=a.port, connect_timeout=10)
    c.autocommit = True
    return c


def make_reload_sql(data_dir, log):
    """由 postgres_create_stats.sql 生成【只重建表、不重建数据库】的加载脚本。

    动机：建库脚本开头是 `drop database if exists stats`，而 DROP DATABASE 会
    RequestCheckpoint(WAIT)，在受限环境（禁止向 checkpointer 发信号）下直接报
    `could not signal for checkpoint: Operation not permitted`；且只要本进程还连着该库
    就会报 "being accessed by other users"。改成"重建表"可以**同时消除这两类失败**，
    语义等价 —— 该库本来就只承载这 8 张表。

    返回生成的文件名（相对 data_dir）；失败返回 None。
    """
    import re
    d = os.path.expanduser(data_dir)
    src = os.path.join(d, "postgres_create_stats.sql")
    if not os.path.exists(src):
        log(f"  [reload] 找不到 {src}")
        return None
    out_name = "postgres_reload_stats.sql"
    dst = os.path.join(d, out_name)
    # 只剥离【库级】语句；`\c stats` 必须保留 —— 它只是客户端重连到目标库，
    # 不触发 checkpoint，剥掉就会把表建到默认库（postgres）里。
    skip = re.compile(r"^\s*(drop\s+database|create\s+database)\b", re.I)
    kept, dropped = [], 0
    with open(src, encoding="utf-8", errors="replace") as f:
        for line in f:
            if skip.match(line):
                dropped += 1
                continue
            kept.append(line)

    # 关闭这 8 张表的自动清理/自动统计。
    #   为什么必须关：保边际置换会改动表里 47%~100% 的行，far 超过 autovacuum 的
    #   "10% + 50 行" 自动统计阈值 -> AUTOANALYZE 会在我们测完之后自己跑一遍，
    #   用【新样本】重估统计量，把"统计量不变"这个待验证前提直接破坏掉。
    #   注意 autovacuum_enabled=false 只关自动的；手动 VACUUM / ANALYZE 照常可用。
    kept.append("\n-- 关闭自动清理/自动统计：防止 AUTOANALYZE 在漂移后自行重估统计量\n")
    for t in ("badges", "comments", "posthistory", "postlinks",
              "posts", "tags", "users", "votes"):
        kept.append(f"ALTER TABLE {t} SET (autovacuum_enabled = false);\n")

    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(kept)
    log(f"  [reload] 由 {os.path.basename(src)} 生成 {out_name}"
        f"（剥离 {dropped} 行库级语句；只重建表；并对 8 张表关闭 autovacuum）")
    return out_name


def ensure_db(a, log):
    """若目标库不存在则创建（DROP DATABASE 不可用时无法靠建库脚本兜底）。"""
    try:
        import psycopg2
        c = psycopg2.connect(dbname=a.db, user=a.user, password=a.password,
                             host=a.host, port=a.port, connect_timeout=10)
        c.close()
        return True
    except Exception:
        pass
    try:
        import psycopg2
        from psycopg2 import sql as _sql
        c = psycopg2.connect(dbname="postgres", user=a.user, password=a.password,
                             host=a.host, port=a.port, connect_timeout=10)
        c.autocommit = True
        cur = c.cursor()
        cur.execute(_sql.SQL("CREATE DATABASE {}").format(_sql.Identifier(a.db)))
        c.close()
        log(f"  [ensure_db] 已创建数据库 {a.db}")
        return True
    except Exception as e:
        log(f"  [ensure_db][FAIL] {str(e)[:200]}")
        return False


def prep(cur, timeout_s):
    cur.execute("SET statement_timeout=%s;", (int(timeout_s * 1000),))
    cur.execute("SET work_mem='128MB';")
    # 让本会话里的 ANALYZE 也扫全表（见 STATS_TARGET 的说明）
    cur.execute("SET default_statistics_target=%s;", (STATS_TARGET,))
    cur.execute("SELECT setseed(0.42);")          # 固定随机置换 -> 可重复


def pg_estimate(cur, sql):
    q = re.sub(r"(?i)SELECT\s+COUNT\(\*\)\s+FROM", "SELECT * FROM", sql, count=1)
    cur.execute("EXPLAIN (FORMAT JSON) " + q)
    return float(cur.fetchone()[0][0]["Plan"]["Plan Rows"])


def restore(a, log):
    """整库回滚：重跑【只重建表】的加载脚本（幂等）。

    注意：调用前必须已断开本进程到该库的连接，否则 `drop table` 会等锁。
    psql 默认【遇错继续且返回 0】，所以命令里带 -v ON_ERROR_STOP=1；
    回滚后再查一次 posts/posthistory 行数做实证校验（不能只看返回码）。

    ★ 默认用【当前用户】走 TCP 连 postgres，而不是 `sudo -u postgres psql`：
      `\\copy` 是 psql 客户端读文件，读文件用的是 psql 进程的身份。
      若以 postgres 用户跑 psql，而数据放在 /root 下（/root 权限 0700），
      postgres 无法穿越 /root -> `\\copy` 直接 Permission denied。
      以当前用户跑，则读文件权限与"谁上传的数据"一致，放哪都能读。
    """
    env = None
    if a.restore_cmd:
        cmd = a.restore_cmd.split()          # 显式指定（本地 Windows 测试用）
        env = dict(os.environ, PGOPTIONS=f"-c default_statistics_target={STATS_TARGET}")
    else:
        cmd = ["psql", "-h", str(a.host), "-p", str(a.port), "-U", a.user,
               "-v", "ON_ERROR_STOP=1", "-f", "postgres_reload_stats.sql"]
        env = dict(os.environ, PGPASSWORD=str(a.password),
                   # ★ 关键：reload 脚本自带 analyze，必须让【那个 psql 会话】也扫全表，
                   #   否则重载后的统计量仍带采样噪声（第一版修复就是漏了这一处）。
                   PGOPTIONS=f"-c default_statistics_target={STATS_TARGET}")
    log(f"  [restore] {' '.join(cmd)}  (cwd={a.data_dir})")
    r = subprocess.run(cmd, cwd=os.path.expanduser(a.data_dir),
                       capture_output=True, text=True, timeout=1800, env=env)
    if r.returncode != 0:
        log(f"  [restore][FAIL] rc={r.returncode}\n{r.stderr[-800:]}")
        return False
    # 实证校验：新连一次，确认真值表确实被重建且非空
    try:
        import psycopg2
        c2 = psycopg2.connect(dbname=a.db, user=a.user, password=a.password,
                              host=a.host, port=a.port, connect_timeout=10)
        c2.autocommit = True
        cur2 = c2.cursor()
        cur2.execute("SELECT count(*) FROM posts;")
        n = cur2.fetchone()[0]
        cur2.execute("SELECT count(*) FROM posthistory;")
        n2 = cur2.fetchone()[0]
        c2.close()
        log(f"  [restore] 校验 posts={n} posthistory={n2}")
        if n == 0:
            log("  [restore][FAIL] posts 为空，回滚未生效")
            return False
    except Exception as e:
        log(f"  [restore][FAIL] 校验查询失败: {str(e)[:200]}")
        return False
    return True


def run_config(cur, conn, a, sqls, log):
    """在当前数据状态上：执行全部查询取真值 + EXPLAIN 取 pg_est。

    异常分三类，必须区别对待（上一版把它们混在一起，日志会误导人）：
      1) QueryCanceled —— statement_timeout 到点。
         ★ 陷阱：psycopg2 里 QueryCanceled(SQLSTATE 57014) **继承自 OperationalError**，
           所以只 catch OperationalError 会把它误标成"断线"，还会白白重连一次。
         autocommit 下被取消的语句不会毒化会话，**无需重连**，连接可继续用。
      2) OperationalError —— 真的连接断了，必须重连。
      3) 其它 Exception —— 语法/类型等单条错误，跳过即可。
    """
    import psycopg2
    trues, pgs, ok = [], [], []
    t0 = time.time()
    for i, sql in enumerate(sqls):
        try:
            cur.execute(sql)
            t = float(cur.fetchone()[0])
            p = pg_estimate(cur, sql)
            trues.append(t); pgs.append(p); ok.append(True)
        except psycopg2.errors.QueryCanceled:
            log(f"    [超时 {i}] 单条超过 {a.timeout}s，跳过（连接可继续用，不重连）")
            trues.append(None); pgs.append(None); ok.append(False)
        except psycopg2.OperationalError as e:
            log(f"    [断线 {i}] 真连接断开，重连并跳过: {str(e)[:60]}")
            try:
                conn.close()
                conn = connect(a)
                cur = conn.cursor(); prep(cur, a.timeout)
            except Exception:
                pass
            trues.append(None); pgs.append(None); ok.append(False)
        except Exception as e:
            if a.verbose:
                log(f"    [跳过 {i}] {str(e)[:70]}")
            trues.append(None); pgs.append(None); ok.append(False)
        if (i + 1) % 200 == 0:
            log(f"    {i+1}/{len(sqls)}  ({time.time()-t0:.0f}s)")
    return trues, pgs, ok, time.time() - t0, conn, cur


# ----------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("PGDATABASE", "stats"))
    ap.add_argument("--host", default=os.environ.get("PGHOST", "localhost"))
    ap.add_argument("--port", default=int(os.environ.get("PGPORT", "5432")), type=int)
    ap.add_argument("--user", default=os.environ.get("PGUSER", "postgres"))
    ap.add_argument("--password", default=os.environ.get("PGPASSWORD", "postgres"))
    ap.add_argument("--data-dir", default="~/stats_data",
                    help="含 postgres_create_<db>.sql 与 *.csv 的目录（restore 用）")
    ap.add_argument("--workload", default="stats_multi_table.tsv")
    ap.add_argument("--limit", default=2000, type=int, help="查询条数上限（0=不限）")
    ap.add_argument("--timeout", default=30.0, type=float, help="单条查询超时(秒)")
    ap.add_argument("--restore-cmd", default=None,
                    help="回滚命令覆盖（相对 data-dir 执行）。默认不用它：自动用 psql 以当前用户走 "
                         "TCP 执行 postgres_reload_stats.sql —— 这样 \\copy 以当前用户身份读文件，"
                         "数据放在 /root 下也能读。")
    ap.add_argument("--out", default="pool_stats.pkl")
    ap.add_argument("--only", default=None,
                    help="只跑指定状态（逗号分隔），如 --only baseline,perm_creationdate。"
                         "用于只补一个新状态，避免重跑全部（每个状态开头都会独立重载，"
                         "所以与既有池的 baseline 可比）。")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-restore", action="store_true", help="不自动回滚（调试用）")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    def log(m):
        print(m, flush=True)

    # --only：只跑指定状态（每个状态开头都会独立重载，故与既有池的 baseline 可比）
    cfgs = DRIFT_CONFIGS
    if a.only:
        want = [x.strip() for x in a.only.split(",") if x.strip()]
        cfgs = [c for c in DRIFT_CONFIGS if c[0] in want]
        missing = [w for w in want if w not in [c[0] for c in DRIFT_CONFIGS]]
        if missing:
            log(f"[FAIL] --only 里有未知状态: {missing}")
            return

    log("=" * 92)
    log(f"真实数据漂移池 | db={a.db} @ {a.host}:{a.port} | 配置 {len(cfgs)} 个 | 单条超时 {a.timeout}s")
    if a.only:
        log(f"  --only 生效，本次只跑: {[c[0] for c in cfgs]}")
    log("=" * 92)

    if a.dry_run:
        for name, stmts, fv, ra, vt in cfgs:
            log(f"\n[{name}]  feature_valid={fv}  reanalyze={ra}  vacuum={vt or '无'}")
            for s in stmts:
                log("  " + " ".join(s.split())[:400])
        log("\n(--dry-run 结束，未连接数据库)")
        return

    sqls_pairs = load_queries(a.workload, a.limit or None)
    idxs = [i for i, _ in sqls_pairs]
    sqls = [s for _, s in sqls_pairs]
    log(f"workload={a.workload} -> {len(sqls)} 条；行号范围 [{idxs[0]}, {idxs[-1]}]")
    log("  行号与本地 features40_s3a.pkl 条目一一对应（已本地验证 50000/50000）")

    # 生成"只重建表"的加载脚本；建库（DROP DATABASE 不可用的环境下也能跑）
    make_reload_sql(a.data_dir, log)
    if not ensure_db(a, log):
        return
    # 先做一次完整加载：保证 baseline 状态是【干净且完整】的，而不是继承上一次的残留
    if not a.skip_restore:
        log("\n[初始化] 全量加载数据 -> 作为 baseline 状态")
        if not restore(a, log):
            log("[中止] 初始加载失败（数据未就绪，继续跑没有意义）")
            return

    conn = connect(a); cur = conn.cursor(); prep(cur, a.timeout)
    pool = {"meta": dict(db=a.db, host=a.host, workload=a.workload, n_queries=len(sqls),
                         configs=[n for n, _, _, _, _ in cfgs],
                         feature_valid={n: fv for n, _, fv, _, _ in cfgs},
                         reanalyze={n: ra for n, _, _, ra, _ in cfgs},
                         vacuum_tables={n: vt for n, _, _, _, vt in cfgs},
                         stats_target=STATS_TARGET,
                         stats_target_note="ANALYZE 扫全表(300*target>=表行数) -> 统计量确定性，"
                                            "消除跨状态采样噪声",
                         timeout=a.timeout,
                         idx_semantics="0-based index of non-empty line in workload file; "
                                       "aligns with features40_s3a.pkl entry index (verified)",
                         ts=time.strftime("%Y-%m-%d %H:%M:%S")),
            "idx": idxs, "sqls": sqls, "states": {}}

    for name, stmts, fv, ra, vt in cfgs:
        log(f"\n########## {name} ##########")
        log(f"  特征是否可复用本地 622 维: {fv} | 重新 ANALYZE: {ra} | VACUUM FULL: {vt or '无'}")
        if name != "baseline" and not a.skip_restore:
            # 必须先断开自己的连接：建库脚本第一句是 `drop database if exists stats`，
            # 只要本进程还连着 stats，就会报 "database stats is being accessed by other users"。
            try:
                cur.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            if not restore(a, log):
                log("  [跳过该配置：回滚失败]"); continue
            conn = connect(a); cur = conn.cursor(); prep(cur, a.timeout)
        for s in stmts:
            t = time.time()
            cur.execute(s)
            log(f"  漂移 SQL 完成 ({time.time()-t:.0f}s): {' '.join(s.split())[:100]}...")
        for tb in vt:
            # 原地 UPDATE/DELETE 会留死元组、把文件撑大，规划器据此把行数估高。
            # VACUUM FULL 把表体重写紧凑并实测更新 pg_class（relpages/reltuples），
            # 但【不触碰 pg_statistic】—— 消掉物理假象，保留统计量不变。
            t = time.time()
            cur.execute(f"VACUUM (FULL) {tb};")
            log(f"  VACUUM FULL {tb} 完成 ({time.time()-t:.0f}s)：消除死元组膨胀，pg_statistic 未动")
        if ra:
            cur.execute("ANALYZE;")
            log("  ANALYZE 完成；执行查询...")
        else:
            log("  跳过 ANALYZE（保边际置换不改统计量，重算只会引入采样噪声）；执行查询...")
        trues, pgs, ok, secs, conn, cur = run_config(cur, conn, a, sqls, log)
        pool["states"][name] = dict(true=trues, pg_est=pgs, ok=ok, secs=secs)
        log(f"  成功 {sum(ok)}/{len(ok)}；耗时 {secs:.0f}s")

    with open(a.out, "wb") as f:
        pickle.dump(pool, f)
    log(f"\n已存 {a.out}（{os.path.getsize(a.out)/1e6:.1f} MB）—— 拷回本地做 q* 分析")
    log("提示：相关性漂移下统计量(直方图)不变 -> 622 维特征不变，本地可按 SQL 匹配复用现有特征。")


if __name__ == "__main__":
    main()
