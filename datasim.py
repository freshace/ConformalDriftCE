# -*- coding: utf-8 -*-
"""datasim —— 本地真实数据漂移引擎（sqlite3，无服务、无云）。

为什么需要它
------------
现有漂移是【标签级注入】：drift.py 对"过滤列数>=2"的查询把真值乘 exp(mu)。
这是一个【二元条件 × 常数】，而该条件变量（过滤列数）就在 622 维特征里
-> 修正器只需学一个阶梯函数 -> "500 条标签就够"是注入方式的算术结果（循环论证）。

本模块改为【真数据漂移】：
  1. 从 datas/datas/datasets/<db>/*.csv 把真实表载入 sqlite（真值可执行、可复算）；
  2. 施加数据级漂移（保持边际不变地置换某列 = 断开列间相关性）；
  3. 在漂移后的数据上【重新执行查询】得到新的真基数。

于是：
  - 真值是"真数据上算出来的"，不是乘出来的；
  - 边际（以及依赖边际的基模估计 pg_est）在相关性漂移下【可验证地不变】；
  - 修正量随查询【非线性、值依赖】变化 —— 不再是常数（消除循环论证）。

与既有资产的关系（不重造）
  - 统计量抽取：price_original/code/utils/statistics/gen_*.py（直接从 CSV 算，不依赖 PG）
  - 特征构造：unpublished internal prototype 的特征提取 + feature_tool.Sql2Feature
  - 底模 pg_est：数据集 workload 文件里的第 3 列（真 PG 产出）
"""
import os, re, sqlite3, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "datas", "datas")


# ---------------------------------------------------------------- 载入真实数据
def _copy_directives(ds_dir, db):
    sql_file = os.path.join(ds_dir, f"postgres_create_{db}.sql")
    out = []
    with open(sql_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"\s*\\copy\s+(\S+)\s+from\s+'([^']+)'", line, re.I)
            if m:
                out.append((m.group(1), m.group(2)))
    return out


class StatsDB:
    """sqlite 内存库，承载一个真实数据库的全部表。"""

    def __init__(self, db="stats", data_dir=None, verbose=True, query_timeout_s=3.0):
        self.db = db
        self.ds_dir = os.path.join(data_dir or DATA_DIR, "datasets", db)
        self.con = sqlite3.connect(":memory:")
        self.tables = {}
        self.timeout_s = float(query_timeout_s)
        self._deadline = None
        # 查询级硬超时：progress handler 每 10 万条 VM 指令回调一次，超时返回非 0 中断
        self.con.set_progress_handler(self._abort_check, 100000)
        self._load(verbose)

    def _abort_check(self):
        if self._deadline is None:
            return 0
        return 1 if time.time() > self._deadline else 0

    def _set_deadline(self, seconds=None):
        self._deadline = time.time() + (self.timeout_s if seconds is None else seconds)

    def _load(self, verbose):
        t0 = time.time()
        for tbl, fn in _copy_directives(self.ds_dir, self.db):
            p = os.path.join(self.ds_dir, fn.lstrip("./"))
            df = pd.read_csv(p, sep="|", quotechar='"', escapechar="\\",
                             keep_default_na=False, na_values=["NULL"], low_memory=False)
            df.columns = [c.lower() for c in df.columns]
            df.to_sql(tbl, self.con, index=False, if_exists="replace")
            self.tables[tbl] = df.shape
        if verbose:
            print(f"[datasim] {self.db}: {len(self.tables)} 表, {sum(v[0] for v in self.tables.values())} 行, "
                  f"载入 {time.time()-t0:.1f}s")

    # ------------------------------------------------------------ 索引
    def ensure_indices(self, sqls, verbose=False):
        """在 workload 用到的连接键上建索引（幂等）。"""
        return create_join_indices(self.con, sqls, verbose)

    # ------------------------------------------------------------ 执行查询
    def exec_count(self, sql):
        self._set_deadline()
        cur = self.con.cursor()
        cur.execute(sql)
        return float(cur.fetchone()[0])

    def exec_many(self, sqls, verbose=False):
        """返回 list[float|None]；超时/失败的记 None。"""
        cur = self.con.cursor()
        out = []
        for i, sql in enumerate(sqls):
            self._set_deadline()
            try:
                cur.execute(sql)
                out.append(float(cur.fetchone()[0]))
            except Exception as e:
                if verbose:
                    print(f"  [{i}] 失败/超时: {str(e)[:80]}")
                out.append(None)
        self._deadline = None
        return out

    def run_timed(self, sqls):
        """返回 (val, ms)；超时/失败 val=None。"""
        cur = self.con.cursor()
        out = []
        for sql in sqls:
            self._set_deadline()
            t = time.time()
            try:
                cur.execute(sql)
                v = float(cur.fetchone()[0])
            except Exception:
                v = None
            out.append((v, (time.time() - t) * 1000.0))
        self._deadline = None
        return out

    # ------------------------------------------------------------ 边际快照
    def column_marginal(self, table, col, bins=40):
        """返回该列的（精确）边际直方图与基数——用于验证"边际是否不变"。

        CSV 里数值列以文本读入，故用 pd.to_numeric 强转（PDF 里无值则丢弃）。
        """
        cur = self.con.cursor()
        cur.execute(f'SELECT "{col}" FROM "{table}"')
        vals = pd.to_numeric(pd.Series([r[0] for r in cur.fetchall()]), errors="coerce").dropna().values
        if vals.size == 0:
            return None
        hist, edges = np.histogram(vals, bins=bins)
        return dict(hist=hist.astype(float), edges=edges, n=int(vals.size),
                    mean=float(vals.mean()), std=float(vals.std()))

    def row_counts(self):
        cur = self.con.cursor()
        out = {}
        for t in self.tables:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            out[t] = int(cur.fetchone()[0])
        return out

    # ------------------------------------------------------------ 漂移算子
    def permute_column(self, table, col, frac=1.0, where=None, seed=42):
        """值置换：在 where 选中的行内随机重排 col 的取值。

        frac=1 时边际【严格不变】（只是行的排列），列间相关性被断开；
        0<frac<1 时按比例部分置换，构成漂移强度旋钮。
        """
        rng = np.random.RandomState(seed)
        cond = f" WHERE {where}" if where else ""
        cur = self.con.cursor()
        cur.execute(f"SELECT rowid, CAST({col} AS INTEGER) FROM {table}{cond}")
        rows = cur.fetchall()
        rids = np.array([r[0] for r in rows], dtype=np.int64)
        vals = np.array([r[1] for r in rows], dtype=np.int64)

        perm = rng.permutation(len(vals))
        swapped = vals[perm]
        if frac >= 1.0:
            newv = swapped
        else:
            take = rng.rand(len(vals)) < frac
            newv = np.where(take, swapped, vals)

        cur.execute("CREATE TEMP TABLE IF NOT EXISTS _perm(rowid_ INTEGER PRIMARY KEY, v INTEGER)")
        cur.execute("DELETE FROM _perm")
        cur.executemany("INSERT INTO _perm(rowid_, v) VALUES (?,?)",
                        list(zip(rids.tolist(), newv.tolist())))
        cur.execute(f"UPDATE {table} SET {col} = (SELECT v FROM _perm WHERE _perm.rowid_ = {table}.rowid) "
                    f"WHERE rowid IN (SELECT rowid_ FROM _perm)")
        self.con.commit()
        n_moved = int((newv != vals).sum())
        return dict(table=table, col=col, rows=len(vals), moved=n_moved, frac=frac)

    def subsample(self, table, keep_frac, seed=42):
        """按行抽样（尺度漂移）：行数变化、边际形状大致保持。"""
        rng = np.random.RandomState(seed)
        cur = self.con.cursor()
        cur.execute(f"SELECT rowid FROM {table}")
        rids = np.array([r[0] for r in cur.fetchall()], dtype=np.int64)
        keep = rids[rng.rand(len(rids)) < keep_frac]
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS _keep(rowid_ INTEGER PRIMARY KEY)")
        cur.execute("DELETE FROM _keep")
        cur.executemany("INSERT INTO _keep(rowid_) VALUES (?)", [(int(x),) for x in keep])
        cur.execute(f"DELETE FROM {table} WHERE rowid NOT IN (SELECT rowid_ FROM _keep)")
        self.con.commit()
        return dict(table=table, kept=int(len(keep)), of=int(len(rids)))

    def restore_from_disk(self):
        """丢弃内存库，重新从 CSV 载入（回到原始状态）。"""
        self.con.close()
        self.con = sqlite3.connect(":memory:")
        self.tables = {}
        self._load(verbose=False)


# ---------------------------------------------------------------- join 索引
_ALIAS_DEF = re.compile(r"(?:from|join)\s+([a-z_0-9]+)\s+(?:as\s+)?(st_[a-z_0-9]+)")
_JOIN_EQ = re.compile(r"(st_[a-z_0-9]+)\.([a-z_0-9]+)\s*=\s*(st_[a-z_0-9]+)\.([a-z_0-9]+)")


def join_columns(sqls):
    """从 workload 的 SQL 里抽出所有作为连接键的 (表, 列)，用于建索引。"""
    need = set()
    for s in sqls:
        low = s.lower()
        a2t = {al: tb for tb, al in _ALIAS_DEF.findall(low)}
        for al1, c1, al2, c2 in _JOIN_EQ.findall(low):
            for al, c in ((al1, c1), (al2, c2)):
                tb = a2t.get(al)
                if tb:
                    need.add((tb, c))
    return need


def create_join_indices(con, sqls, verbose=False):
    """在连接键上建索引——SQLite 无索引时会对每张表做全表扫描，这是主要瓶颈之一。"""
    cur = con.cursor()
    made = []
    for tb, c in sorted(join_columns(sqls)):
        try:
            cur.execute(f'CREATE INDEX IF NOT EXISTS "ix_{tb}_{c}" ON "{tb}"("{c}")')
            made.append(f"{tb}.{c}")
        except Exception:
            pass
    con.commit()
    if verbose:
        print(f"[datasim] 建索引 {len(made)} 个: {', '.join(made[:12])}{' ...' if len(made) > 12 else ''}")
    return made


# ---------------------------------------------------------------- 多进程并行
def _worker(args):
    db_name, data_dir, drift_spec, sqls, timeout = args
    d = StatsDB(db_name, data_dir=data_dir, verbose=False, query_timeout_s=timeout)
    d.ensure_indices(sqls)
    if drift_spec:
        t = drift_spec.get("type")
        if t == "permute":
            d.permute_column(drift_spec["table"], drift_spec["col"], frac=drift_spec.get("frac", 1.0),
                             where=drift_spec.get("where"), seed=drift_spec.get("seed", 42))
        elif t == "subsample":
            d.subsample(drift_spec["table"], drift_spec.get("keep_frac", 0.5),
                        seed=drift_spec.get("seed", 42))
    vals = d.exec_many(sqls)
    d.con.close()
    return vals


def exec_pool(db_name, sqls, drift_spec=None, n_workers=8, data_dir=None, timeout=1.5):
    """并行执行同一批查询。

    用【线程池】而非进程池：sqlite3 在 C 层执行查询时会释放 GIL，因此线程能真正并行；
    同时避免多进程所需的名管道 IPC（在受限沙箱下会被拒绝）。
    每个线程持有【独立的 sqlite 连接】，并各自施加同一漂移，保证与串行等价。
    """
    from concurrent.futures import ThreadPoolExecutor
    n = len(sqls)
    parts = [p for p in np.array_split(np.arange(n), max(1, min(n_workers, n))) if len(p)]
    tasks = [(db_name, data_dir, drift_spec, [sqls[i] for i in p], timeout) for p in parts]
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        res = list(ex.map(_worker, tasks))
    out = [None] * n
    for p, vals in zip(parts, res):
        for j, i in enumerate(p):
            out[i] = vals[j]
    return out


# ---------------------------------------------------------------- workload 解析
def load_workload(db="stats", usage="test", data_dir=None, with_true=True):
    """读 workloads.sql / workloads_all.sql，返回 list[dict(sql, true, pg_est)]。

    格式：sql || true [|| pg_est]（与 unpublished internal prototype 的特征提取一致）
    """
    wd = os.path.join(data_dir or DATA_DIR, "workloads", usage, db)
    for fn in ("workloads.sql", "workloads_all.sql"):
        p = os.path.join(wd, fn)
        if os.path.exists(p):
            break
    else:
        raise FileNotFoundError(wd)

    def num(s):
        s = s.strip().strip("'\"")
        try:
            return float(s)
        except ValueError:
            return None

    out = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("--"):
                continue
            parts = line.split("||")
            sql = re.sub(r"^/\*[^*]*\*/\s*", "", parts[0]).strip().rstrip(";").strip()
            true = num(parts[1]) if with_true and len(parts) > 1 else None
            pg = num(parts[2]) if len(parts) > 2 else None
            if true is None or true <= 0:
                continue
            out.append(dict(sql=sql, true=true, pg_est=pg))
    return out
