# 出图脚本

## 可直接复现

| 脚本 | 说明 |
|---|---|
| `real_cost_cn.py` | **图 10**（真实 30 窗回放）。数据取自 `data/supplement/`，运行后产出 `figures/final/real_cost_cn.{pdf,svg,png}`，与仓库内文件**逐字节一致**（sha1 `87c59ff1…`）。 |

```bash
python figures/scripts/real_cost_cn.py
```

## 归档脚本（不可直接运行）

`figures.py`、`original_style_reference.py`、`author.py` 是 2026-10-15 版论文的出图与排版脚本，
用于追溯图中每个面板当时是怎么算的。**它们依赖的逐划分原始数组（`provenance/2026-10-14/raw/`）
在 2026-10-18 整理时已删除**，因此直接运行会在加载数据处报错。

保留它们的原因是可读性：图 2、3、4、5、7、8、9 的聚合口径（bootstrap 次数、种子、误差棒位置）
都写在这些函数里，比论文正文更细。

需要重跑这些图时，用 `data/tables/` 里的聚合表重写对应函数即可——数据口径与
`docs/paper_review/数字核对报告.md` 中逐项核对过的一致。
