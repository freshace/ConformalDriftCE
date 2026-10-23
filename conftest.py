# -*- coding: utf-8 -*-
"""让仓库根目录的松散模块既能 `import config`，也能 `import drift_ce.config`。

背景：`experiments/*.py` 用 `import config, data, model, conformal`（顶层名），
而 `tests/test_*.py` 里混用 `from drift_ce.conformal import ...`（包名）。
把根目录和它的父目录都放进 sys.path，两种写法都能跑，无需改动任何实验脚本。
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_ROOT)

for _p in (_ROOT, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
