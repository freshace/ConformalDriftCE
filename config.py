# -*- coding: utf-8 -*-
"""DriftCE 配置"""
import os

DATA_PATH = os.environ.get("DRIFTCE_DATA_PATH", "data/features40_s3a.pkl")

N_TRAIN = 20000   # 旧数据：训练 LGBM 修正器
N_CAL   = 3000    # conformal 校准集（旧数据）
N_TEST  = 4000    # 测试集（注入漂移 + 评估）
N_ADAPT_TRAIN = 500   # 漂移后：重训 LGBM 的新标签数
N_ADAPT_CAL   = 500   # 漂移后：重校准 conformal 的标签数（与训练集分离）

ALPHA = 0.1       # 目标覆盖率 90%

DRIFT_MEANS = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]
DRIFT_STD   = 0.2

LGBM_PARAMS = dict(n_estimators=300, num_leaves=63, learning_rate=0.06,
                   random_state=42, verbose=-1, n_jobs=1)

SEED = 42
