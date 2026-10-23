# -*- coding: utf-8 -*-
"""LGBM 残差修正器：学 log(true/base) 的残差"""
import numpy as np
import lightgbm as lgb

def fit_corrector(feat, base, true, params):
    resid = np.log(true) - np.log(np.clip(base, 1.0, None))
    m = lgb.LGBMRegressor(**params)
    m.fit(feat, resid)
    return m

def predict_corrector(model, feat):
    return np.clip(model.predict(feat), -30.0, 30.0)
