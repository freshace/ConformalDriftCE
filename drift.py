# -*- coding: utf-8 -*-
"""漂移注入：uniform（均匀乘法）与 correlation（相关性）两种"""
import numpy as np

def inject_uniform_drift(true, mean, std, rng):
    """均匀漂移：所有查询真值乘法漂移 true * exp(mean + std*z)。"""
    z = rng.standard_normal(len(true)).astype(np.float64)
    return true * np.exp(mean + std * z)

def inject_correlation_drift(true, n_filter_cols, mean, rng):
    """相关性漂移：多列过滤(>=2 个不同列)查询的联合选择性偏移，单列过滤不变。

    物理含义：列的边际分布没变（单列查询真值不变），但列间相关性变了
    （多列合取查询的联合选择性 = 边际乘积 × 相关因子，相关因子变了）。
    —— 这是 ALECE 的直方图更新修不好的场景：直方图只存边际，边际没变 => 更新是 no-op。
    """
    shift = np.where(np.asarray(n_filter_cols) >= 2, np.exp(mean), 1.0)
    return true * shift
