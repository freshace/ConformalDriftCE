# -*- coding: utf-8 -*-
"""旧版经验基线：覆盖率/残差/特征信号的标签辅助可靠性失效监控对比。
任务：给定一段查询（漂移 or 无漂移），比较经验监控信号。
对比指标：TPR（真漂移检出）、FPR（无漂移误报）、在多个漂移强度下。

三种检测器（都在观察一段查询后判断）：
    1. conformal 覆盖率：覆盖率 < 1-alpha -> 报警（仅作经验可靠性失效信号）。
  2. 残差均值检测：平均 |log error| 超阈值 -> 报警（无保证）。
  3. 特征分布距离：训练 vs 窗口的特征均值 L2 距离超阈值 -> 报警（无真值、无保证）。

公平性：三种都用同样的窗口切分；阈值各自按"无漂移窗口"的 90 分位标定（经验 FPR 目标=10%），
再比较漂移窗口的检出率（TPR）。
该旧脚本不满足新版独立检测阈值评估协议，论文不得引用其结果；新版结果来自独立的 CI 脚本。
输出到 results_repro/experiment_detection.txt
"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config, data, model, conformal

N_WINDOW = 200   # 每窗口查询数
DRIFT_FACTORS = [1.0, 1.35, 2.0, 2.72, 4.48, 7.39]

def main():
    rng = np.random.RandomState(config.SEED)
    base, true, feat, nfc = data.load_stats(config.DATA_PATH)
    (f_tr, b_tr, y_tr, _), (f_cal, b_cal, y_cal, _), (f_te, b_te, y_te, nfc_te) = \
        data.split(feat, base, true, nfc, config.N_TRAIN, config.N_CAL, config.N_TEST, rng)

    m = model.fit_corrector(f_tr, b_tr, y_tr, config.LGBM_PARAMS)
    q90 = conformal.calibrate_q(conformal.score(y_cal, b_cal, model.predict_corrector(m, f_cal)), config.ALPHA)
    corr_te = model.predict_corrector(m, f_te)

    # 构造窗口：把测试集切成多个窗口，随机分配漂移/无漂移
    n_te = len(f_te)
    n_win = n_te // N_WINDOW
    wins = [np.arange(i*N_WINDOW, (i+1)*N_WINDOW) for i in range(n_win)]

    # 对每个漂移因子，生成漂移窗口的标签和检测分数
    out = []
    def log(s):
        out.append(s); print(s, flush=True)

    log("drift_factor | conf_cov_TPR | conf_cov_FPR | resid_TPR | resid_FPR | feat_TPR | feat_FPR")
    log("-" * 90)

    # 无漂移基准：用漂移因子=1.0 时所有窗口都是"无漂移"
    # 标定阈值：在无漂移窗口上取 90 分位
    # 先算无漂移下三个检测器的分数
    cov_scores_nodrift = []
    resid_scores_nodrift = []
    feat_scores_nodrift = []
    feat_train_mean = f_tr.mean(axis=0)

    for w in wins:
        s_cov = conformal.coverage(conformal.score(y_te[w], b_te[w], corr_te[w]), q90)
        cov_scores_nodrift.append(1.0 - s_cov)   # 覆盖率缺口越大越可能漂移
        err = np.abs(np.log(y_te[w]) - np.log(np.clip(b_te[w]*np.exp(corr_te[w]), 1, None)))
        resid_scores_nodrift.append(float(err.mean()))
        fd = float(np.linalg.norm(f_te[w].mean(axis=0) - feat_train_mean))
        feat_scores_nodrift.append(fd)

    cov_thr = np.percentile(cov_scores_nodrift, 90)
    resid_thr = np.percentile(resid_scores_nodrift, 90)
    feat_thr = np.percentile(feat_scores_nodrift, 90)

    for df in DRIFT_FACTORS:
        if df == 1.0:
            # 无漂移，所有窗口都是负例
            y_drift = y_te
            labels = np.zeros(len(wins))
        else:
            y_drift = y_te * np.where(nfc_te >= 2, np.exp(np.log(df)), 1.0)
            labels = np.ones(len(wins))
        # 每个窗口算三个检测器的分数
        c_conf, c_res, c_feat = [], [], []
        for w in wins:
            s_cov = conformal.coverage(conformal.score(y_drift[w], b_te[w], corr_te[w]), q90)
            c_conf.append(1.0 - s_cov)
            err = np.abs(np.log(y_drift[w]) - np.log(np.clip(b_te[w]*np.exp(corr_te[w]), 1, None)))
            c_res.append(float(err.mean()))
            c_feat.append(float(np.linalg.norm(f_te[w].mean(axis=0) - feat_train_mean)))
        pred_cov = (np.array(c_conf) > cov_thr).astype(int)
        pred_res = (np.array(c_res) > resid_thr).astype(int)
        pred_feat = (np.array(c_feat) > feat_thr).astype(int)
        tpr = lambda p: (p[labels==1].mean()*100) if labels.sum()>0 else 0.0
        fpr = lambda p: (p[labels==0].mean()*100) if (len(labels)-labels.sum())>0 else 0.0
        log(f"{df:>12.2f} | {tpr(pred_cov):>11.1f}% | {fpr(pred_cov):>11.1f}% | {tpr(pred_res):>9.1f}% | {fpr(pred_res):>9.1f}% | {tpr(pred_feat):>9.1f}% | {fpr(pred_feat):>9.1f}%")

    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_repro")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "experiment_detection.txt"), "w", encoding="utf-8") as f:
        f.write("exit=0\n" + "\n".join(out) + "\n")
    print("\nsaved to results_repro/experiment_detection.txt", flush=True)

if __name__ == "__main__":
    main()
