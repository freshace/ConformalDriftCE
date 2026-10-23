"""Auditable labeled monitoring and next-window repair on real-query synthetic replay.

No database calls. Baseline query cardinalities/features are cached real data; the
time trajectory and changed cardinalities are synthetic. Evaluation labels never
enter threshold fitting, the sentinel monitor, training, or calibration.
"""
import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import config
import conformal
import features
import model

DETECTORS = ("coverage_exceedance", "mean_absolute_log_residual", "qerror_gt10_fraction")
POLICIES = ("frozen", "trigger_recalibrate", "trigger_retrain", "every_window_recalibrate")
PATTERNS = ("none", "abrupt", "gradual", "periodic")


def partition_indices(query_ids, sizes, seed):
    query_ids = np.asarray(query_ids)
    if len(np.unique(query_ids)) != len(query_ids):
        raise ValueError("duplicate query identifiers must be deduplicated before splitting")
    if sum(sizes.values()) > len(query_ids):
        raise ValueError("insufficient unique queries for all disjoint roles")
    order = np.random.default_rng(seed).permutation(query_ids)
    result, cursor = {}, 0
    for role, size in sizes.items():
        result[role] = order[cursor:cursor + size]
        cursor += size
    return result


def fit_thresholds(validation_signals, target_fpr):
    """Fit solely on independent baseline validation windows; strict > at test."""
    values = np.asarray(validation_signals, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 10:
        raise ValueError("at least ten independent validation windows are required")
    return np.quantile(values, 1 - target_fpr, axis=0, method="higher")


def alarm_matrix(signals, thresholds):
    return np.asarray(signals) > np.asarray(thresholds)


def label_ledger(n_train, n_cal, n_threshold, n_monitor, n_update_train,
                 n_update_cal, n_eval, n_negative, update_cap):
    update_labels = n_update_train + n_update_cal
    if update_labels > update_cap:
        raise ValueError("update label budget exceeded")
    values = dict(initial_training=n_train, initial_calibration=n_cal,
                  threshold_validation=n_threshold, monitoring=n_monitor,
                  update_training=n_update_train, update_calibration=n_update_cal,
                  stream_evaluation=n_eval, negative_evaluation=n_negative)
    values.update(update_labels=update_labels,
                  deployment_feedback_labels=n_monitor + update_labels,
                  evaluation_labels=n_eval + n_negative,
                  total_label_accesses=sum(values.values()))
    return values


def residuals(truth, estimate):
    estimate = np.maximum(np.asarray(estimate), 1.0)
    scores = np.abs(np.log(truth) - np.log(estimate))
    return scores, np.maximum(truth / estimate, estimate / truth)


def monitor_signals(scores, radius):
    scores = np.asarray(scores)
    return np.array([np.mean(scores > radius), np.mean(scores), np.mean(scores > np.log(10))])


def drift_trajectory(pattern, windows, start, maximum):
    rho = np.zeros(windows)
    if pattern == "none":
        return rho
    if pattern == "abrupt":
        rho[start:] = maximum
    elif pattern == "gradual":
        rho[start:] = np.minimum(maximum, maximum * np.arange(1, windows - start + 1) / 5)
    elif pattern == "periodic":
        rho[start:] = [maximum if ((t - start) % 4) < 2 else 0 for t in range(start, windows)]
    else:
        raise ValueError(pattern)
    return rho


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def load_real_queries(pkl_path, workload_path):
    """Deduplicate exact whitespace-normalized SQL identities, preserving raw row ids."""
    with open(pkl_path, "rb") as stream:
        raw = pickle.load(stream)
    sql_rows = []
    with open(workload_path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("--"):
                continue
            parts = line.strip().split("||")
            if len(parts) < 2:
                raise ValueError("workload requires SQL and baseline cardinality")
            sql_rows.append((" ".join(parts[0].strip().rstrip(";").split()), float(parts[1])))
    if len(sql_rows) != len(raw["features"]):
        raise ValueError("raw feature rows and SQL rows must be aligned")
    sg = raw["schema_graph"]
    n_tables = max(sg["table_index"].values()) + 1
    n_cols = max(sg["col_index"].values()) + 1
    bases, truths, feats, nfcs, raw_ids, keys = [], [], [], [], [], []
    seen, excluded, duplicates = set(), 0, 0
    for raw_id, record in enumerate(raw["features"]):
        ft, label, pgv, *_, extra = record
        truth, base = float(np.expm1(label - 1)), float(np.expm1(pgv - 1))
        if truth <= 0 or base <= 0 or not np.isfinite(truth + base):
            excluded += 1
            continue
        sql, sql_truth = sql_rows[raw_id]
        if not np.isclose(truth, sql_truth, rtol=2e-5, atol=2e-4):
            raise ValueError(f"feature/workload cardinality mismatch at raw row {raw_id}")
        if sql in seen:
            duplicates += 1
            continue
        seen.add(sql)
        f = np.nan_to_num(np.concatenate([
            features.build_schema_feature(ft, extra, n_tables, n_cols), features.stats9(ft),
            [np.log(max(base, 1.0))]]), nan=0., posinf=1e6, neginf=-1e6).astype(np.float32)
        bases.append(base); truths.append(truth); feats.append(f)
        nfcs.append(len(set(extra.get("filter_col_identifiers", []))))
        raw_ids.append(raw_id); keys.append(hashlib.sha256(sql.encode()).hexdigest())
    return dict(base=np.asarray(bases), truth=np.asarray(truths), feat=np.asarray(feats),
                nfc=np.asarray(nfcs), raw_ids=np.asarray(raw_ids), query_hash=np.asarray(keys),
                meta=dict(raw_rows=len(sql_rows), retained_unique_positive=len(bases),
                          excluded_nonpositive_or_nonfinite=excluded, duplicate_sql=duplicates,
                          feature_dimension=len(feats[0])))


def estimate(corrector, ids, dataset):
    # Existing model API returns a clipped residual; use the same estimate floor everywhere.
    return np.maximum(dataset["base"][ids] * np.exp(model.predict_corrector(corrector, dataset["feat"][ids])), 1.)


def shifted_truth(dataset, ids, rho):
    return dataset["truth"][ids] * np.exp(rho * (dataset["nfc"][ids] >= 2))


def metrics(truth, estimates, radius):
    scores, qerr = residuals(truth, estimates)
    width = estimates * (np.exp(radius) - np.exp(-radius))
    return dict(coverage=float(np.mean(scores <= radius)),
                multiplicative_width=float(np.exp(2 * radius)),
                median_absolute_width=float(np.median(width)),
                q50=float(np.quantile(qerr, .5)), q90=float(np.quantile(qerr, .9)),
                q99=float(np.quantile(qerr, .99)))


def bootstrap_mean(values):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(9302026)
    means = a[rng.integers(0, len(a), size=(10000, len(a)))].mean(axis=1)
    return dict(mean=float(a.mean()), std=float(a.std(ddof=1)) if len(a) > 1 else 0.,
                ci95=[float(x) for x in np.quantile(means, [.025, .975])], n=len(a))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run_seed(seed, dataset, settings, output):
    p = settings
    sizes = dict(train=p["train"], cal=p["cal"],
                 threshold=p["threshold_windows"] * p["monitor"],
                 negative=p["negative_windows"] * p["monitor"],
                 stream=p["windows"] * (p["monitor"] + p["eval"]),
                 pool=p["windows"] * p["repair"])
    ids = partition_indices(np.arange(len(dataset["base"])), sizes, seed)
    # Source query ids/hashes are saved for independent leakage checks.
    raw = {f"split_{role}": dataset["raw_ids"][index] for role, index in ids.items()}
    raw.update({f"query_hash_{role}": dataset["query_hash"][index] for role, index in ids.items()})
    fit_start = time.perf_counter()
    params = dict(p["lgbm_params"], random_state=seed)
    m0 = model.fit_corrector(dataset["feat"][ids["train"]], dataset["base"][ids["train"]],
                             dataset["truth"][ids["train"]], params)
    fit_seconds = time.perf_counter() - fit_start
    cal_start = time.perf_counter()
    q0 = conformal.calibrate_q(residuals(dataset["truth"][ids["cal"]], estimate(m0, ids["cal"], dataset))[0], p["alpha"])
    cal_seconds = time.perf_counter() - cal_start
    threshold_start = time.perf_counter()
    val_scores = residuals(dataset["truth"][ids["threshold"]], estimate(m0, ids["threshold"], dataset))[0]
    val_signals = np.asarray([monitor_signals(window, q0) for window in val_scores.reshape(-1, p["monitor"])])
    thresholds = fit_thresholds(val_signals, p["target_fpr"])
    threshold_seconds = time.perf_counter() - threshold_start
    negative_scores = residuals(dataset["truth"][ids["negative"]], estimate(m0, ids["negative"], dataset))[0]
    negative_signals = np.asarray([monitor_signals(window, q0) for window in negative_scores.reshape(-1, p["monitor"])])
    negative_alarms = alarm_matrix(negative_signals, thresholds)
    raw.update(q0=np.array(q0), thresholds=thresholds, validation_signals=val_signals,
               negative_signals=negative_signals, negative_alarms=negative_alarms)
    stream_windows = ids["stream"].reshape(p["windows"], p["monitor"] + p["eval"])
    monitor_start = time.perf_counter()
    sentinel_monitor_est = estimate(m0, stream_windows[:, :p["monitor"]].flatten(), dataset).reshape(p["windows"], p["monitor"])
    monitoring_inference_seconds = time.perf_counter() - monitor_start
    sentinel_eval_est = estimate(m0, stream_windows[:, p["monitor"]:].flatten(), dataset).reshape(p["windows"], p["eval"])
    sentinel_est = np.concatenate([sentinel_monitor_est, sentinel_eval_est], axis=1)
    detector_rows, policy_rows, window_rows = [], [], []
    for pattern in PATTERNS:
        rho = drift_trajectory(pattern, p["windows"], p["drift_start"], p["max_rho"])
        truths = np.asarray([shifted_truth(dataset, w, rho[t]) for t, w in enumerate(stream_windows)])
        monitor_start = time.perf_counter()
        mon_scores = np.abs(np.log(truths[:, :p["monitor"]]) - np.log(sentinel_est[:, :p["monitor"]]))
        signals = np.asarray([monitor_signals(window, q0) for window in mon_scores])
        alarms = alarm_matrix(signals, thresholds)
        monitoring_statistic_seconds = time.perf_counter() - monitor_start
        frozen_cov = np.mean(np.abs(np.log(truths[:, p["monitor"]:]) -
                                    np.log(sentinel_est[:, p["monitor"]:])) <= q0, axis=1)
        failures = np.flatnonzero(frozen_cov < p["failure_coverage"])
        onset = int(failures[0]) if len(failures) else None
        raw.update({f"{pattern}_rho": rho, f"{pattern}_monitor_scores": mon_scores,
                    f"{pattern}_monitor_estimate": sentinel_est[:, :p["monitor"]],
                    f"{pattern}_monitor_truth": truths[:, :p["monitor"]],
                    f"{pattern}_signals": signals, f"{pattern}_alarms": alarms,
                    f"{pattern}_evaluation_truth": truths[:, p["monitor"]:]})
        for d, name in enumerate(DETECTORS):
            after = np.flatnonzero(alarms[:, d] & (np.arange(p["windows"]) >= (onset if onset is not None else p["windows"])))
            positive = rho > 0
            detector_rows.append(dict(seed=seed, pattern=pattern, detector=name, q0=q0,
                qerror_equivalent_threshold=float(np.exp(q0)), threshold=float(thresholds[d]),
                validation_alarm_rate=float(np.mean(val_signals[:, d] > thresholds[d])),
                independent_negative_fpr=float(negative_alarms[:, d].mean()),
                positive_window_tpr=float(alarms[positive, d].mean()) if positive.any() else None,
                prechange_alarms=int(alarms[:p["drift_start"], d].sum()),
                early_before_failure=int(alarms[:onset, d].sum()) if onset is not None else int(alarms[:, d].sum()),
                frozen_failure_onset=onset, failure_delay=int(after[0] - onset) if len(after) else None,
                failure_detected=bool(len(after)), alarms=int(alarms[:, d].sum()),
                monitor_labels=p["windows"] * p["monitor"],
                independent_negative_labels=sizes["negative"]))
        for policy in POLICIES:
            current_model, current_q = m0, q0
            last_update, train_labels, cal_labels, updates = -100, 0, 0, 0
            inference_seconds, update_seconds = 0., 0.
            all_estimates, radii = [], []
            for t, w in enumerate(stream_windows):
                start = time.perf_counter()
                eval_estimate = estimate(current_model, w[p["monitor"]:], dataset)
                inference_seconds += time.perf_counter() - start
                y_eval = truths[t, p["monitor"]:]
                all_estimates.append(eval_estimate); radii.append(current_q)
                row = dict(seed=seed, pattern=pattern, policy=policy, window=t, rho=float(rho[t]),
                           radius=current_q, **metrics(y_eval, eval_estimate, current_q))
                # A fixed reference sentinel has the same labels, radius, threshold and decisions
                # for every repair policy; it is not recalibrated with the repaired predictor.
                fired = policy == "every_window_recalibrate" or (
                    policy in ("trigger_recalibrate", "trigger_retrain") and
                    alarms[t, 0] and t - last_update >= p["cooldown"])
                # No update after the final evaluated window; it cannot affect any result.
                fired = bool(fired and t < p["windows"] - 1)
                if fired:
                    update_start = time.perf_counter()
                    repair_ids = ids["pool"][t * p["repair"]:(t + 1) * p["repair"]]
                    repair_truth = shifted_truth(dataset, repair_ids, rho[t])
                    if policy == "trigger_retrain":
                        middle = p["repair"] // 2
                        train_ids, cal_ids = repair_ids[:middle], repair_ids[middle:]
                        current_model = model.fit_corrector(dataset["feat"][train_ids], dataset["base"][train_ids],
                                                            repair_truth[:middle], params)
                        train_labels += len(train_ids)
                        cal_truth = repair_truth[middle:]
                    else:
                        cal_ids, cal_truth = repair_ids, repair_truth
                    current_q = conformal.calibrate_q(residuals(cal_truth, estimate(current_model, cal_ids, dataset))[0], p["alpha"])
                    cal_labels += len(cal_ids); updates += 1; last_update = t
                    update_seconds += time.perf_counter() - update_start
                row.update(update_after_evaluation=fired, updates_to_date=updates,
                           update_training_labels_to_date=train_labels, update_calibration_labels_to_date=cal_labels)
                window_rows.append(row)
            all_estimates = np.asarray(all_estimates)
            radii = np.asarray(radii)
            y_eval = truths[:, p["monitor"]:]
            scores, qerr = residuals(y_eval, all_estimates)
            pooled = dict(coverage=float(np.mean(scores <= radii[:, None])),
                multiplicative_width=float(np.mean(np.exp(2 * radii))),
                median_absolute_width=float(np.median(all_estimates * (np.exp(radii[:, None]) - np.exp(-radii[:, None])))),
                q50=float(np.quantile(qerr, .5)), q90=float(np.quantile(qerr, .9)), q99=float(np.quantile(qerr, .99)))
            post = np.arange(p["windows"]) >= p["drift_start"]
            ledger = label_ledger(sizes["train"], sizes["cal"], sizes["threshold"], p["windows"] * p["monitor"],
                                  train_labels, cal_labels, p["windows"] * p["eval"], sizes["negative"], sizes["pool"])
            policy_rows.append(dict(seed=seed, pattern=pattern, policy=policy, updates=updates, **pooled,
                postchange_coverage=float(np.mean(scores[post] <= radii[post, None])),
                initial_fit_seconds=fit_seconds, initial_calibration_seconds=cal_seconds,
                threshold_fit_seconds=threshold_seconds, monitoring_inference_seconds=monitoring_inference_seconds,
                monitoring_statistic_seconds=monitoring_statistic_seconds,
                evaluation_inference_seconds=inference_seconds, update_compute_seconds=update_seconds,
                local_model_compute_seconds=fit_seconds + cal_seconds + threshold_seconds + inference_seconds + update_seconds + monitoring_inference_seconds + monitoring_statistic_seconds,
                **ledger))
            raw[f"{pattern}_{policy}_evaluation_estimate"] = all_estimates
            raw[f"{pattern}_{policy}_radii"] = radii
    np.savez_compressed(output / f"monitoring_seed_{seed}.npz", **raw)
    return detector_rows, policy_rows, window_rows


def summarize(detector_rows, policy_rows):
    detector, policies = [], []
    for pattern in PATTERNS:
        for name in DETECTORS:
            rows = [r for r in detector_rows if r["pattern"] == pattern and r["detector"] == name]
            fields = ("independent_negative_fpr", "positive_window_tpr", "prechange_alarms", "early_before_failure", "failure_delay", "alarms")
            record = dict(pattern=pattern, detector=name)
            for field in fields:
                values = [r[field] for r in rows if r[field] is not None]
                record[field] = bootstrap_mean(values) if values else None
            record.update(failure_onset_runs=sum(r["frozen_failure_onset"] is not None for r in rows),
                          detected_failure_runs=sum(r["failure_detected"] for r in rows),
                          missed_failure_runs=sum(r["frozen_failure_onset"] is not None and not r["failure_detected"] for r in rows))
            detector.append(record)
        for name in POLICIES:
            rows = [r for r in policy_rows if r["pattern"] == pattern and r["policy"] == name]
            fields = ("coverage", "postchange_coverage", "multiplicative_width", "median_absolute_width", "q50", "q90", "q99", "updates",
                      "update_labels", "deployment_feedback_labels", "total_label_accesses", "local_model_compute_seconds", "update_compute_seconds")
            policies.append(dict(pattern=pattern, policy=name, **{field: bootstrap_mean([r[field] for r in rows]) for field in fields}))
    return dict(detectors=detector, policies=policies)


def report(summary, settings, metadata):
    lines = ["# Labeled monitoring and repair: cached real stats queries, synthetic time replay", "",
        "Baseline data are aligned real stats query features/cardinalities. Changed cardinalities and time schedules are synthetic; no database was executed.",
        "", f"Protocol: {len(settings['seeds'])} seeds; {settings['windows']} windows; {settings['monitor']} monitoring labels and {settings['eval']} separate evaluation labels/window.",
        f"Disjoint roles: {settings['train']} training, {settings['cal']} calibration, {settings['threshold_windows'] * settings['monitor']} threshold validation, {settings['negative_windows'] * settings['monitor']} independent negative evaluation, {settings['windows'] * (settings['monitor'] + settings['eval'])} stream, {settings['windows'] * settings['repair']} repair pool.",
        f"SQL deduplication: {metadata['raw_rows']} raw rows, {metadata['retained_unique_positive']} unique positive queries; {metadata['duplicate_sql']} duplicates removed and {metadata['excluded_nonpositive_or_nonfinite']} nonpositive/nonfinite rows excluded.",
        "All three labeled detectors see exactly the same 100 labels/window. Each independently uses the higher 90th validation quantile and strict > decisions (nominal per-window FPR 10%); measured FPR uses only the separate negative windows.",
        "Coverage exceedance at fixed q is exactly the historical q-error tail fraction above exp(q), so this experiment does not claim a distinct source of detection information.",
        "Updates use a frozen reference predictor/radius as a shadow sentinel. All repair policies share its alarm decisions; this isolates the repair choice, not an adaptive-detector comparison.",
        "Every policy has the same 4800-label update cap, but actual adaptive spending differs. Each update spends 400 fresh disjoint labels: recalibration uses all 400; retraining uses 200 training + 200 calibration. Cooldown is two windows. Evaluation precedes repair, which first affects the following window; no final-window update is charged.",
        "Offline evaluation and negative-flow labels are counted separately and never become free deployment feedback. Local compute time excludes label acquisition, DB execution, and one-time file/feature loading. Synchronous label availability and instantaneous synthetic state access are assumptions.",
        "Coverage uses all evaluation queries, with max(estimate,1) everywhere. Width is H/L=exp(2q), averaged over windows; absolute width is the pooled median H-L. q50/q90/q99 are pooled within each seed across all windows, then averaged across seeds.",
        "Uncertainty is the 2.5/97.5 percentiles of 10,000 bootstrap resamples of seed-level metrics. It captures partition/model variability conditional on this cached dataset, not independent database populations.",
        "Failure onset is the first frozen evaluation window with coverage <85%. Delay is the first subsequent alarm minus this onset; missed failures are reported separately. Early alarms before onset are not conflated with independently measured baseline FPR.", "",
        "| Pattern | Detector | independent FPR | positive-window TPR | delay / detected failure runs |", "|---|---|---:|---:|---:|"]
    for r in summary["detectors"]:
        tpr = "--" if r["positive_window_tpr"] is None else f"{r['positive_window_tpr']['mean']:.3f}"
        delay = "--" if r["failure_delay"] is None else f"{r['failure_delay']['mean']:.2f}"
        lines.append(f"| {r['pattern']} | {r['detector']} | {r['independent_negative_fpr']['mean']:.3f} | {tpr} | {delay} / {r['detected_failure_runs']} |")
    lines += ["", "| Pattern | Policy | coverage | width H/L | q50 | q90 | q99 | update labels | compute s |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary["policies"]:
        lines.append(f"| {r['pattern']} | {r['policy']} | {r['coverage']['mean']:.3f} | {r['multiplicative_width']['mean']:.1f} | {r['q50']['mean']:.2f} | {r['q90']['mean']:.2f} | {r['q99']['mean']:.2f} | {r['update_labels']['mean']:.0f} | {r['local_model_compute_seconds']['mean']:.3f} |")
    lines += ["", "Interpretation: compare detectors at their measured FPR, not solely their nominal target. Recalibration cannot alter point q-error. Coverage gains can require wider intervals; retraining can change point tails and may worsen them with only 200 fresh training labels. This is a small-budget protocol demonstration, not evidence that one policy dominates or a production latency result."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=Path, default=HERE / "results_revision_20260930")
    parser.add_argument("--train", type=int, default=6000)
    args = parser.parse_args()
    if args.seeds < 10:
        raise ValueError("at least ten seeds are required")
    args.output.mkdir(exist_ok=True, parents=True)
    settings = dict(seeds=list(range(42, 42 + args.seeds)), train=args.train, cal=1000,
        threshold_windows=40, negative_windows=20, windows=12, monitor=100, eval=150,
        repair=400, alpha=.1, target_fpr=.1, drift_start=4, max_rho=2.,
        failure_coverage=.85, cooldown=2,
        lgbm_params=dict(n_estimators=80, num_leaves=31, learning_rate=.06, verbose=-1, n_jobs=1))
    workload = ROOT / "datas" / "datas" / "workloads" / "finetune" / "stats" / "workloads.sql"
    started = time.perf_counter()
    print("Loading aligned cached real queries (no database execution)...", flush=True)
    dataset = load_real_queries(Path(config.DATA_PATH), workload)
    load_seconds = time.perf_counter() - started
    print(json.dumps(dataset["meta"]), flush=True)
    detector_rows, policy_rows, window_rows = [], [], []
    for seed in settings["seeds"]:
        tick = time.perf_counter()
        d, p, w = run_seed(seed, dataset, settings, args.output)
        detector_rows += d; policy_rows += p; window_rows += w
        print(f"seed {seed} complete in {time.perf_counter() - tick:.2f}s", flush=True)
    summary = summarize(detector_rows, policy_rows)
    source_paths = [Path(__file__), HERE / "features.py", HERE / "model.py", HERE / "conformal.py", HERE / "config.py",
                    HERE / "tests" / "test_revision_monitoring.py", Path(config.DATA_PATH), workload]
    provenance = dict(settings=settings, data=dataset["meta"],
        source_sha256={str(path): sha256(path) for path in source_paths},
        loading_feature_seconds=load_seconds, total_wall_seconds=time.perf_counter() - started,
        command=".venv/Scripts/python.exe drift_ce/experiments/experiment_revision_monitoring.py --seeds 10",
        replay="synthetic correlation cardinality multipliers applied only to queries with >=2 distinct filter columns",
        label_feedback="only monitor and disjoint repair-pool labels; evaluation labels never used for adaptation",
        confidence_interval="seed-level percentile bootstrap, 10000 resamples, RNG seed 9302026",
        packages={"numpy": np.__version__, "lightgbm": model.lgb.__version__})
    write_csv(args.output / "monitoring_detectors.csv", detector_rows)
    write_csv(args.output / "monitoring_policies.csv", policy_rows)
    write_csv(args.output / "monitoring_windows.csv", window_rows)
    for name, obj in (("monitoring_summary.json", summary), ("monitoring_protocol.json", provenance),
                      ("monitoring_raw.json", dict(detectors=detector_rows, policies=policy_rows))):
        (args.output / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (args.output / "monitoring_report.md").write_text(report(summary, settings, dataset["meta"]), encoding="utf-8")
    print(f"Saved audit artifacts to {args.output}; total wall {provenance['total_wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    # sklearn's feature-name warning is emitted by LGBM even when both sides are arrays.
    warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
    main()
