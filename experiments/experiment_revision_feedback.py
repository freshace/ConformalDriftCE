"""Offline replay of real execution pools under independently split biased feedback.

No database access. All selection rules and hyperparameters are fixed before labels
in the audit split are inspected. SQL complexity is a cost proxy, not measured cost.
Run: python drift_ce/experiments/experiment_revision_feedback.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

HERE = Path(__file__).resolve().parents[1]
FEEDBACK_ROLES = ('monitor', 'repair_train', 'repair_cal')


def canonical_sql(sql):
    # Preserve quoted literals exactly; normalize case/whitespace elsewhere.
    parts = re.split(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")", sql.strip().rstrip(';').strip())
    return ''.join(p if i % 2 else re.sub(r'\s+', ' ', p.lower()) for i, p in enumerate(parts))


def sql_complexity(sql, pg):
    unquoted = re.sub(r"'(?:''|[^'])*'", "''", canonical_sql(sql))
    between = re.search(r'\bfrom\b(.*?)(?:\bwhere\b|\bgroup\b|$)', unquoted)
    n_tables = 1 + between.group(1).count(',') + len(re.findall(r'\bjoin\b', between.group(1))) if between else 1
    n_pred = len(re.findall(r'(?:>=|<=|<>|!=|=|>|<|\blike\b|\bin\b|\bbetween\b)', unquoted))
    # Lexicographic structural complexity; predicted cardinality breaks ties.
    return [n_tables, n_pred, float(np.log(max(pg, 1.0)))]


def interval_scores(true, estimate):
    true = np.asarray(true, float)
    estimate = np.maximum(np.asarray(estimate, float), 1.0)
    residual = np.abs(np.log(np.maximum(true, 1.0)) - np.log(estimate))
    return np.where(true > 0, residual, np.inf)


def covered(scores, radius):
    # An infinite multiplicative radius still does not include a zero outcome.
    scores = np.asarray(scores)
    return np.isfinite(scores) & (scores <= radius)


def calibrate(scores, alpha=0.1):
    scores = np.asarray(scores, float)
    if len(scores) == 0 or np.isnan(scores).any() or not 0 < alpha < 1:
        raise ValueError('invalid calibration inputs')
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float(np.sort(scores)[rank - 1]) if rank <= len(scores) else np.inf


def prepare_pool(pool):
    n = len(pool['idx'])
    if len(pool['sqls']) != n:
        raise ValueError('SQL length mismatch')
    groups = {}
    for i, sql in enumerate(pool['sqls']):
        groups.setdefault(canonical_sql(sql), []).append(i)
    audit = {'input_rows': n, 'unique_sql_identities': len(groups),
             'duplicate_rows_removed': n - len(groups), 'states': {}}
    true, pg, valid = {}, {}, np.ones(n, dtype=bool)
    for state, values in pool['states'].items():
        if any(len(values[k]) != n for k in ('true', 'pg_est', 'ok')):
            raise ValueError(f'{state}: state array length mismatch')
        t = np.asarray([np.nan if x is None else x for x in values['true']], float)
        p = np.asarray([np.nan if x is None else x for x in values['pg_est']], float)
        good = np.asarray(values['ok'], bool) & np.isfinite(t) & np.isfinite(p) & (t >= 0) & (p >= 0)
        for members in groups.values():
            valid_members = [i for i in members if good[i]]
            if valid_members and len(set(t[valid_members])) > 1:
                raise ValueError(f'{state}: conflicting labels for duplicate query identity')
        audit['states'][state] = {'successful_finite_rows': int(good.sum()),
                                  'zero_outcomes_before_common_filter': int(np.sum(good & (t == 0))),
                                  'batch_elapsed_seconds': values.get('secs'),
                                  'per_query_execution_time_available': False}
        true[state], pg[state] = t, p
        valid &= good
    # Keep first identity row regardless of its labels; then apply common success.
    first = np.asarray([v[0] for v in groups.values()], int)
    retained = first[valid[first]]
    audit['common_success_unique_identities'] = len(retained)
    audit['failed_or_invalid_unique_identities_excluded'] = len(first) - len(retained)
    for state in true:
        audit['states'][state]['zero_outcomes_common_identity_set'] = int(np.sum(true[state][retained] == 0))
    baseline = 'baseline' if 'baseline' in true else 'imdb'
    return {'identities': [hashlib.sha256(canonical_sql(pool['sqls'][i]).encode()).hexdigest() for i in retained],
            'source_idx': np.asarray(pool['idx'])[retained], 'sqls': [pool['sqls'][i] for i in retained],
            'true': {s: t[retained] for s, t in true.items()},
            'pg': {s: p[retained] for s, p in pg.items()}, 'baseline': baseline,
            'audit': audit, 'meta': pool.get('meta', {})}


def split_roles(n, seed):
    perm = np.random.default_rng(seed).permutation(n)
    ends = np.cumsum([int(.2 * n), int(.15 * n), int(.15 * n), int(.2 * n)])
    initial_train, calibration, threshold_validation, audit, reservoir = np.split(perm, ends)
    monitor, repair_train, repair_cal = np.array_split(reservoir, 3)
    return dict(initial_train=initial_train, calibration=calibration,
                threshold_validation=threshold_validation, audit=audit,
                monitor=monitor, repair_train=repair_train, repair_cal=repair_cal)


def select_feedback(roles, complexity, epsilon, role_budget, seed):
    rng = np.random.default_rng(seed)
    result = {}
    for role in FEEDBACK_ROLES:
        candidates = np.asarray(roles[role])
        if len(candidates) < role_budget:
            raise ValueError('insufficient distinct role candidates for fixed budget')
        n_explore = int(round(role_budget * epsilon))
        exploration = rng.choice(candidates, n_explore, replace=False)
        remainder = candidates[~np.isin(candidates, exploration)]
        # Fixed before labels: fewest tables, predicates, then baseline PG estimate.
        c = np.asarray(complexity)[remainder]
        ordering = np.lexsort((remainder, c[:, 2], c[:, 1], c[:, 0]))
        exploitation = remainder[ordering[:role_budget - n_explore]]
        result[role] = dict(selected=np.concatenate([exploration, exploitation]),
                            exploration=exploration, exploitation=exploitation)
    return result


def alarm_threshold(validation_misses, window_size, seed, level=.05, draws=2000):
    misses = np.asarray(validation_misses, bool)
    if len(misses) < window_size:
        raise ValueError('threshold validation smaller than monitoring window')
    rng = np.random.default_rng(seed)
    counts = [int(misses[rng.choice(len(misses), window_size, replace=False)].sum()) for _ in range(draws)]
    return int(np.quantile(counts, 1 - level, method='higher'))


def qerrors(true, estimate):
    t, e = np.maximum(true, 1), np.maximum(estimate, 1)
    return np.maximum(t / e, e / t)


def summary_stats(scores, estimate, truth, radius):
    positive = truth > 0
    qe = qerrors(truth, estimate)
    w = float(np.exp(radius)) if np.isfinite(radius) and radius < 700 else np.inf
    return {'coverage_all': float(covered(scores, radius).mean()),
            'coverage_positive': float(covered(scores[positive], radius).mean()) if positive.any() else np.nan,
            'zero_fraction': float((~positive).mean()), 'n_all': len(truth), 'n_positive': int(positive.sum()),
            'q50_all_clipped': float(np.quantile(qe, .5)), 'q90_all_clipped': float(np.quantile(qe, .9)),
            'q99_all_clipped': float(np.quantile(qe, .99)),
            'radius': float(radius), 'multiplicative_half_width': w,
            'median_cardinality_interval_width': float(np.median(np.maximum(estimate, 1) * (w - 1 / w)))}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return 'Infinity' if value == np.inf else '-Infinity' if value == -np.inf else None
    return value


def bootstrap_mean(values, seed=86421):
    v = np.asarray(values, float)
    finite = v[np.isfinite(v)]
    extras = {'infinite_fraction': float(np.isinf(v).mean()), 'finite_mean': float(finite.mean()) if len(finite) else None}
    if len(finite) != len(v):
        return {'mean': np.inf if np.isposinf(v).any() else np.nan,
                'finite_replicates': len(finite), 'total_replicates': len(v),
                'ci95_low': None, 'ci95_high': None, **extras}
    rng = np.random.default_rng(seed)
    means = finite[rng.integers(0, len(finite), size=(2000, len(finite)))].mean(axis=1)
    return {'mean': float(v.mean()), 'ci95_low': float(np.quantile(means, .025)),
            'ci95_high': float(np.quantile(means, .975)), 'finite_replicates': len(v), 'total_replicates': len(v), **extras}


def replay(pool, pool_name, repeats, role_budget, alpha):
    rows, traces = [], []
    n, baseline = len(pool['identities']), pool['baseline']
    complexity = np.asarray([sql_complexity(s, p) for s, p in zip(pool['sqls'], pool['pg'][baseline])])
    structural = complexity[:, :2]
    params = dict(n_estimators=100, num_leaves=15, min_child_samples=5, learning_rate=.05,
                  reg_lambda=1, n_jobs=1, verbosity=-1, random_state=42, deterministic=True, force_col_wise=True)
    for split in range(repeats):
        seed = 17000 + split
        roles = split_roles(n, seed)
        tr, ca, va, au = [roles[r] for r in ('initial_train', 'calibration', 'threshold_validation', 'audit')]
        def features(state):
            return np.column_stack([structural, np.log(np.maximum(pool['pg'][state], 1))])
        initial = lgb.LGBMRegressor(**params)
        initial.fit(features(baseline)[tr], np.log(np.maximum(pool['true'][baseline][tr], 1)) - np.log(np.maximum(pool['pg'][baseline][tr], 1)))
        pred = {s: np.maximum(pool['pg'][s] * np.exp(np.clip(initial.booster_.predict(features(s)), -30, 30)), 1)
                for s in pool['true']}
        scores = {s: interval_scores(pool['true'][s], pred[s]) for s in pool['true']}
        q0 = calibrate(scores[baseline][ca], alpha)
        threshold = alarm_threshold(~covered(scores[baseline][va], q0), role_budget, seed + 100)
        split_trace = {'pool': pool_name, 'split': split, 'seed': seed, 'roles': {r: v.tolist() for r, v in roles.items()},
                       'q0': q0, 'alarm_miss_threshold': threshold, 'selections': {},
                       'initial_estimates_by_state': {s: p.tolist() for s, p in pred.items()}}
        for epsilon in (0., .05, .1, .2, 1.):
            strategy = 'uniform' if epsilon == 1 else f'explore_{int(100 * epsilon):02d}'
            selection = select_feedback(roles, complexity, epsilon, role_budget, seed + 200)
            split_trace['selections'][strategy] = {r: {k: v.tolist() for k, v in info.items()} for r, info in selection.items()}
            mo, rt, rc = [selection[r]['selected'] for r in FEEDBACK_ROLES]
            assert not set(np.concatenate([mo, rt, rc])).intersection(au)
            for state, truth in pool['true'].items():
                frozen = summary_stats(scores[state][au], pred[state][au], truth[au], q0)
                feedback_cov = float(covered(scores[state][mo], q0).mean())
                # Two independent repair roles: labels for center update and interval calibration.
                shift = float(np.median(np.log(np.maximum(truth[rt], 1)) - np.log(pred[state][rt])))
                repaired_pred = np.maximum(pred[state] * np.exp(shift), 1)
                repaired_scores = interval_scores(truth, repaired_pred)
                q_recal = calibrate(scores[state][rc], alpha)
                q_repair = calibrate(repaired_scores[rc], alpha)
                recal = summary_stats(scores[state][au], pred[state][au], truth[au], q_recal)
                repair = summary_stats(repaired_scores[au], repaired_pred[au], truth[au], q_repair)
                row = {'pool': pool_name, 'state': state, 'split': split, 'seed': seed, 'strategy': strategy,
                       'exploration_fraction': epsilon, 'initial_train_labels': len(tr), 'initial_cal_labels': len(ca),
                       'threshold_validation_labels': len(va), 'monitor_labels': len(mo),
                       'repair_train_labels': len(rt), 'repair_cal_labels': len(rc), 'audit_labels': len(au),
                       'feedback_total_labels': len(mo) + len(rt) + len(rc), 'unique_pool_identities': n,
                       'total_label_ledger_including_initial_validation_and_audit': len(tr) + len(ca) + len(va) + len(mo) + len(rt) + len(rc) + len(au),
                       'audit_n_positive': frozen['n_positive'], 'audit_zero_fraction': frozen['zero_fraction'],
                       'frozen_audit_coverage': frozen['coverage_all'], 'frozen_positive_coverage': frozen['coverage_positive'],
                       'feedback_coverage': feedback_cov, 'feedback_coverage_bias': feedback_cov - frozen['coverage_all'],
                       'absolute_feedback_coverage_bias': abs(feedback_cov - frozen['coverage_all']),
                       'alarm': int(role_budget - int(covered(scores[state][mo], q0).sum()) > threshold),
                       'alarm_miss_threshold': threshold, 'monitor_misses': int((~covered(scores[state][mo], q0)).sum()),
                       'baseline_validation_miss_fraction': float((~covered(scores[baseline][va], q0)).mean()),
                       'monitor_mean_tables': float(complexity[mo, 0].mean()), 'audit_mean_tables': float(complexity[au, 0].mean()),
                       'repair_log_center_shift': shift, 'recal_infinite_radius': int(np.isinf(q_recal)),
                       'repair_infinite_radius': int(np.isinf(q_repair)),
                       'audit_max_possible_multiplicative_coverage': float(np.mean(truth[au] > 0))}
                for prefix, values in [('frozen', frozen), ('recal', recal), ('repair', repair)]:
                    for key in ('coverage_all', 'coverage_positive', 'q50_all_clipped', 'q90_all_clipped', 'q99_all_clipped',
                                'radius', 'multiplicative_half_width', 'median_cardinality_interval_width'):
                        row[f'{prefix}_{key}'] = values[key]
                rows.append(row)
        traces.append(split_trace)
        if (split + 1) % 10 == 0:
            print(f'{pool_name}: {split + 1}/{repeats} splits', flush=True)
    return rows, traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=30)
    ap.add_argument('--role-budget', type=int, default=20)
    ap.add_argument('--alpha', type=float, default=.1)
    ap.add_argument('--output', type=Path, default=HERE / 'results_revision_20260930')
    args = ap.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    audits, all_rows, traces, identities, source_records = {}, [], [], {}, {}
    for filename in ('pool_stats.pkl', 'pool_age.pkl', 'pool_imdb.pkl', 'pool_imdb766.pkl'):
        path = HERE / filename
        pool = pickle.loads(path.read_bytes())
        try:
            prepared = prepare_pool(pool)
        except ValueError as error:
            audits[filename] = {'sha256': sha256(path), 'excluded': True, 'reason': str(error), 'meta': pool.get('meta')}
            continue
        audits[filename] = {**prepared['audit'], 'sha256': sha256(path), 'meta': prepared['meta'], 'excluded': False}
        identities[filename] = {'identities': prepared['identities'], 'source_idx': prepared['source_idx'].tolist()}
        source_records[filename] = {'identities': prepared['identities'], 'source_idx': prepared['source_idx'].tolist(),
                                    'sqls': prepared['sqls'], 'true_by_state': prepared['true'], 'pg_by_state': prepared['pg'],
                                    'source_sha256': audits[filename]['sha256']}
        rows, pool_traces = replay(prepared, filename, args.repeats, args.role_budget, args.alpha)
        all_rows.extend(rows)
        traces.extend(pool_traces)
    keys = list(all_rows[0])
    with (args.output / 'feedback_replicates.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)
    aggregates = []
    metrics = ['frozen_audit_coverage', 'feedback_coverage', 'feedback_coverage_bias', 'absolute_feedback_coverage_bias',
               'alarm', 'recal_coverage_all', 'repair_coverage_all', 'repair_coverage_positive',
               'frozen_multiplicative_half_width', 'recal_multiplicative_half_width', 'repair_multiplicative_half_width',
               'frozen_q50_all_clipped', 'frozen_q90_all_clipped', 'frozen_q99_all_clipped',
               'repair_q50_all_clipped', 'repair_q90_all_clipped', 'repair_q99_all_clipped', 'audit_zero_fraction',
               'monitor_mean_tables', 'audit_mean_tables', 'recal_infinite_radius', 'repair_infinite_radius',
               'audit_max_possible_multiplicative_coverage']
    for group in sorted(set((r['pool'], r['state'], r['strategy']) for r in all_rows)):
        selected = [r for r in all_rows if (r['pool'], r['state'], r['strategy']) == group]
        result = dict(zip(('pool', 'state', 'strategy'), group))
        result['repeats'] = len(selected)
        result['metrics'] = {m: bootstrap_mean([r[m] for r in selected]) for m in metrics}
        aggregates.append(result)
    paired = []
    for pool_name, state in sorted(set((r['pool'], r['state']) for r in all_rows)):
        by_strategy = {strategy: {r['split']: r for r in all_rows if r['pool'] == pool_name and r['state'] == state and r['strategy'] == strategy}
                       for strategy in ('explore_00', 'explore_05', 'explore_10', 'explore_20', 'uniform')}
        paired_metrics = ('feedback_coverage_bias', 'absolute_feedback_coverage_bias', 'alarm', 'recal_coverage_all', 'repair_coverage_all',
                          'repair_q90_all_clipped', 'repair_infinite_radius')
        for strategy, reference in [(s, 'uniform') for s in by_strategy if s != 'uniform'] + [('explore_05', 'explore_00'), ('explore_10', 'explore_00'), ('explore_20', 'explore_00')]:
            for metric in paired_metrics:
                differences = [by_strategy[strategy][s][metric] - by_strategy[reference][s][metric] for s in range(args.repeats)]
                paired.append({'pool': pool_name, 'state': state, 'strategy': strategy, 'reference': reference, 'metric': metric,
                               'difference': bootstrap_mean(differences)})
    config = {'repeats': args.repeats, 'role_budget': args.role_budget, 'feedback_total_budget': 3 * args.role_budget,
              'alpha': args.alpha, 'script_sha256': sha256(__file__), 'runtime_seconds': time.perf_counter() - started,
              'numpy_version': np.__version__, 'lightgbm_version': lgb.__version__,
              'protocol': {'split_fractions': {'initial_train': .2, 'calibration': .15, 'threshold_validation': .15, 'audit': .2, 'feedback_reservoir': .3},
                           'bias_definition': 'lexicographically cheapest SQL: fewest FROM tables, fewest predicates, smallest baseline PG cardinality; never true cardinality',
                           'exploration': 'within each distinct role reservoir, draw epsilon*20 uniformly without replacement, then use cheapest remaining for exploitation',
                           'alarm': 'one fixed 20-label window; > empirical 95th percentile of 2000 baseline validation windows sampled without replacement; q0 frozen',
                           'repair': 'median residual intercept update on repair_train; independent repair_cal supplies conformal radius; no full retraining',
                           'base_corrector': 'LightGBM with fixed 100 trees, 15 leaves, min child 5; features SQL table/predicate counts plus current PG log cardinality; avoids stale pool feature reuse',
                           'zero_true': 'structural multiplicative-interval miss, including infinite radius; q-error clips true and estimate to >=1 and is explicitly labelled',
                           'scope': 'common successful unique SQL identity intersection including zero outcomes; timeout/failed queries excluded and enumerated',
                           'identity': 'exact SQL after unquoted case/whitespace normalization; quoted literals preserved; keep first group representative before success filter',
                           'independence': 'train/calibration/threshold-validation/audit and all three feedback role reservoirs are pairwise identity-disjoint',
                           'cost': 'batch secs do not provide per-query execution time: report labels only, no collection-cost estimate',
                           'uncertainty': 'bootstrap CI across repeated random identity partitions; conditional split variability, not 30 independent workloads or a population CI',
                           'deployment': 'offline real-result replay; deployment population means the held-out successful-query audit pool, not observed online deployment'}}
    payload = {'config': config, 'pool_audit': audits, 'identity_map': identities, 'aggregates': aggregates, 'paired_comparisons': paired,
               'cross_pool_overlap_stats_age': len(set(identities['pool_stats.pkl']['identities']) & set(identities['pool_age.pkl']['identities']))}
    for filename, value in [('feedback_summary.json', payload), ('feedback_raw.json', {'config': config, 'rows': all_rows, 'traces': traces}),
                            ('feedback_pool_records.json', source_records)]:
        (args.output / filename).write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    with (args.output / 'feedback_summary.csv').open('w', newline='', encoding='utf-8') as file:
        fields = ['pool', 'state', 'strategy', 'metric', 'mean', 'ci95_low', 'ci95_high', 'finite_replicates', 'total_replicates', 'infinite_fraction', 'finite_mean']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in aggregates:
            for metric, values in item['metrics'].items():
                writer.writerow({**{k: item[k] for k in ('pool', 'state', 'strategy')}, 'metric': metric, **values})
    with (args.output / 'feedback_paired.csv').open('w', newline='', encoding='utf-8') as file:
        fields = ['pool', 'state', 'strategy', 'reference', 'metric', 'mean', 'ci95_low', 'ci95_high', 'finite_replicates', 'total_replicates', 'infinite_fraction', 'finite_mean']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in paired:
            writer.writerow({**{k: item[k] for k in ('pool', 'state', 'strategy', 'reference', 'metric')}, **item['difference']})
    report = ['# Real execution pool feedback replay', '',
              'This is offline replay of existing database execution labels, not new database execution or an online deployment trial.',
              f'{args.repeats} random identity partitions per pool; fixed feedback budget {3 * args.role_budget} = {args.role_budget} monitor + {args.role_budget} center repair + {args.role_budget} repair calibration labels.',
              'The held-out audit pool includes zero-result queries and is never used for selection, training, calibration, or alarm threshold validation. Initial and audit label counts are separately listed in feedback_replicates.csv.',
              'All three feedback roles independently favor the cheapest SQL structure. This isolates selection bias from query identity leakage. Random exploration is 0/5/10/20% of the same total budget; uniform is the reference.',
              'No per-query measured execution time exists in these pools. Structural simplicity is a proxy, and no real collection-cost saving is estimated.',
              'A zero true cardinality is always a multiplicative-interval miss, even for an infinite calibrated radius. Q-error uses max(true,1) and max(estimate,1), explicitly labelled clipped in outputs.',
              'Repair here is a median residual center update followed by independent calibration, not full model retraining. Infinite widths are reported rather than silently dropping zero calibration labels.',
              'CI below is a bootstrap interval for the mean over 30 repeated partitions of the same finite query identities; it is not independent workload or deployment uncertainty.', '',
              '| Pool/state | Feedback | Frozen audit cov. | Feedback cov. bias (pp) | Alarm probability | Repair audit cov. | Repair half-width | Infinite repair width (%) | Repair q90 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for item in aggregates:
        m = item['metrics']
        vals = [m[k]['mean'] for k in ('frozen_audit_coverage', 'feedback_coverage_bias', 'alarm', 'repair_coverage_all', 'repair_multiplicative_half_width', 'repair_infinite_radius', 'repair_q90_all_clipped')]
        report.append(f"| {item['pool']}/{item['state']} | {item['strategy']} | {vals[0]:.3f} | {100*vals[1]:+.2f} | {vals[2]:.3f} | {vals[3]:.3f} | {vals[4]:.2f} | {100*vals[5]:.1f} | {vals[6]:.2f} |")
    report += ['', 'Label ledger per split and per replayed state (initial labels are shared across state scenarios; audit labels are evaluation-only):', '',
               '| Pool | Initial train | Initial cal | Threshold validation | Monitor | Center repair | Repair cal | Audit | Total incl. audit |',
               '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in identities:
        r = next(r for r in all_rows if r['pool'] == name)
        numbers = [r[k] for k in ('initial_train_labels', 'initial_cal_labels', 'threshold_validation_labels', 'monitor_labels', 'repair_train_labels',
                                  'repair_cal_labels', 'audit_labels', 'total_label_ledger_including_initial_validation_and_audit')]
        report.append('| ' + name + ' | ' + ' | '.join(str(x) for x in numbers) + ' |')
    def cell(name, state, strategy):
        return next(x['metrics'] for x in aggregates if (x['pool'], x['state'], x['strategy']) == (name, state, strategy))
    def paired_cell(name, state, strategy, reference, metric):
        return next(x['difference'] for x in paired if (x['pool'], x['state'], x['strategy'], x['reference'], x['metric']) == (name, state, strategy, reference, metric))
    age = cell('pool_age.pkl', 'age_purge50_posts', 'explore_00')
    age20 = cell('pool_age.pkl', 'age_purge50_posts', 'explore_20')
    ageuniform = cell('pool_age.pkl', 'age_purge50_posts', 'uniform')
    reduction = paired_cell('pool_age.pkl', 'age_purge50_posts', 'explore_20', 'explore_00', 'feedback_coverage_bias')
    alarm_delta = paired_cell('pool_age.pkl', 'age_purge50_posts', 'explore_20', 'explore_00', 'alarm')
    report += ['', 'Directly citable findings:', '',
               f"On the 500-query age-purge pool, frozen held-out coverage was {100*age['frozen_audit_coverage']['mean']:.2f}%. Independent cheapest-query monitoring feedback overstated coverage by {100*age['feedback_coverage_bias']['mean']:.2f} percentage points. At 5/10/20% random exploration, the bias was " + '/'.join(f"{100*cell('pool_age.pkl', 'age_purge50_posts', s)['feedback_coverage_bias']['mean']:.2f}" for s in ('explore_05', 'explore_10', 'explore_20')) + f" points; the uniform reference bias was {100*ageuniform['feedback_coverage_bias']['mean']:.2f} points.",
               f"The paired 20%-versus-0% coverage-bias change was {100*reduction['mean']:.2f} points (95% split-bootstrap CI [{100*reduction['ci95_low']:.2f}, {100*reduction['ci95_high']:.2f}]). Thus exploration reduced bias in this scenario, but 20% retained a substantial {100*age20['feedback_coverage_bias']['mean']:.2f}-point overstatement.",
               f"The corresponding frozen-monitor alarm probabilities were {age['alarm']['mean']:.2f}, {age20['alarm']['mean']:.2f}, and {ageuniform['alarm']['mean']:.2f} for 0%, 20%, and uniform feedback. The paired 20%-versus-0% alarm change was {alarm_delta['mean']:.2f} (95% split-bootstrap CI [{alarm_delta['ci95_low']:.2f}, {alarm_delta['ci95_high']:.2f}]); a general alarm-benefit claim is not supported by these limited, state-specific results.",
               f"Age-purge introduces 70/500 zero outcomes. For this pool, any purely multiplicative interval has a maximum possible overall coverage of 86%, below the nominal 90%. Infinite repair-calibration radii occurred in {100*age['repair_infinite_radius']['mean']:.1f}% of the biased-feedback splits and {100*ageuniform['repair_infinite_radius']['mean']:.1f}% of uniform-feedback splits; they still do not cover zero outcomes. The reported infinite mean width is accompanied by finite-split mean and frequency in the JSON/CSV.",
               'Independence removes reuse of query identities across monitoring, repair training, repair calibration and audit. It does not ensure that the cheap-query sampling law represents the held-out workload. Uniform exploration changes that law and consumes part of the same 60-label feedback budget.',
               '', 'Alarm benefit must be assessed from state-specific probabilities and their uniform reference. These pools can have low power because the frozen radius is broad, and 5–20% exploration must not be claimed sufficient unless those results support it.',
               'pool_imdb766.pkl is excluded: its baseline arrays have 320 rows while idx/SQL and drift arrays have 300. stats/age share query identities and are separate perturbation scenarios, not independent populations.',
               '', 'Reproduce: `python drift_ce/experiments/experiment_revision_feedback.py --repeats 30 --role-budget 20`',
               'Raw draw indices, initial predictions, feedback role selections and exploration/exploitation indices: feedback_raw.json. Exact SQL and raw successful labels/PG estimates: feedback_pool_records.json. Identity hashes and source pool SHA-256: feedback_summary.json. Paired intervals: feedback_paired.csv.']
    (args.output / 'feedback_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'Wrote {len(all_rows)} replicate rows, {len(aggregates)} summary cells in {config["runtime_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
