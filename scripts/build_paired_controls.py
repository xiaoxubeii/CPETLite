#!/usr/bin/env python3
"""Accepted paired-control aggregates -> separate, clinical-array-free paper tables."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'cpet_standard/results/cpetlite_review_controls_cqa1_20261021_r1/collected_r2'
ARCHIVE_SHA = 'e6d9501525e45ff91b4bbfe4c6ef66f8d7a23c879ba6cd3a1fe265799b83e454'
FAMILY = 'cpetlite_review_controls_v552_20261021'
ARCHITECTURES = ('direct', 'gru')
ARMS = ('full_package', 'mean_endpoint', 'protocol_static')
DOMAINS = ('internal', 'external')
SEEDS = (17, 42, 73)
METRICS = ('primary_trajectory_rmse', 'at_time_s_mae', 'vo2peak_mae')
MODEL = dict(direct='Transformer', gru='GRU')
LABEL = dict(full_package='Complete', mean_endpoint='Mean + endpoints', protocol_static='Protocol/static')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_collection(source, expected_sha):
    """Validate the accepted collection before returning any reportable result."""
    source = Path(source)
    acceptance = json.loads((source / 'ACCEPTANCE.json').read_text())
    assert acceptance['status'] == 'verified', 'Unaccepted control collection'
    assert acceptance['student_fits'] == 18 and acceptance['evaluations'] == 36
    assert acceptance['teacher_retrained'] is False
    assert acceptance['archive_sha256'] == expected_sha == digest(source / 'RESULTS.tar.gz'), 'Archive checksum mismatch'
    inventory = json.loads((source / 'RESULTS_INVENTORY.json').read_text())
    member_sha = {}
    with tarfile.open(source / 'RESULTS.tar.gz') as archive:
        def read(name):
            member = archive.getmember(name)
            assert member.isfile(), 'Non-file evidence member'
            payload = archive.extractfile(member).read()
            expected = inventory[name]
            assert len(payload) == expected['bytes'], 'Member size mismatch: ' + name
            actual = hashlib.sha256(payload).hexdigest()
            assert actual == expected['sha256'], 'Member checksum mismatch: ' + name
            member_sha[name] = actual
            return json.loads(payload)

        plan = read('PLAN.json')
        assert member_sha['PLAN.json'] == acceptance['plan_sha256'], 'Unbound control plan'
        assert plan['expected_fits'] == 18 and plan['expected_evaluations'] == 36
        assert plan['teacher_retrained'] is False and plan['seeds'] == list(SEEDS)
        assert plan['split_counts'] == dict(train=11404, val=1270, test=2385)
        expected_fits = set(itertools.product(ARCHITECTURES, ARMS, SEEDS))
        expected_evals = {(*fit, domain) for fit in expected_fits for domain in DOMAINS}
        assert {(t['architecture'], t['arm'], t['seed']) for t in plan['tasks']} == expected_fits
        base = 'foundation/artifacts/eval/' + FAMILY + '/'
        fit_audit = read(base + 'FIT_AND_COVERAGE_AUDIT.json')
        prediction_audit = read(base + 'PREDICTION_AUDIT.json')
        assert fit_audit['status'] == prediction_audit['status'] == 'verified'
        assert fit_audit['student_fits'] == 18 and fit_audit['evaluation_count'] == 36
        assert fit_audit['teacher_retrained'] is False and prediction_audit['evaluations'] == 36
        assert len(fit_audit['fits']) == 18 and len(fit_audit['evaluations']) == 36
        assert {(f['architecture'], f['arm'], f['seed']) for f in fit_audit['fits']} == expected_fits
        assert {(f['architecture'], f['arm'], f['seed'], f['domain']) for f in fit_audit['evaluations']} == expected_evals
        for key in ('unique_examination_sets_verified', 'source_row_values_and_masks_verified',
                    'source_endpoint_labels_and_masks_verified', 'frozen_scale_consistency_verified',
                    'all_headline_metrics_recomputed'):
            assert prediction_audit[key] is True, 'Missing prediction evidence: ' + key
        records, summary, pairs = [], [], []
        for architecture in ARCHITECTURES:
            data = read(f'foundation/artifacts/eval/{FAMILY}_{architecture}/summary.json')
            seen = set()
            for row in data['records']:
                key = row['stage'], row['seed'], row['domain']
                assert key not in seen, 'Duplicate seed metric'
                seen.add(key)
                assert all(math.isfinite(row['metrics'][m]) and row['metrics'][m] >= 0 for m in METRICS)
                assert row['at_count'] == (909 if row['domain'] == 'internal' else 2375), 'AT denominator mismatch'
                assert row['peak_count'] == (1270 if row['domain'] == 'internal' else 2385), 'Peak denominator mismatch'
                records.append(dict(architecture=architecture, arm=row['stage'], seed=row['seed'],
                                    domain=row['domain'], examinations=row['peak_count'],
                                    at_count=row['at_count'], peak_count=row['peak_count'], **row['metrics']))
            assert seen == set(itertools.product(ARMS, SEEDS, DOMAINS)), 'Incomplete seed/domain coverage'
            for arm, domain in itertools.product(ARMS, DOMAINS):
                selected = [r for r in records if (r['architecture'], r['arm'], r['domain']) == (architecture, arm, domain)]
                result = dict(architecture=architecture, arm=arm, domain=domain,
                              examinations=selected[0]['examinations'], at_count=selected[0]['at_count'],
                              peak_count=selected[0]['peak_count'])
                for metric in METRICS:
                    values = [r[metric] for r in selected]
                    mean, sd = statistics.mean(values), statistics.stdev(values)
                    archived = data['summary'][arm + ':' + domain][metric]
                    assert math.isclose(mean, archived['mean'], abs_tol=1e-12)
                    assert math.isclose(sd, archived['sd'], abs_tol=1e-12)
                    result[metric + '_mean'], result[metric + '_sd'] = mean, sd
                summary.append(result)
            for arm, domain in itertools.product(ARMS[1:], DOMAINS):
                result = data['paired'][arm + '_minus_full_package:' + domain]
                assert result['seeds'] == list(SEEDS) and result['conditional_on_fitted_models'] is True
                assert result['bootstrap_unit'] == 'examination' and result['bootstrap_draws'] == 2000
                for metric in METRICS:
                    interval = result['metrics'][metric]
                    assert all(math.isfinite(v) for v in interval.values())
                    assert interval['ci95_low'] <= interval['ci95_high']
                    left = next(r for r in summary if (r['architecture'], r['arm'], r['domain']) == (architecture, arm, domain))
                    right = next(r for r in summary if (r['architecture'], r['arm'], r['domain']) == (architecture, ARMS[0], domain))
                    # Native scalar evaluators and per-examination paired replay
                    # use different floating-point reductions. Keep this bound
                    # well below the printed precision (observed AT delta <1 us).
                    tolerance = 1e-6 if metric == 'at_time_s_mae' else 1e-8
                    assert math.isclose(interval['mean_paired_difference'], left[metric + '_mean'] - right[metric + '_mean'], abs_tol=tolerance)
                    pairs.append(dict(architecture=architecture, control=arm, domain=domain, metric=metric, **interval))
    fit_rows = [{k: f[k] for k in ('architecture', 'arm', 'seed', 'epochs', 'best_epoch', 'monitor')}
                for f in fit_audit['fits']]
    assert all(f['monitor'] == 'teacher_core_trajectory_rmse' and 1 <= f['best_epoch'] <= f['epochs'] <= 150 for f in fit_rows)
    return dict(status='verified', archive_sha256=expected_sha, member_sha256=member_sha,
                input_sha256={p: digest(source / p) for p in ('ACCEPTANCE.json', 'RESULTS_INVENTORY.json')},
                dataset_sha256=plan['dataset_sha256'], teacher_package_sha256=plan['package_sha256'],
                teacher_sha256=plan['teacher_sha256'], student_fits=18, evaluations=36,
                teacher_retrained=False, seeds=list(SEEDS), split_counts=plan['split_counts'],
                selection=plan['selection'], max_epochs=plan['max_epochs'],
                early_stopping_patience=plan['early_stopping_patience'], min_delta=plan['min_delta'],
                paired_sign='control minus complete supervision, within this runtime only',
                bootstrap_unit='examination', bootstrap_draws=2000, conditional_on_fitted_models=True,
                records=records, summary=summary, paired=pairs, fits=fit_rows)


def summary_tex(rows):
    lines = [r'\begin{tabular}{llrrr}\toprule',
             r'Model & Control arm & Core RMSE & AT MAE (s) & Peak MAE \\\midrule']
    for row in rows:
        values = [f"${row[m + '_mean']:.{p}f}\\pm{row[m + '_sd']:.{p}f}$" for m, p in zip(METRICS, (4, 2, 3))]
        lines.append(' & '.join([MODEL[row['architecture']], LABEL[row['arm']], *values]) + r' \\')
    return '\n'.join([*lines, r'\bottomrule\end{tabular}', ''])


def generate(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('summary', 'records', 'paired', 'fits'):
        with (output / (name + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report[name][0]))
            writer.writeheader()
            writer.writerows(report[name])
    for domain in DOMAINS:
        (output / (domain + '.tex')).write_text(summary_tex([r for r in report['summary'] if r['domain'] == domain]))
    for domain in DOMAINS:
        lines = [r'\begin{tabular}{llrrrrr}\toprule',
                 r'Model & Arm & Seed & $n_{\rm AT}$ & Core RMSE & AT MAE (s) & Peak MAE \\\midrule']
        for row in report['records']:
            if row['domain'] != domain:
                continue
            values = [f'{row[m]:.{p}f}' for m, p in zip(METRICS, (4, 2, 3))]
            lines.append(' & '.join([MODEL[row['architecture']], LABEL[row['arm']], str(row['seed']),
                                      str(row['at_count']), *values]) + r' \\')
        (output / ('seeds_' + domain + '.tex')).write_text('\n'.join([*lines, r'\bottomrule\end{tabular}', '']))
    for domain in DOMAINS:
        lines = [r'\begin{tabular}{lllrrr}\toprule',
                 r'Model & Control minus complete & Metric & Difference & Lower 95\% & Upper 95\% \\\midrule']
        for row in report['paired']:
            if row['domain'] != domain:
                continue
            metric = dict(zip(METRICS, ('Core RMSE', 'AT MAE (s)', 'Peak MAE')))[row['metric']]
            p = 4 if row['metric'] == METRICS[0] else 3
            values = [f"{row[k]:.{p}f}" for k in ('mean_paired_difference', 'ci95_low', 'ci95_high')]
            lines.append(' & '.join([MODEL[row['architecture']], LABEL[row['control']], metric, *values]) + r' \\')
        (output / ('paired_' + domain + '.tex')).write_text('\n'.join([*lines, r'\bottomrule\end{tabular}', '']))
    report['output_sha256'] = {p.name: digest(p) for p in sorted(output.iterdir()) if p.suffix in ('.csv', '.tex')}
    report['generator_sha256'] = digest(Path(__file__))
    (output / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    from reporting_runtime import require_runtime
    require_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--archive-sha256', default=ARCHIVE_SHA, help='Explicit trusted archive binding (default: accepted clinical controls)')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'supplementary/v552_controls')
    args = parser.parse_args()
    report = load_collection(args.source, args.archive_sha256)
    generate(report, args.output_dir)
    print(json.dumps(dict(status=report['status'], fits=18, evaluations=36,
                          summary_rows=len(report['summary']), seed_rows=len(report['records']),
                          paired_rows=len(report['paired']), output=str(args.output_dir))))


if __name__ == '__main__':
    main()
