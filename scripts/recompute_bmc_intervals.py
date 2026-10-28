#!/usr/bin/env python3
"""Independently replay the executed shared-examination bootstrap contract."""
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from zipfile import ZipFile

import numpy as np

from bmc_bootstrap import paired_intervals
import build_v552_paper as principal
import build_paired_controls as control

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'content/bmc_author_review_20261028/audit'
METRICS = [x[0] for x in principal.METRICS]


def frozen_metric_functions():
    capsule = ROOT / 'supplementary/v552_package/reproducibility/cpetlite-v552-code-config.zip'
    with ZipFile(capsule) as z:
        source = z.read('scripts/summarize_distillation_component_ablation.py')
    parsed = ast.parse(source)
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('_case_components', '_component_metrics')]
    assert len(functions) == 2
    namespace = dict(np=np, Path=Path, PRIMARY_CHANNELS=('VO2', 'VCO2', 'VE'))
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<verified frozen metric functions>', 'exec'), namespace)
    return namespace['_case_components'], namespace['_component_metrics'], hashlib.sha256(source).hexdigest()


def components(archive_path, pattern, expected_hash, reader):
    assert principal.digest(archive_path) == expected_hash
    result = {}
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            match = re.search(pattern, member.name)
            if match:
                key = tuple(match.groups())
                assert key not in result, 'Duplicate experiment evidence key'
                result[key] = reader(io.BytesIO(archive.extractfile(member).read()))
    return result


def main():
    reader, metric, source_sha = frozen_metric_functions()
    primary = components(principal.ARCHIVE,
        '/' + re.escape(principal.EXPERIMENT) + r'_(direct|gru)/(measured|ordinary_package)_seed(17|42|73)_(internal|external)_predictions\.npz$',
        principal.ARCHIVE_SHA, reader)
    controls = components(control.SOURCE / 'RESULTS.tar.gz',
        '/' + re.escape(control.FAMILY) + r'_(direct|gru)/(full_package|mean_endpoint|protocol_static)_seed(17|42|73)_(internal|external)_predictions\.npz$',
        control.ARCHIVE_SHA, reader)
    assert len(primary) == 24 and len(controls) == 36
    rows = []
    for family, data, comparisons in [('principal', primary, [('ordinary_package', 'measured')]),
                                      ('controls', controls, [('mean_endpoint', 'full_package'), ('protocol_static', 'full_package')])]:
        for architecture in ['direct', 'gru']:
            for domain in ['internal', 'external']:
                for current, previous in comparisons:
                    pairs = []
                    common_ids = None
                    for seed in ['17', '42', '73']:
                        ids_a, a = data[(architecture, current, seed, domain)]
                        ids_b, b = data[(architecture, previous, seed, domain)]
                        assert np.array_equal(ids_a, ids_b)
                        if common_ids is None:
                            common_ids = ids_a
                        else:
                            assert len(set(ids_a)) == len(ids_a) and set(ids_a) == set(common_ids)
                            order = {identifier: i for i, identifier in enumerate(ids_a)}
                            indices = np.asarray([order[identifier] for identifier in common_ids])
                            a = {k: values[indices] for k, values in a.items()}
                            b = {k: values[indices] for k, values in b.items()}
                        for key in ['core_n', 'at_metric_mask', 'peak_mask']:
                            assert np.array_equal(a[key], b[key])
                        pairs.append((a, b))
                    bootstrap_seed = 42 if domain == 'internal' else 142
                    values = paired_intervals(pairs, metric, draws=2000, seed=bootstrap_seed)
                    for i, name in enumerate(METRICS):
                        rows.append(dict(family=family, architecture=architecture, domain=domain,
                            contrast=current + '_minus_' + previous, metric=name,
                            mean_difference=float(values[0, i]), ci95_low=float(values[1, i]),
                            ci95_high=float(values[2, i]), examinations=len(common_ids), seeds=3,
                            bootstrap_draws=2000, bootstrap_seed=bootstrap_seed,
                            index_rule='align examination identifiers, then share examination indices across paired models and all seeds'))
    # The actual executed evaluator uses shared indices and domain-specific RNG
    # seeds. The generic ablation summarizer in the capsule is not this run's
    # entry point. Require numerical agreement with the accepted intervals.
    accepted_primary = json.loads((DEST / 'principal_results.json').read_text())
    accepted_controls = json.loads((DEST / 'control_results.json').read_text())
    for row in rows:
        if row['family'] == 'principal':
            expected = accepted_primary[{'direct': 'Transformer', 'gru': 'GRU'}[row['architecture']]]['paired'][row['contrast'] + ':' + row['domain']]['metrics'][row['metric']]
            point = expected['mean_paired_difference']
        else:
            arm = row['contrast'].split('_minus_full_package')[0]
            expected = next(r for r in accepted_controls['paired'] if (r['architecture'], r['control'], r['domain'], r['metric']) == (row['architecture'], arm, row['domain'], row['metric']))
            point = expected['mean_paired_difference']
        assert np.isclose(row['mean_difference'], point, atol=1e-6, rtol=0)
        for key in ['ci95_low', 'ci95_high']:
            assert np.isclose(row[key], expected[key], atol=1e-6, rtol=0), (row['family'], row['architecture'], row['domain'], row['metric'], key, row[key], expected[key])
    (DEST / 'paired_interval_replay.json').write_text(json.dumps(dict(
        method='2000 common-examination bootstrap draws; fixed teacher and three fitted student seeds; percentile 95% intervals',
        finding='The actual executed evaluator already shares examination indices across seeds; internal RNG seed 42 and external seed 142. Accepted intervals are preserved after independent replay.',
        metric_function_source_sha256=source_sha, principal_archive_sha256=principal.ARCHIVE_SHA,
        control_archive_sha256=control.ARCHIVE_SHA, examination_identifier_alignment=True,
        training_performed=False, rows=rows), indent=2) + '\n')
    with (DEST / 'paired_interval_replay.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Verified 36 accepted metric intervals for 12 model contrasts with common examinations.', flush=True)
    for row in rows:
        print(f"{row['family']} {row['architecture']} {row['domain']} {row['contrast']} {row['metric']}: {row['mean_difference']:.6f} [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}]")


if __name__ == '__main__':
    main()
