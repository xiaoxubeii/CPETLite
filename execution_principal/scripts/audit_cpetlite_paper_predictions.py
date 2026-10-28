"""Independently replay full-population predictions against derivative source rows."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


CORE = ['VO2', 'VCO2', 'VE']


def source_endpoints(group):
    metadata_ids = group['metadata_table/Examination_ID'].asstr()[:]
    eligible = dict(zip(metadata_ids, zip(group['supervision_eligibility/at'][:],
                                         group['supervision_eligibility/peak'][:])))
    targets = group['targets/raw']
    ids = targets['Examination_ID'].asstr()[:]
    assert len(ids) == len(set(ids)) == len(eligible) and set(ids) == set(eligible)
    at = pd.to_numeric(targets['Time_at_AT'][:], errors='coerce').astype(float)
    peak = pd.to_numeric(targets['VO2_kg_peak_phase1'][:], errors='coerce').astype(float)
    at_valid = group['validity_masks/targets/Time_at_AT'][:].astype(bool) & np.isfinite(at)
    peak_valid = group['validity_masks/targets/VO2_kg_peak_phase1'][:].astype(bool) & np.isfinite(peak)
    return {str(exam): (at[i], peak[i], bool(at_valid[i] and eligible[exam][0]),
                       bool(peak_valid[i] and eligible[exam][1])) for i, exam in enumerate(ids)}


def verify_endpoint_source(predictions, endpoints, stem):
    ids = predictions['exam_id'].astype(str)
    expected = np.asarray([endpoints[exam] for exam in ids])
    at = predictions['at_metric_mask'] > 0
    peak = predictions['vo2peak_mask'] > 0
    assert np.array_equal(at, expected[:, 2].astype(bool)), (stem, 'source AT mask mismatch')
    assert np.array_equal(peak, expected[:, 3].astype(bool)), (stem, 'source peak mask mismatch')
    assert np.allclose(predictions['at_target_s'][at], expected[at, 0], rtol=2e-6, atol=2e-5), (stem, 'source AT label mismatch')
    assert np.allclose(predictions['vo2peak_target'][peak], expected[peak, 1], rtol=2e-6, atol=2e-5), (stem, 'source peak label mismatch')


def select_reference(time, values, validity, max_len=192):
    keep = np.flatnonzero(np.isfinite(time))
    keep = keep[np.argsort(time[keep], kind='stable')]
    if len(keep) > max_len:
        keep = keep[np.round(np.linspace(0, len(keep)-1, max_len)).astype(int)]
    value = np.zeros((max_len, len(CORE)), dtype=float)
    mask = np.zeros_like(value, dtype=bool)
    value[:len(keep)] = values[keep]
    mask[:len(keep)] = validity[keep] & np.isfinite(values[keep])
    return value, mask


def audit(dataset, prefix, root):
    verified = 0
    common_mean = common_std = None
    with h5py.File(dataset) as h:
        source = {}
        for part, domain in [('val', 'internal'), ('test', 'external')]:
            g = h[f'splits/{part}']
            ids = g['features/Examination_ID'].asstr()[:]
            time = pd.to_numeric(g['features/Time'][:], errors='coerce').astype(float)
            values = np.column_stack([pd.to_numeric(g[f'features/{k}'][:], errors='coerce') for k in CORE])
            validity = np.column_stack([g[f'validity_masks/features/{k}'][:].astype(bool) for k in CORE])
            reference = {}
            groups = pd.Series(np.arange(len(ids))).groupby(ids, sort=False)
            for exam, positions in groups:
                p = positions.to_numpy()
                reference[str(exam)] = select_reference(time[p], values[p], validity[p])
            expected = set(g['metadata_table/Examination_ID'].asstr()[:])
            assert set(reference) == expected
            source[domain] = (reference, expected, source_endpoints(g))
        for arch in ['direct', 'gru']:
            directory = root/f'artifacts/eval/{prefix}_{arch}'
            summary = json.loads((directory/'summary.json').read_text())
            for record in summary['records']:
                stem = f"{record['stage']}_seed{record['seed']}_{record['domain']}"
                manifest = json.loads((directory/f'{stem}_manifest.json').read_text())
                metric = json.loads((directory/f'{stem}.json').read_text())
                reference, expected, endpoints = source[record['domain']]
                with np.load(manifest['predictions_path']) as z:
                    ids = z['exam_id'].astype(str)
                    assert len(ids) == len(set(ids)) == len(expected)
                    assert set(ids) == expected
                    verify_endpoint_source(z, endpoints, stem)
                    idx = [list(z['channel_names'].astype(str)).index(k) for k in CORE]
                    mean, std = z['feature_mean'][:, idx], z['feature_std'][:, idx]
                    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
                    assert np.allclose(mean, mean[0], rtol=0, atol=0)
                    assert np.allclose(std, std[0], rtol=0, atol=0)
                    if common_mean is None:
                        common_mean, common_std = mean[0], std[0]
                    assert np.array_equal(common_mean, mean[0]) and np.array_equal(common_std, std[0])
                    target = z['trajectory_target'][:, :, idx].astype(float)
                    channel_mask = z['target_channel_mask'][:, :, idx] > 0
                    for i, exam in enumerate(ids):
                        value, mask = reference[exam]
                        assert np.array_equal(channel_mask[i], mask), (stem, 'source mask mismatch')
                        normalized = (value-mean[i])/std[i]
                        assert np.allclose(target[i][mask], normalized[mask], rtol=2e-5, atol=2e-5), (stem, 'source value mismatch')
                    valid = channel_mask & (z['task_mask'][:, :, None] > 0)
                    pred = z['trajectory_pred'][:, :, idx].astype(float)
                    assert np.isfinite(pred[valid]).all()
                    rmse = np.sqrt(np.square(pred-target)[valid].mean())
                    assert np.isclose(rmse, metric['primary_trajectory_rmse'], rtol=2e-6, atol=2e-6), (stem, 'core RMSE mismatch')
                    at = z['at_metric_mask'] > 0
                    peak = z['vo2peak_mask'] > 0
                    assert int(at.sum()) == record['at_count'] == metric['at_time_s_evaluable_count']
                    assert int(peak.sum()) == record['peak_count'] == metric['vo2peak_evaluable_count']
                    at_prediction = z['time_start_s'].astype(float) + z['at_pred_norm'].astype(float)*z['duration_s'].astype(float)
                    at_mae = np.abs(at_prediction[at]-z['at_target_s'].astype(float)[at]).mean()
                    peak_mae = np.abs(z['vo2peak_pred'].astype(float)[peak]-z['vo2peak_target'].astype(float)[peak]).mean()
                    assert np.isclose(at_mae, metric['at_time_s_mae'], rtol=2e-6, atol=2e-5)
                    assert np.isclose(peak_mae, metric['vo2peak_mae'], rtol=2e-6, atol=2e-5)
                verified += 1
    assert verified == 30
    result = {'status': 'verified', 'experiment': prefix, 'evaluations': verified,
              'unique_examination_sets_verified': True, 'source_row_values_and_masks_verified': True,
              'source_endpoint_labels_and_masks_verified': True,
              'frozen_scale_consistency_verified': True, 'all_headline_metrics_recomputed': True}
    (root/f'artifacts/eval/{prefix}/PREDICTION_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--config-prefix', required=True)
    args = parser.parse_args()
    audit(args.dataset_dir.resolve()/'dataset.h5', args.config_prefix, Path(__file__).resolve().parents[1])
