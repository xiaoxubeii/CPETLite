"""Prepare isolated current-paper experiments from a verified Standard5.4.1 derivative."""
import json
from pathlib import Path

import yaml

from run_cpetlite_paper_v5 import ROOT, config_file, verified_cohort
from cpet_foundation.training.runner import load_yaml


def rebind(value):
    if isinstance(value, str):
        return (value.replace('cpetlite_paper_standard_v53_20261010', 'cpetlite_paper_standard_v541_20261012')
                .replace('cpetlite_paper_v53', 'cpetlite_paper_v541')
                .replace('cpetlite_teacher_only_paper_v53', 'cpetlite_teacher_only_paper_v541'))
    if isinstance(value, dict):
        return {k: rebind(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rebind(v) for v in value]
    return value


def main():
    folder = ROOT.parent/'cpet_former/artifacts/dataset/derivatives/cpetlite_paper_standard_v541_20261012'
    report = verified_cohort(folder)
    manifest = json.loads((folder/'DERIVATIVE_MANIFEST.json').read_text())
    if (report.get('reserved_cohort_exclusion_enabled') is not False
            or manifest.get('source_relative_path') != 'registry/registry.h5'
            or manifest['recipe']['source']['stage'] != 'registry'):
        raise ValueError('Current paper requires accepted registry and no human-comparison reservation')
    names = ['train_cpetlite_paper_v53_'+s for s in
             ['teacher', 'measured', 'package', 'weighted', 'gru_measured', 'gru_package']]
    names += ['export_cpetlite_paper_v53']
    for name in names:
        config_file(name.replace('paper_v53', 'paper_v541'),
                    rebind(yaml.safe_load((ROOT/'configs'/f'{name}.yaml').read_text())))
    for arch in ['direct', 'gru']:
        matrix = rebind(yaml.safe_load((ROOT/f'configs/cpetlite_paper_v53_{arch}.yaml').read_text()))
        matrix['evaluation_counts'] = {'internal': report['splits']['val']['examinations'],
                                       'external': report['splits']['test']['examinations']}
        config_file(f'cpetlite_paper_v541_{arch}', matrix)
        for stage, spec in matrix['stages'].items():
            cfg = load_yaml(ROOT/spec['base_config'])
            assert cfg['epochs'] == 80 and cfg['early_stopping']['patience'] == 12
            if stage != 'measured':
                assert cfg['supervision_mode'] == 'teacher_only'
                assert all(cfg.get(k) is None for k in ['source_h5', 'split_path', 'normalization_stats_path', 'teacher_checkpoint'])
            else:
                assert (ROOT/cfg['source_h5']).resolve() == folder/'dataset.h5'
    print(json.dumps({'config_prefix': 'cpetlite_paper_v541', 'student_fits': 15,
                      'evaluations': 30, 'cohort': report['splits']}))


if __name__ == '__main__':
    main()
