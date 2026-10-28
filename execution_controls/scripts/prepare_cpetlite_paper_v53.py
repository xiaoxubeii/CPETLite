"""Clone the complete paper design into a verified 5.3.0 experiment namespace."""
import json
from pathlib import Path

from run_cpetlite_paper_v5 import ROOT, config_file, verified_cohort
from cpet_foundation.training.runner import load_yaml


def rebind(value):
    if isinstance(value, str):
        return value.replace('cpetlite_paper_standard_v5_20261008', 'cpetlite_paper_standard_v53_20261010').replace(
            'cpetlite_paper_v5', 'cpetlite_paper_v53').replace(
            'cpetlite_teacher_only_paper_v5', 'cpetlite_teacher_only_paper_v53')
    if isinstance(value, dict):
        return {k: rebind(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rebind(v) for v in value]
    return value


def main():
    import yaml
    folder=ROOT.parent/'cpet_former/artifacts/dataset/derivatives/cpetlite_paper_standard_v53_20261010'
    report=verified_cohort(folder)
    if report.get('reserved_cohort_exclusion_enabled') is not False:
        raise ValueError('The requested cohort must retain human-comparison records')
    names=['train_cpetlite_paper_v5_'+s for s in ['teacher','measured','package','weighted','gru_measured','gru_package']]
    names+=['export_cpetlite_paper_v5']
    for name in names:
        # Keep inheritance and the underlying architecture/optimization unchanged.
        cfg=yaml.safe_load((ROOT/'configs'/f'{name}.yaml').read_text())
        config_file(name.replace('paper_v5','paper_v53'),rebind(cfg))
    for arch in ('direct','gru'):
        matrix=rebind(yaml.safe_load((ROOT/f'configs/cpetlite_paper_v5_{arch}.yaml').read_text()))
        matrix['evaluation_counts']={'internal':report['splits']['val']['examinations'],
                                     'external':report['splits']['test']['examinations']}
        config_file(f'cpetlite_paper_v53_{arch}',matrix)
        for stage,spec in matrix['stages'].items():
            cfg=load_yaml(ROOT/spec['base_config'])
            assert cfg['epochs']==80 and cfg['early_stopping']['patience']==12
            if stage!='measured':
                assert cfg['supervision_mode']=='teacher_only'
                assert all(cfg.get(k) is None for k in ['source_h5','split_path','normalization_stats_path','teacher_checkpoint'])
            else:
                assert (ROOT/cfg['source_h5']).resolve()==folder/'dataset.h5'
    print(json.dumps({'config_prefix':'cpetlite_paper_v53','student_fits':15,'evaluations':30,'cohort':report['splits']}))


if __name__=='__main__':
    main()
