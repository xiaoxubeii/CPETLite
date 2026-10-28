"""Fail-closed coverage, population, hash and access audit of a full paper rerun."""
import argparse
import json
import math
from pathlib import Path

import h5py
import yaml

from cpet_foundation.data.distillation_package import audit_distillation_package
from cpet_foundation.training.runner import load_yaml
from run_cpetlite_paper_v5 import ROOT, digest, verified_cohort, verify_existing_checkpoint


METRICS = ('primary_trajectory_rmse', 'at_time_s_mae', 'vo2peak_mae')


def checked_training_budget(config, base, seed):
    """Accept only the selected frozen matrix's budget and stopping policy."""
    if config.get('seed') != seed:
        raise ValueError('Generated training seed differs from matrix')
    if config.get('epochs') != base.get('epochs') or int(base.get('epochs', 0)) < 1:
        raise ValueError('Generated training budget differs from matrix base recipe')
    if config.get('early_stopping') != base.get('early_stopping'):
        raise ValueError('Generated early stopping differs from matrix base recipe')


def checked_evaluation(stem, checkpoint, dataset, split, report):
    manifest=json.loads(Path(str(stem)+'_manifest.json').read_text())
    metrics=json.loads(stem.with_suffix('.json').read_text())
    n=report['splits'][split]['examinations']
    if manifest['requested_examinations']!=n or manifest['loaded_examinations']!=n:
        raise ValueError(f'Evaluation count differs from full cohort: {stem}')
    for field in ('source_h5_sha256', 'split_sha256'):
        if manifest[field]!=dataset:
            raise ValueError(f'Evaluation dataset hash mismatch: {stem}')
    if manifest['checkpoint_sha256']!=digest(checkpoint):
        raise ValueError(f'Evaluation checkpoint hash mismatch: {stem}')
    if Path(manifest['checkpoint']).resolve()!=checkpoint.resolve():
        raise ValueError(f'Evaluation checkpoint mismatch: {stem}')
    if manifest['input_format']!='v5_derivative_split' or manifest['normalization_fit_population']!='internal_training_hdf5':
        raise ValueError('Evaluation uses another normalization/input contract')
    if manifest['normalization_stats_path'] is not None:
        raise ValueError('Evaluation resolves legacy normalization')
    if metrics['at_time_s_evaluable_count']!=report['splits'][split]['at_eligible']:
        raise ValueError(f'AT eligibility mismatch: {stem}')
    if metrics['vo2peak_evaluable_count']!=report['splits'][split]['peak_eligible']:
        raise ValueError(f'Peak eligibility mismatch: {stem}')
    if any(not math.isfinite(metrics[k]) for k in METRICS):
        raise ValueError('Non-finite headline outcome')
    return metrics


def audit(folder, prefix):
    report=verified_cohort(folder)
    dataset=digest(folder/'dataset.h5')
    matrices=[yaml.safe_load((ROOT/f'configs/{prefix}_{arch}.yaml').read_text()) for arch in ('direct','gru')]
    assert matrices[0]['package']==matrices[1]['package']
    package=ROOT/matrices[0]['package'];package_hash=digest(package)
    audit_distillation_package(package)
    teacher=ROOT/matrices[0]['teacher_checkpoint']
    matched = None
    if any(m.get('matched_inputs') for m in matrices):
        if not all(m.get('matched_inputs') for m in matrices):
            raise ValueError('Paper matrices disagree on matched input mode')
        from cpet_foundation.data.matched_inputs import checked_matched_package
        matched=checked_matched_package(package,source=folder/'dataset.h5',teacher=teacher)
    with h5py.File(package) as h:
        assert set(h['splits'])=={'train','validation'}
        metadata=json.loads(h.attrs['metadata_json'])
        assert metadata['contains_measured_full_cpet_targets'] is False
        assert metadata['source_h5_sha256']==dataset
        assert metadata['normalization_stats_sha256']==dataset
        assert metadata['split_sha256']==dataset
        assert metadata['teacher_checkpoint_sha256']==digest(teacher)
        for part,split in [('train','train'),('validation','val')]:
            assert len(h[f'splits/{part}/samples/phase_mask'])==report['splits'][split]['examinations']
            assert 'phase_mask' in h[f'splits/{part}/samples']
    teacher_manifest=verify_existing_checkpoint(teacher)
    assert teacher_manifest['train_samples']==report['splits']['train']['examinations']
    assert teacher_manifest['validation_samples']==report['splits']['val']['examinations']
    assert Path(teacher_manifest['source_h5']).resolve()==(folder/'dataset.h5').resolve()
    fitted,evaluated,isolated,runs=0,0,0,[]
    for arch,matrix in zip(('direct','gru'),matrices):
        assert matrix['seeds']==[17,42,73]
        summary=json.loads((ROOT/matrix['evaluation_dir']/'summary.json').read_text())
        expected_records={(stage,seed,domain) for stage in matrix['stages'] for seed in matrix['seeds'] for domain in ('internal','external')}
        assert {(r['stage'],r['seed'],r['domain']) for r in summary['records']}==expected_records
        for stage,spec in matrix['stages'].items():
            for seed in matrix['seeds']:
                modeldir=ROOT/spec['output_template'].format(seed=seed)
                checkpoint=modeldir/'model.pt';manifest=verify_existing_checkpoint(checkpoint)
                assert manifest['train_samples']==report['splits']['train']['examinations']
                assert manifest['validation_samples']==report['splits']['val']['examinations']
                cfg=load_yaml(ROOT/f'configs/{matrix["config_prefix"]}_{stage}_seed{seed}.yaml')
                checked_training_budget(cfg, load_yaml(ROOT/spec['base_config']), seed)
                if matched is not None:
                    from cpet_foundation.data.matched_inputs import checked_matched_fit
                    checked_matched_fit(modeldir,matched,measured=stage=='measured')
                if stage!='measured':
                    assert manifest['distillation_package_sha256']==package_hash
                    assert manifest['raw_cpet_opened_by_student_trainer'] is False
                    assert manifest['measured_training_targets_available'] is False
                    assert all(cfg.get(k) is None for k in ('source_h5','split_path','normalization_stats_path','teacher_checkpoint'))
                    isolated+=1
                else:
                    assert Path(manifest['source_h5']).resolve()==(folder/'dataset.h5').resolve()
                    assert (ROOT/cfg['source_h5']).resolve()==(folder/'dataset.h5').resolve()
                fitted+=1
                for domain,split in [('internal','val'),('external','test')]:
                    stem=ROOT/matrix['evaluation_dir']/f'{stage}_seed{seed}_{domain}'
                    metrics=checked_evaluation(stem,checkpoint,dataset,split,report)
                    record=next(r for r in summary['records'] if (r['stage'],r['seed'],r['domain'])==(stage,seed,domain))
                    assert all(math.isclose(record['metrics'][k],metrics[k],rel_tol=1e-10,abs_tol=1e-10) for k in METRICS)
                    runs.append({'architecture':arch,'stage':stage,'seed':seed,'domain':domain,'checkpoint_sha256':digest(checkpoint)})
                    evaluated+=1
        for key,values in summary['summary'].items():
            stage,domain=key.split(':')
            records=[r for r in summary['records'] if r['stage']==stage and r['domain']==domain]
            assert len(records)==3
            for metric in METRICS:
                expected=sum(r['metrics'][metric] for r in records)/3
                assert math.isclose(values[metric]['mean'],expected,rel_tol=1e-10,abs_tol=1e-10)
        for domain in ('internal','external'):
            assert f'ordinary_package_minus_measured:{domain}' in summary['paired']
    assert (fitted,evaluated,isolated)==(15,30,9)
    result={'status':'verified','experiment':prefix,'dataset_sha256':dataset,'package_sha256':package_hash,
            'teacher_checkpoint_sha256':digest(teacher),'student_fits':fitted,'evaluations':evaluated,
            'package_only_fits':isolated,'cohort':report,'runs':runs}
    out=ROOT/'artifacts/eval'/prefix/'FINAL_AUDIT.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','cohort')}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--config-prefix',required=True)
    args=parser.parse_args();audit(args.dataset_dir.resolve(),args.config_prefix)
