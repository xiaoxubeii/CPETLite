#!/usr/bin/env python3
"""Resume-safe, isolated full paper rerun on the verified mask-aware v5 cohort."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import torch
from torch.utils.data import DataLoader
import yaml

from cpet_foundation.data.sequences import CpetSequenceDataset, SequenceConfig
from cpet_foundation.data.distillation_package import audit_distillation_package
from cpet_foundation.inference.digital_twin import load_checkpoint
from cpet_foundation.training.runner import load_yaml, evaluate_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "cpet_former/artifacts/dataset/derivatives/cpetlite_paper_standard_v5_20261008"


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def config_file(name, config):
    path=ROOT/'configs'/f'{name}.yaml'
    text=yaml.safe_dump(config,sort_keys=False)
    if path.exists() and path.read_text()!=text:
        raise ValueError(f'Existing generated config differs: {path}')
    path.write_text(text)
    return path


def verified_cohort(folder):
    report=json.loads((folder/'COHORT_VERIFICATION.json').read_text())
    manifest=json.loads((folder/'DERIVATIVE_MANIFEST.json').read_text())
    if report['status']!='verified' or digest(folder/'dataset.h5')!=manifest['artifacts']['dataset.h5']:
        raise ValueError('Dataset verification or checksum failed')
    with h5py.File(folder/'dataset.h5') as h:
        counts={sp:len(h[f'splits/{sp}/metadata_table/Examination_ID']) for sp in ('train','val','test')}
    if (any(report['splits'][sp]['examinations']!=n for sp,n in counts.items())
            or report['total_examinations']!=sum(counts.values())):
        raise ValueError('Verified cohort count differs from actual HDF5')
    return report


def verify_existing_checkpoint(checkpoint):
    manifest=checkpoint.parent/'manifest.json'
    if not checkpoint.is_file() or not manifest.is_file():
        raise ValueError('Completed checkpoint requires a fitting manifest')
    record=json.loads(manifest.read_text())
    if Path(record['checkpoint']).resolve()!=checkpoint.resolve():
        raise ValueError('Checkpoint manifest identifies another run')
    return record


def setup(smoke, config_prefix='cpetlite_paper_v5'):
    prefix=config_prefix+'_smoke' if smoke else config_prefix
    teacher=load_yaml(ROOT/f'configs/train_{config_prefix}_teacher.yaml')
    export=load_yaml(ROOT/f'configs/export_{config_prefix}.yaml')
    if smoke:
        teacher.update(limit_exams=32,epochs=2,output_dir=f'artifacts/models/{prefix}_teacher_seed42')
        export.update(limit_exams=32,teacher_checkpoint=teacher['output_dir']+'/model.pt')
    teacher_path=config_file(prefix+'_teacher_run',teacher)
    export_path=config_file(prefix+'_export_run',export)
    ordinary=yaml.safe_load((ROOT/f'configs/{config_prefix}_direct.yaml').read_text())
    package=f'artifacts/dataset/{prefix}_teacher_package/distillation.h5' if smoke else ordinary['package']
    matrices=[]
    for arch in ('direct','gru'):
        matrix=yaml.safe_load((ROOT/f'configs/{config_prefix}_{arch}.yaml').read_text())
        if smoke:
            matrix.update(experiment_id=prefix+'_'+arch,config_prefix=prefix+'_'+arch,
                package=package,teacher_checkpoint=teacher['output_dir']+'/model.pt',
                log_directory=f'artifacts/logs/{prefix}_{arch}',evaluation_dir=f'artifacts/eval/{prefix}_{arch}')
            for stage,spec in matrix['stages'].items():
                cfg=load_yaml(ROOT/spec['base_config'])
                cfg.update(epochs=2,limit_exams=32)
                if stage!='measured': cfg['distillation_package_path']=package
                elif matrix.get('matched_inputs'): cfg['matched_input_package']=package
                spec['base_config']=str(config_file(f'{prefix}_{arch}_{stage}_base',cfg).relative_to(ROOT))
                spec['output_template']=f'artifacts/models/{prefix}_{arch}_{stage}_seed{{seed}}'
            matrix['secure_evaluation_config']=matrix['stages']['measured']['base_config']
        matrices.append(config_file(prefix+'_'+arch+'_matrix',matrix))
    return prefix,teacher_path,export_path,ROOT/package,matrices


def teacher_metrics(checkpoint,output,limit):
    model, ckpt=load_checkpoint(checkpoint)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device).eval();cfg=ckpt['config'];reports={}
    for split in ('val','test'):
        ds=CpetSequenceDataset(SequenceConfig(source_h5=DATA/'dataset.h5',input_format='v5_derivative_split',split_name=split,
            feature_columns=tuple(cfg['feature_columns']),metadata_columns=tuple(cfg['metadata_columns']),max_len=cfg['max_len'],limit_exams=limit))
        reports[split]={'n':len(ds),'metrics':evaluate_model(model,DataLoader(ds,batch_size=64),
            {n:i for i,n in enumerate(cfg['feature_columns'])},device,input_mode='full')}
    output.write_text(json.dumps(reports,indent=2)+'\n')


def main():
    global DATA
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--dataset-dir',type=Path,default=DATA)
    parser.add_argument('--config-prefix',default='cpetlite_paper_v5')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--prepare-only',action='store_true',help='Materialize and report the resolved experiment plan without fitting')
    args=parser.parse_args();DATA=args.dataset_dir.resolve()
    if args.workers < 1: parser.error('--workers must be positive')
    report=verified_cohort(DATA)
    prefix,teacher_cfg,export_cfg,package,matrices=setup(args.smoke,args.config_prefix)
    cfg=load_yaml(teacher_cfg)
    if (ROOT/cfg['source_h5']).resolve()!=(DATA/'dataset.h5').resolve():
        raise ValueError('Teacher config differs from requested derivative')
    matching_modes=[bool(yaml.safe_load(path.read_text()).get('matched_inputs')) for path in matrices]
    if any(matching_modes) and not all(matching_modes):
        raise ValueError('Paper matrices disagree on matched input mode')
    for matrix_path in matrices:
        matrix=yaml.safe_load(matrix_path.read_text())
        if not args.smoke and matrix['evaluation_counts']!={'internal':report['splits']['val']['examinations'],'external':report['splits']['test']['examinations']}:
            raise ValueError('Matrix evaluation count differs from current cohort')
    if args.config_prefix=='cpetlite_paper_v53' and report.get('reserved_cohort_exclusion_enabled') is not False:
        raise ValueError('v53 requires human-comparison reservation disabled')
    if args.prepare_only:
        plans=[yaml.safe_load(path.read_text()) for path in matrices]
        fits=sum(len(plan['stages'])*(1 if args.smoke else len(plan['seeds'])) for plan in plans)
        print(json.dumps({'status':'prepared','config_prefix':prefix,'teacher_config':str(teacher_cfg),
            'export_config':str(export_cfg),'package':str(package),'matrices':[str(p) for p in matrices],
            'student_fits':fits,'evaluations':2*fits,'matched_inputs':all(p.get('matched_inputs') for p in plans)},indent=2))
        return
    logs=ROOT/'artifacts/logs'/prefix;logs.mkdir(parents=True,exist_ok=True)
    reportdir=ROOT/'artifacts/eval'/prefix;reportdir.mkdir(parents=True,exist_ok=True)
    state={'experiment':prefix,'smoke':args.smoke,'dataset_sha256':digest(DATA/'dataset.h5'),'completed':[]}
    def status(stage):
        state.update(stage=stage,updated_at=datetime.now(timezone.utc).isoformat())
        (logs/'status.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({'stage':stage,'completed':state['completed']}),flush=True)
    def run(stage,arguments):
        status(stage)
        env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'src');env['PYTHONUNBUFFERED']='1'
        env.setdefault('OMP_NUM_THREADS','4');env.setdefault('MKL_NUM_THREADS','4')
        with (logs/f'{stage}.log').open('w') as out:
            p=subprocess.run([sys.executable,'-u',*arguments],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
        if p.returncode:
            status(stage+'_FAILED');raise RuntimeError(f'{stage} failed; see {logs/stage}.log')
        state['completed'].append(stage)
    teacher=ROOT/load_yaml(teacher_cfg)['output_dir']/'model.pt'
    if not teacher.exists(): run('teacher',['scripts/train_config.py','--config',str(teacher_cfg)])
    else:
        verify_existing_checkpoint(teacher)
        state['completed'].append('teacher_existing')
    status('teacher_evaluation');teacher_metrics(teacher,reportdir/'teacher_metrics.json',6 if args.smoke else None)
    state['completed'].append('teacher_evaluation')
    if not package.exists():
        run('export',['scripts/export_teacher_only_package.py','--config',str(export_cfg),'--output',str(package)])
    audit=audit_distillation_package(package)
    matched_mode=all(yaml.safe_load(path.read_text()).get('matched_inputs') for path in matrices)
    if matched_mode:
        from cpet_foundation.data.matched_inputs import checked_matched_package
        checked_matched_package(package,config=load_yaml(export_cfg),source=DATA/'dataset.h5',teacher=teacher)
    with h5py.File(package) as h:
        weighted='teacher_observability_weight' in h['splits/train/samples']
        for sp in ('train','validation'):
            g=h[f'splits/{sp}/samples'];assert 'phase_mask' in g
    if not weighted: run('weights',['scripts/add_teacher_package_observability.py','--package',str(package)])
    package_hash=digest(package);state['package_sha256']=package_hash
    package_configs=[str(ROOT/yaml.safe_load(m.read_text())['stages'][s]['base_config'])
        for m in matrices for s in yaml.safe_load(m.read_text())['stages'] if s!='measured']
    audit_args=['scripts/audit_cpetlite_v4_package.py','--package',str(package)]
    for c in package_configs: audit_args+=['--student-config',c]
    run('package_audit',audit_args)
    for arch,matrix in zip(('direct','gru'),matrices):
        cmd=['scripts/run_cpetlite_v4_package_comparison.py','--matrix',str(matrix),'--train','--workers',str(args.workers)]
        if args.smoke: cmd+=['--seed','42']
        run(arch+'_fits',cmd)
        cmd=['scripts/evaluate_cpetlite_v4_package_comparison.py','--matrix',str(matrix),'--reuse-existing']
        if args.smoke:cmd+=['--seed','42','--limit','6']
        run(arch+'_evaluation',cmd)
    assert digest(package)==package_hash,'Frozen teacher package changed'
    summaries={};manifests=[]
    for arch,matrix_path in zip(('direct','gru'),matrices):
        matrix=yaml.safe_load(matrix_path.read_text());evaluation=ROOT/matrix['evaluation_dir']
        summaries[arch]=json.loads((evaluation/('pilot_summary.json' if args.smoke else 'summary.json')).read_text())
        for stage,spec in matrix['stages'].items():
            for seed in ([42] if args.smoke else matrix['seeds']):
                p=ROOT/spec['output_template'].format(seed=seed)
                manifest=json.loads((p/'manifest.json').read_text())
                verify_existing_checkpoint(p/'model.pt')
                if matched_mode:
                    from cpet_foundation.data.matched_inputs import checked_matched_fit
                    checked_matched_fit(p,checked_matched_package(package),measured=stage=='measured')
                if stage!='measured':
                    assert manifest['distillation_package_sha256']==package_hash
                    assert manifest['raw_cpet_opened_by_student_trainer'] is False
                    assert manifest['measured_training_targets_available'] is False
                manifests.append({'architecture':arch,'stage':stage,'seed':seed,'manifest':manifest})
    result={'cohort':report,'package_sha256':package_hash,'teacher_metrics':json.loads((reportdir/'teacher_metrics.json').read_text()),
        'summaries':summaries,'runs':manifests,'smoke':args.smoke}
    (reportdir/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    text=[f'# {prefix} rerun','',f"Cohort: {report['splits']['train']['examinations']} train, {report['splits']['val']['examinations']} validation, {report['splits']['test']['examinations']} Liuyuan test. Package SHA256: {package_hash}",'',
        '| Architecture | Arm | Domain | Core RMSE | AT MAE (s) | Peak MAE (mL/kg/min) |','|---|---|---|---:|---:|---:|']
    for arch,summary in summaries.items():
        for key,metrics in summary['summary'].items():
            stage,domain=key.split(':');values=[metrics[k]['mean'] for k in ('primary_trajectory_rmse','at_time_s_mae','vo2peak_mae')]
            text.append(f'| {arch} | {stage} | {domain} | {values[0]:.4f} | {values[1]:.2f} | {values[2]:.3f} |')
    text+=['','One frozen teacher, three paired student seeds; examination-bootstrap intervals are conditional on fitted models.',
        'Missing patient IDs remain singletons; this does not independently exclude unidentified repeated patients.',
        'No external label enters fitting or checkpoint selection. New results do not establish wearable validity or a formal privacy guarantee.']
    (reportdir/'REPORT.md').write_text('\n'.join(text)+'\n')
    if not args.smoke:
        run('final_audit',['scripts/audit_cpetlite_paper_rerun.py','--dataset-dir',str(DATA),'--config-prefix',args.config_prefix])
    status('COMPLETE')


if __name__=='__main__': main()
