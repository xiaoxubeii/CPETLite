#!/usr/bin/env python3
"""Reproduce the archived CPETLite v4 teacher-package experiment in new paths."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from cpet_foundation.training.runner import load_yaml
from cpet_foundation.data.distillation_package import audit_distillation_package


ROOT = Path(__file__).resolve().parents[1]
METRICS = ('primary_trajectory_rmse', 'at_time_s_mae', 'vo2peak_mae')


def preflight_contract(matrix_path: Path, teacher_path: Path, export_path: Path) -> dict:
    old_teacher = load_yaml(ROOT/'configs/train_protocol_sde_cpetlite_v4.yaml')
    new_teacher = load_yaml(teacher_path)
    old_export = load_yaml(ROOT/'configs/export_teacher_package_cpetlite_v4.yaml')
    new_export = load_yaml(export_path)
    if {k:v for k,v in old_teacher.items() if k!='output_dir'} != {k:v for k,v in new_teacher.items() if k!='output_dir'}:
        raise ValueError('teacher scientific config differs from archived v4')
    if {k:v for k,v in old_export.items() if k!='teacher_checkpoint'} != {k:v for k,v in new_export.items() if k!='teacher_checkpoint'}:
        raise ValueError('teacher export scientific config differs from archived v4')
    matrix = yaml.safe_load(matrix_path.read_text())
    archived = yaml.safe_load((ROOT/'configs/cpetlite_v4_package_comparison.yaml').read_text())
    if matrix['seeds'] != archived['seeds'] or list(matrix['stages']) != ['measured','ordinary_package']:
        raise ValueError('student seed/stage contract differs from archived v4')
    for stage in matrix['stages']:
        old = load_yaml(ROOT/archived['stages'][stage]['base_config'])
        new = load_yaml(ROOT/matrix['stages'][stage]['base_config'])
        if old != new:
            raise ValueError(f'{stage} scientific config differs from archived v4')
    if matrix['secure_evaluation_config'] != archived['secure_evaluation_config']:
        raise ValueError('secure evaluator config differs from archived v4')
    if (ROOT/new_teacher['output_dir']/'model.pt').resolve() != (ROOT/new_export['teacher_checkpoint']).resolve():
        raise ValueError('new export does not reference the new teacher')
    if (ROOT/matrix['teacher_checkpoint']).resolve() != (ROOT/new_teacher['output_dir']/ 'model.pt').resolve():
        raise ValueError('new matrix does not reference the new teacher checkpoint')
    if (ROOT/matrix['package']).resolve() == (ROOT/archived['package']).resolve():
        raise ValueError('rerun would overwrite archived package')
    for stage in matrix['stages']:
        if matrix['stages'][stage]['output_template'] == archived['stages'][stage]['output_template']:
            raise ValueError(f'rerun would overwrite archived {stage} checkpoints')
    return {'seeds':matrix['seeds'], 'stages':list(matrix['stages']),
            'teacher_feature_count':len(new_teacher['feature_columns']),
            'teacher_epochs':new_teacher['epochs'],
            'split_path':new_teacher['split_path'],
            'normalization_stats_path':new_teacher['normalization_stats_path']}


def run_command(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'src')
    with log.open('w') as output:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); inspect {log}')


def write_config(path: Path, value: dict) -> Path:
    path.write_text(yaml.safe_dump(value,sort_keys=False))
    return path


def reproduce_delta(new_summary: Path, destination: Path) -> None:
    old=json.loads((ROOT/'artifacts/eval/cpetlite_v4_package_observability/summary.json').read_text())
    new=json.loads(new_summary.read_text())
    old_map={(x['stage'],x['seed'],x['domain']):x for x in old['records']}
    rows=[]
    for item in new['records']:
        key=(item['stage'],item['seed'],item['domain'])
        prior=old_map[key]
        rows.append({'stage':key[0],'seed':key[1],'domain':key[2],
                     'delta_new_minus_archived':{m:item['metrics'][m]-prior['metrics'][m] for m in METRICS}})
    destination.write_text(json.dumps({'per_seed':rows,
       'mean_new_minus_archived':{k:{m:new['summary'][k][m]['mean']-old['summary'][k][m]['mean'] for m in METRICS}
                                for k in new['summary'] if k in old['summary']},
       'interpretation':'replication delta; no historical checkpoint or package reused'},indent=2))


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    name='cpetlite_v4_legacy_retest'
    matrix_path=ROOT/f'configs/{name}.yaml'
    teacher_path=ROOT/f'configs/train_protocol_sde_{name}.yaml'
    export_path=ROOT/f'configs/export_teacher_package_{name}.yaml'
    report=preflight_contract(matrix_path,teacher_path,export_path)
    print(json.dumps({'preflight':'passed',**report}),flush=True)
    matrix=yaml.safe_load(matrix_path.read_text())
    teacher=load_yaml(teacher_path); export=load_yaml(export_path)
    run_name=name+'_smoke' if args.smoke else name
    log_dir=ROOT/'artifacts/logs'/run_name
    if args.smoke:
        teacher['output_dir']=f'artifacts/models/{run_name}_teacher_seed42'
        teacher.update(limit_exams=32,epochs=2)
        export['teacher_checkpoint']=teacher['output_dir']+'/model.pt'
        matrix['experiment_id']=run_name
        matrix['config_prefix']=run_name
        matrix['teacher_checkpoint']=export['teacher_checkpoint']
        matrix['package']=f'artifacts/dataset/{run_name}/distillation.h5'
        matrix['evaluation_dir']=f'artifacts/eval/{run_name}'
        matrix['log_directory']=f'artifacts/logs/{run_name}/students'
        for stage,spec in matrix['stages'].items():
            spec['output_template']=f'artifacts/models/{run_name}_{stage}_seed{{seed}}'
        teacher_path=write_config(ROOT/f'configs/{run_name}_teacher.yaml',teacher)
        export_path=write_config(ROOT/f'configs/{run_name}_export.yaml',export)
        matrix_path=write_config(ROOT/f'configs/{run_name}_matrix.yaml',matrix)
    checkpoint=ROOT/teacher['output_dir']/'model.pt'
    if not all((checkpoint.parent/file).is_file() for file in ('model.pt','metrics.json','manifest.json')):
        print(json.dumps({'stage':'teacher','status':'training','epochs':teacher['epochs']}),flush=True)
        run_command([sys.executable,'-u',str(ROOT/'scripts/train_config.py'),'--config',str(teacher_path)],log_dir/'teacher.log')
    else:print(json.dumps({'stage':'teacher','status':'existing'}),flush=True)
    package=ROOT/matrix['package']
    if not package.exists():
        print(json.dumps({'stage':'package','status':'exporting'}),flush=True)
        command=[sys.executable,'-u',str(ROOT/'scripts/export_teacher_only_package.py'),
                 '--config',str(export_path),'--output',str(package)]
        if args.smoke:command.extend(['--limit-exams','32'])
        run_command(command,log_dir/'export.log')
    audit=audit_distillation_package(package)
    expect_train=32 if args.smoke else 9779
    if audit['splits']['train']!=expect_train or audit['supervision_mode']!='teacher_only':
        raise RuntimeError(f'Unexpected package population/role: {audit["splits"]}')
    if not audit['metadata'].get('observability'):
        print(json.dumps({'stage':'package','status':'adding_label_free_weights'}),flush=True)
        run_command([sys.executable,str(ROOT/'scripts/add_teacher_package_observability.py'),
                     '--package',str(package)],log_dir/'observability.log')
    print(json.dumps({'package_audit':audit_distillation_package(package)['splits']}),flush=True)
    generated=[]
    for stage,spec in matrix['stages'].items():
        for seed in ([42] if args.smoke else matrix['seeds']):
            c=load_yaml(ROOT/spec['base_config'])
            c.update(seed=seed,output_dir=spec['output_template'].format(seed=seed),
                     experiment_group=matrix['experiment_id'],comparison_role=stage)
            if stage=='ordinary_package':
                c['distillation_package_path']=matrix['package']
                if any(c.get(k) not in (None,'') for k in ('source_h5','split_path','normalization_stats_path','teacher_checkpoint')):
                    raise ValueError('Package student resolves raw source or teacher checkpoint')
            if args.smoke:c.update(epochs=2,limit_exams=32)
            path=write_config(ROOT/f'configs/{matrix["config_prefix"]}_{stage}_seed{seed}.yaml',c)
            generated.append((stage,seed,path,ROOT/c['output_dir']))
    audit_command=[sys.executable,str(ROOT/'scripts/audit_cpetlite_v4_package.py'),
                   '--package',str(package)]
    for stage,seed,path,_ in generated:
        if stage=='ordinary_package':audit_command.extend(['--student-config',str(path)])
    run_command(audit_command,log_dir/'package_audit.json')
    for stage,seed,path,out in generated:
        if (out/'model.pt').exists():
            print(json.dumps({'stage':stage,'seed':seed,'status':'existing'}),flush=True)
            continue
        print(json.dumps({'stage':stage,'seed':seed,'status':'training'}),flush=True)
        run_command([sys.executable,'-u',str(ROOT/'scripts/train_config.py'),'--config',str(path)],
                    log_dir/'students'/f'{stage}_seed{seed}.log')
    print(json.dumps({'stage':'evaluation','status':'running'}),flush=True)
    command=[sys.executable,'-u',str(ROOT/'scripts/evaluate_cpetlite_v4_package_comparison.py'),
             '--matrix',str(matrix_path)]
    if args.smoke:command.extend(['--limit','32','--seed','42'])
    run_command(command,log_dir/'evaluation.log')
    result=ROOT/matrix['evaluation_dir']/('pilot_summary.json' if args.smoke else 'summary.json')
    if not result.exists():raise RuntimeError('Evaluation summary missing')
    reproduce_delta(result,ROOT/matrix['evaluation_dir']/('pilot_reproduction_delta.json' if args.smoke else 'reproduction_delta.json'))
    print(json.dumps({'status':'complete','summary':str(result)}),flush=True)


if __name__=='__main__':main()
