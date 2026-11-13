#!/usr/bin/env python3
"""Resolve a portable CPETLite experiment; execution is always explicit."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = ('source_h5', 'manifest_path', 'split_path', 'normalization_stats_path',
             'teacher_checkpoint', 'distillation_package_path', 'matched_input_package')


def load(path):
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f'Expected a YAML mapping: {path}')
    return value


def local_path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Configured paths must be non-empty strings or null.')
    path = Path(value).expanduser()
    return str((base / path).resolve())


def parameters(document, name):
    """Expand one named set; each explicit override replaces a complete value."""
    spec = document.get('parameter_sets', {}).get(name)
    if not isinstance(spec, dict):
        raise ValueError(f'Unknown parameter set: {name}')
    shared = document.get('shared_parameters', {}).get(spec.get('base'))
    overrides = spec.get('overrides', {})
    if not isinstance(shared, dict) or not isinstance(overrides, dict):
        raise ValueError(f'Invalid shared parameters or overrides for: {name}')
    return {**shared, **overrides}


def resolve(value, paths, base, output):
    if isinstance(value, dict):
        return {k: resolve(v, paths, base, output) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, paths, base, output) for v in value]
    if not isinstance(value, str) or not value.startswith('${'):
        return value
    if value == '${run.output_dir}':
        return str(output)
    if not re.fullmatch(r'\$\{[a-z_]+\.[a-z_][a-z_0-9]*\}', value):
        raise ValueError(f'Invalid path token: {value}')
    group, key = value[2:-1].split('.')
    group_paths = paths.get(group, {})
    if not isinstance(group_paths, dict):
        raise ValueError(f'Expected paths group: {group}')
    configured = group_paths.get(key)
    if configured is None:
        return None
    return local_path(configured, base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'prepare', 'run', 'evaluate'])
    parser.add_argument('--experiment')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--paths', type=Path, default=ROOT / 'configs/paths.yaml')
    parser.add_argument('--checkpoint', type=Path, help='Student checkpoint for one evaluation seed.')
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    parser.add_argument('--execute', action='store_true', help='Actually invoke train/export/evaluate; otherwise only prepare.')
    args = parser.parse_args()
    document = load(ROOT / 'configs/experiments.yaml')
    catalogue = document['experiments']
    if args.action == 'list':
        if args.execute:
            parser.error('--execute is not valid with list')
        print(json.dumps(catalogue, indent=2))
        return
    if args.experiment not in catalogue:
        parser.error('Unknown experiment. Use list to see supported identifiers.')
    spec = catalogue[args.experiment]
    seeds = args.seeds if args.seeds is not None else spec['seeds']
    if len(set(seeds)) != len(seeds) or not seeds or any(s not in spec['seeds'] for s in seeds):
        parser.error(f'Choose distinct supported seeds: {spec["seeds"]}')
    if args.action == 'prepare' and args.execute:
        parser.error('prepare never executes; use run --execute instead')
    if args.action == 'evaluate' and (spec['kind'] != 'train' or not spec.get('evaluation_parameters')):
        parser.error('Evaluation requires a student experiment.')
    if args.checkpoint and (args.action != 'evaluate' or len(seeds) != 1):
        parser.error('--checkpoint requires evaluate with exactly one seed')
    paths_file = args.paths.expanduser().resolve()
    paths = load(paths_file)
    base = paths_file.parent
    if paths.get('output_root') is None:
        raise ValueError('Set output_root in the paths YAML.')
    output_root = Path(local_path(paths['output_root'], base))
    snapshot = ROOT / spec['snapshot']
    plans = []
    for seed in seeds:
        name = f'{args.experiment}_seed{seed}'
        is_eval = args.action == 'evaluate'
        output = output_root / ('evaluation' if is_eval else 'models') / name
        if is_eval:
            output = output / args.split
        suffix = f'_evaluation_{args.split}' if is_eval else ''
        config_path = output_root / 'configs' / (name + suffix + '.yaml')
        parameter_name = spec['evaluation_parameters'] if is_eval else spec['parameters']
        raw_parameters = parameters(document, parameter_name)
        config = resolve(raw_parameters, paths, base, output)
        if not is_eval:
            config['seed'] = seed
            config['experiment_group'] = args.experiment
            config['comparison_role'] = spec.get('role', args.experiment)
        if is_eval:
            checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else output_root / 'models' / name / 'model.pt'
            command = [sys.executable, str(snapshot / 'scripts/evaluate_wearable_checkpoint.py'),
                       '--checkpoint', str(checkpoint), '--config', str(config_path),
                       '--split-name', args.split, '--output', str(output / 'metrics.json'),
                       '--predictions-output', str(output / 'predictions.npz'),
                       '--manifest-output', str(output / 'manifest.json')]
            teacher_group = paths.get(spec['path_group'], {})
            teacher = teacher_group.get('teacher_checkpoint')
            if teacher:
                command += ['--teacher-checkpoint', local_path(teacher, base)]
        elif spec['kind'] == 'export':
            command = [sys.executable, str(snapshot / 'scripts/export_teacher_only_package.py'),
                       '--config', str(config_path), '--output', str(output / 'distillation.h5')]
        else:
            command = [sys.executable, '-m', 'cpet_foundation.cli', 'train', '--config', str(config_path)]
        if output.exists():
            raise ValueError(f'Refusing an existing output directory: {output}. Select a new output_root.')
        missing = [k for k in PATH_KEYS if k in config and config[k] is None
                   and '${' in str(raw_parameters.get(k, ''))]
        if args.execute:
            if missing:
                raise ValueError('Configure required input paths before execution: ' + ', '.join(missing))
            required = [config[k] for k in PATH_KEYS if config.get(k)]
            if is_eval:
                required.append(str(checkpoint))
                group = paths.get(spec['path_group'], {})
                if not group.get('teacher_checkpoint'):
                    raise ValueError('Configure teacher_checkpoint for secure evaluation.')
                required.append(local_path(group['teacher_checkpoint'], base))
            for path in required:
                if not Path(path).is_file():
                    raise ValueError(f'Required input file does not exist: {path}')
        rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        if config_path.exists() and config_path.read_text() != rendered:
            raise ValueError(f'Refusing to overwrite different resolved configuration: {config_path}')
        plans.append({'experiment': args.experiment, 'seed': seed, 'config': str(config_path),
                      'output_dir': str(output), 'command': command, 'cwd': str(snapshot),
                      'pythonpath': str(snapshot / 'src'), 'missing_path_settings': missing,
                      '_rendered': rendered})
    for plan in plans:
        path = Path(plan['config'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(plan.pop('_rendered'))
        if args.execute:
            output = Path(plan['output_dir'])
            if args.action == 'evaluate':
                output.mkdir(parents=True, exist_ok=False)
            log = path.with_suffix('.log')
            with log.open('x') as stream:
                subprocess.run(plan['command'], cwd=plan['cwd'], check=True, stdout=stream,
                               stderr=subprocess.STDOUT,
                               env={**os.environ, 'PYTHONPATH': plan['pythonpath'], 'PYTHONDONTWRITEBYTECODE': '1'})
            plan['log'] = str(log)
    print(json.dumps({'executed': args.execute, 'runs': plans}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, yaml.YAMLError, subprocess.CalledProcessError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(2)
