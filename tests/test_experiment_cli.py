"""Public path/seed CLI contract; no clinical inputs or training required."""
from pathlib import Path
import json
import os
import subprocess
import shutil
import sys
import pytest
import yaml

ROOT = Path(os.environ.get('CPETLITE_PORTABLE_ROOT', Path(__file__).resolve().parents[1]))


def call(*args):
    return subprocess.run([sys.executable, str(ROOT / 'scripts/run_experiment.py'), *map(str, args)],
                          capture_output=True, text=True)


def test_reader_can_prepare_selected_seed_with_paths_relative_to_yaml(tmp_path):
    paths = tmp_path / 'paths.yaml'
    paths.write_text(yaml.safe_dump({'principal': {'supervision': 'inputs/teacher.h5'},
                                    'output_root': 'outputs'}))
    result = call('prepare', '--experiment', 'principal.transformer.teacher',
                  '--seeds', '17', '--paths', paths)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['executed'] is False
    assert len(plan['runs']) == 1
    run = plan['runs'][0]
    config = yaml.safe_load(Path(run['config']).read_text())
    assert config['seed'] == 17
    assert config['source_h5'] is None
    assert config['teacher_checkpoint'] is None
    assert config['distillation_package_path'] == str(tmp_path / 'inputs/teacher.h5')
    assert Path(config['output_dir']).is_relative_to(tmp_path / 'outputs')
    assert run['command'][1:4] == ['-m', 'cpet_foundation.cli', 'train']
    assert not Path(config['output_dir']).exists()


@pytest.mark.parametrize('extra', [('--seeds', '17', '17'), ('--seeds', '99'),
                                  ('--experiment', 'unknown')])
def test_invalid_selection_is_rejected_without_creating_outputs(tmp_path, extra):
    paths = tmp_path / 'paths.yaml'
    paths.write_text(yaml.safe_dump({'output_root': 'outputs'}))
    result = call('prepare', '--experiment', 'principal.gru.teacher', '--paths', paths, *extra)
    assert result.returncode != 0
    assert not (tmp_path / 'outputs').exists()


def test_missing_supervision_blocks_explicit_training_before_files_are_created(tmp_path):
    paths = tmp_path / 'paths.yaml'
    paths.write_text(yaml.safe_dump({'controls': {'supervision': 'missing.h5'}, 'output_root': 'outputs'}))
    result = call('run', '--experiment', 'controls.gru.full_package', '--seeds', '17',
                  '--paths', paths, '--execute')
    assert result.returncode != 0
    assert 'Required input file does not exist' in result.stderr
    assert not (tmp_path / 'outputs').exists()


def test_evaluation_preparation_preserves_original_evaluation_settings(tmp_path):
    paths = tmp_path / 'paths.yaml'
    paths.write_text(yaml.safe_dump({'principal': {'source_h5': 'dataset.h5', 'teacher_checkpoint': 'teacher.pt'},
                                    'output_root': 'outputs'}))
    result = call('evaluate', '--experiment', 'principal.gru.teacher', '--seeds', '17', '--paths', paths)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    run = plan['runs'][0]
    config = yaml.safe_load(Path(run['config']).read_text())
    assert config['source_h5'] == str(tmp_path / 'dataset.h5')
    # Original principal matrices used a seed42 measured-supervision config for scoring.
    # Keep that evaluation setup independent of the selected checkpoint's training seed.
    assert config['seed'] == 42
    assert config['architecture'] == 'wearable_student'
    assert '--teacher-checkpoint' in run['command']
    assert '--split-name' in run['command']


def test_prepared_paths_are_independent_of_caller_working_directory(tmp_path):
    paths = tmp_path / 'paths.yaml'
    paths.write_text(yaml.safe_dump({'output_root': 'outputs'}))
    script = (ROOT / 'scripts/run_experiment.py').resolve()
    result = subprocess.run([sys.executable, str(script), 'prepare', '--experiment', 'teacher.train',
                             '--paths', str(paths)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    config = yaml.safe_load(Path(json.loads(result.stdout)['runs'][0]['config']).read_text())
    assert Path(config['output_dir']).is_relative_to(tmp_path / 'outputs')


def test_measured_supervision_resolves_matched_input_package_as_well_as_raw_data(tmp_path):
    paths = tmp_path / 'paths.yaml'
    paths.write_text(yaml.safe_dump({'principal': {'source_h5': 'dataset.h5', 'supervision': 'teacher.h5'},
                                    'output_root': 'outputs'}))
    result = call('prepare', '--experiment', 'principal.transformer.measured', '--seeds', '73', '--paths', paths)
    assert result.returncode == 0, result.stderr
    config = yaml.safe_load(Path(json.loads(result.stdout)['runs'][0]['config']).read_text())
    assert config['matched_input_package'] == str(tmp_path / 'teacher.h5')
    assert config['source_h5'] == str(tmp_path / 'dataset.h5')


def test_user_can_edit_local_paths_without_breaking_source_integrity(tmp_path):
    local = tmp_path / 'code'
    shutil.copytree(ROOT, local, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    paths = local / 'configs/paths.yaml'
    paths.write_text(yaml.safe_dump({'output_root': '../results', 'principal': {'source_h5': '/user/data.h5'}}))
    result = subprocess.run([sys.executable, str(local / 'scripts/cpetlite_release.py'), 'check'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_internal_and_external_evaluation_outputs_do_not_collide(tmp_path):
    paths = tmp_path / 'paths.yaml'
    paths.write_text(yaml.safe_dump({'output_root': 'outputs'}))
    plans = []
    for split in ('validation', 'test'):
        result = call('evaluate', '--experiment', 'controls.gru.full_package', '--seeds', '17',
                      '--paths', paths, '--split', split)
        assert result.returncode == 0, result.stderr
        plans.append(json.loads(result.stdout)['runs'][0])
    assert plans[0]['output_dir'] != plans[1]['output_dir']
    assert plans[0]['config'] != plans[1]['config']


def test_central_parameters_run_without_separate_template_files(tmp_path):
    local = tmp_path / 'code'
    shutil.copytree(ROOT, local, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    shutil.rmtree(local / 'configs/templates', ignore_errors=True)
    catalogue = {
        'schema_version': 2,
        'shared_parameters': {'training': {'architecture': 'wearable_student', 'seed': 42,
            'source_h5': None, 'teacher_checkpoint': None, 'learning_rate': 0.0007,
            'loss_weights': {'trajectory': 1.2, 'endpoint': 0.3},
            'distillation_package_path': '${principal.supervision}', 'output_dir': '${run.output_dir}'}},
        'parameter_sets': {'student': {'base': 'training',
            'overrides': {'learning_rate': 0.0001, 'loss_weights': {'trajectory': 4.0}}}},
        'experiments': {'principal.transformer.teacher': {'snapshot': 'execution_principal',
            'parameters': 'student', 'kind': 'train', 'path_group': 'principal', 'seeds': [17, 42, 73]}}}
    (local / 'configs/experiments.yaml').write_text(yaml.safe_dump(catalogue))
    paths = tmp_path / 'paths.yaml'
    paths.write_text(yaml.safe_dump({'principal': {'supervision': 'teacher.h5'}, 'output_root': 'outputs'}))
    result = subprocess.run([sys.executable, str(local / 'scripts/run_experiment.py'), 'prepare',
                             '--experiment', 'principal.transformer.teacher', '--seeds', '17', '--paths', str(paths)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    config = yaml.safe_load(Path(json.loads(result.stdout)['runs'][0]['config']).read_text())
    assert config['seed'] == 17
    assert config['learning_rate'] == 0.0001
    assert config['loss_weights'] == {'trajectory': 4.0}
    assert config['source_h5'] is None
    assert config['distillation_package_path'] == str(tmp_path / 'teacher.h5')
