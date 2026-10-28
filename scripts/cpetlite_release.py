#!/usr/bin/env python3
"""Code-only CPETLite integrity checks, CPU examples and authorized interval replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def require_hash(path, expected):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Required authorized input is missing: ' + path.name)
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError('Input hash mismatch: ' + path.name)


def check(root):
    manifest = json.loads((root / 'RELEASE_MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        relative = Path(name)
        resolved = (root / relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not resolved.is_relative_to(root.resolve()):
            raise ValueError('Invalid release manifest path')
        require_hash(resolved, expected)
    return manifest


def synthetic(root):
    for directory in ('execution_principal', 'execution_controls'):
        env = {**os.environ, 'PYTHONPATH': str(root / directory / 'src'),
               'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''}
        subprocess.run([sys.executable, str(root / 'scripts/reproducibility_smoke.py')], cwd=root, env=env, check=True)
        subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                        'tests/test_synthetic_masks.py'], cwd=root, env=env, check=True)


def replay(root, manifest, args):
    inputs = manifest['controlled_input_hashes']
    require_hash(args.principal_results, inputs['principal_results'])
    require_hash(args.control_results, inputs['control_results'])
    for name, sha in inputs['reference_summaries'].items():
        require_hash(args.reference_dir / name, sha)
    output = args.output.resolve()
    if output.is_relative_to(root.resolve()) or output.exists():
        raise ValueError('Output must be a new directory outside the code release')
    if args.control_results.name != 'RESULTS.tar.gz':
        raise ValueError('Control archive must retain the filename RESULTS.tar.gz')
    import reporting_runtime
    reporting_runtime.require_runtime()
    import recompute_bmc_intervals as runner
    output.mkdir(parents=True)
    for name in inputs['reference_summaries']:
        shutil.copy2(args.reference_dir / name, output / name)
    runner.ROOT = root
    runner.DEST = output
    runner.principal.ARCHIVE = args.principal_results.resolve()
    runner.control.SOURCE = args.control_results.resolve().parent
    runner.main()
    return {'paired_intervals_replayed': 36, 'training_performed': False, 'output': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('check', help='Verify code hashes; optionally run synthetic CPU tests')
    command.add_argument('--synthetic', action='store_true')
    command = sub.add_parser('replay-intervals', help='Requires separately authorized inputs, never downloads data')
    for name in ('principal-results', 'control-results', 'reference-dir', 'output'):
        command.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = check(ROOT)
    result = {'version': manifest['version'], 'files_verified': len(manifest['files']),
              'clinical_data_included': False}
    if args.command == 'check':
        if args.synthetic:
            synthetic(ROOT)
        result['synthetic_tests_requested'] = args.synthetic
        result['clinical_reproduction'] = False
    else:
        result.update(replay(ROOT, manifest, args))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
