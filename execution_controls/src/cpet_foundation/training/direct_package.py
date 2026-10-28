"""Three-channel masked teacher and isolated direct-Transformer students.

Canonical x order: VO2, VCO2, VE, HR, Load_Phase, Power_Load, Time, SpO2.
Measured validation remains in a separate scorer process, not fitting targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np
import torch
from torch import nn
import yaml

from cpet_foundation.data.sequences import CpetSequenceDataset, SequenceConfig, ids_from_split
from cpet_foundation.model.world_model import ModelConfig, ProtocolConditionedLatentSDEWorldModel

FEATURES = ('VO2', 'VCO2', 'VE', 'HR', 'Load_Phase', 'Power_Load', 'Time', 'SpO2')
META = ('Age', 'Gender', 'Height_cm', 'Weight_kg')
MODULE = 'cpet_foundation.training.direct_package'


def restricted_input(x, availability, phase, spo2=False):
    idx = [3, 6, 5] + ([7] if spo2 else [])
    m = availability[..., idx]
    phase_valid = availability[..., 4:5]
    p = nn.functional.one_hot(phase.clamp(0, 3), 4).float() * phase_valid
    return torch.cat((torch.nan_to_num(x[..., idx]) * m, m, p), -1)


def hidden_teacher_input(x, availability, phase, channel, hidden=None):
    if hidden is None:
        hidden = torch.zeros_like(x[..., :3])
        hidden[..., channel] = 1
    visible = availability[..., :3] * (1 - hidden)
    return torch.cat((torch.nan_to_num(x[..., :3]) * visible, visible, hidden,
                      restricted_input(x, availability, phase)), -1)


def masked_mse(pred, target, mask):
    valid = (mask > 0) & torch.isfinite(target)
    delta = pred - torch.where(valid, target, pred.detach())
    weights = torch.where(valid, mask, torch.zeros_like(mask))
    return (delta.square() * weights).sum() / weights.sum().clamp_min(1e-8)


class DirectCPET(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        width = cfg['d_model']
        self.projection = nn.Linear(input_dim, width)
        self.static = nn.Linear(len(META), width)
        t = torch.arange(cfg['max_len']).float().unsqueeze(1)
        f = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000) / width))
        pos = torch.zeros(cfg['max_len'], width)
        pos[:, 0::2], pos[:, 1::2] = torch.sin(t * f), torch.cos(t * f)
        self.register_buffer('position', pos)
        layer = nn.TransformerEncoderLayer(width, cfg['n_heads'], width * 4,
                    cfg['dropout'], batch_first=True, norm_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, cfg['n_layers'], enable_nested_tensor=False)
        self.curve = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))
        self.endpoints = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                        nn.GELU(), nn.Linear(width, 2))

    def forward(self, inputs, metadata, mask):
        z = self.projection(inputs) + self.static(metadata)[:, None] + self.position[:inputs.shape[1]]
        z = self.encoder(z, src_key_padding_mask=mask <= 0)
        pooled = (z * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        e = self.endpoints(pooled)
        return self.curve(z), torch.stack((e[:, 0].sigmoid() * 1.5,
                                          nn.functional.softplus(e[:, 1]) * 10), -1)


class SDETeacher(nn.Module):
    """Existing SDE dynamics, adapted to the same mask-aware teacher boundary.

    Variance heads are neither fitting targets nor exported knowledge in this arm.
    Stochastic training; deterministic drift-only export and checkpoint evaluation.
    """
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.world = ProtocolConditionedLatentSDEWorldModel(ModelConfig(
            n_features=input_dim, n_metadata=len(META), max_len=cfg['max_len'],
            d_model=cfg['d_model'], n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
            dropout=cfg['dropout'], protocol_feature_indices=(10, 11),
            sde_dt=cfg.get('sde_dt', .05), sde_noise_scale=cfg.get('sde_noise_scale', .01)))
        self.world.traj_mu = nn.Linear(cfg['d_model'], 3)

    def forward(self, inputs, metadata, mask):
        phase = inputs[..., 15:19].argmax(-1)
        outputs = self.world(inputs, metadata, mask, phase)
        endpoints = torch.stack((outputs['at_norm'] * 1.5,
                          nn.functional.softplus(outputs['vo2peak']) * 10), -1)
        return outputs['trajectory_mu'], endpoints


class CurrentInputSDETeacher(nn.Module):
    """Original SDE structure with seven current values, not 19 learned inputs.

    The shared 19-field transport representation is unpacked before projection.
    Endpoint heads retain original sigmoid-AT/linear-peak parameterization.
    """
    full_information_export = True

    def __init__(self, input_dim, cfg):
        super().__init__()
        if input_dim != 7:
            raise ValueError('Current-input SDE teacher requires seven value channels')
        self.unmasked_training_probability = cfg.get('teacher_unmasked_probability', .25)
        self.gas_denoise_std = cfg.get('teacher_gas_denoise_std', .02)
        self.world = ProtocolConditionedLatentSDEWorldModel(ModelConfig(
            n_features=7, n_metadata=len(META), max_len=cfg['max_len'],
            d_model=cfg['d_model'], n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
            dropout=cfg['dropout'], protocol_feature_indices=(6,4,5),
            sde_dt=cfg.get('sde_dt',.05), sde_noise_scale=cfg.get('sde_noise_scale',.01)))
        self.world.traj_mu = nn.Linear(cfg['d_model'],3)
        self.world.traj_logvar = nn.Linear(cfg['d_model'],3)
        self.world.process_to_feature_var = nn.Linear(cfg['d_model'],3)

    def forward(self, inputs, metadata, mask):
        phase = inputs[...,15:19].argmax(-1)
        values = torch.cat((inputs[...,:3], inputs[...,9:10], phase[...,None].to(inputs.dtype),
                            inputs[...,11:12],inputs[...,10:11]),-1)
        outputs = self.world(values,metadata,mask,phase)
        return outputs['trajectory_mu'], torch.stack((outputs['at_norm'],outputs['vo2peak']),-1)


def build_direct_model(input_dim, cfg, role):
    if role == 'teacher' and cfg.get('teacher_architecture') == 'current_input_sde':
        return CurrentInputSDETeacher(input_dim,cfg)
    if role == 'teacher' and cfg.get('teacher_architecture') == 'sde':
        return SDETeacher(input_dim, cfg)
    return DirectCPET(input_dim, cfg)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def dump_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2))


def tensor_batch(data, indices, device):
    return {k: v[indices].to(device) for k, v in data.items() if isinstance(v, torch.Tensor)}


def hidden_prediction(model, b, random_mask=False):
    if not random_mask and getattr(model,'full_information_export',False):
        hidden = torch.zeros_like(b['x'][...,:3])
        inputs = hidden_teacher_input(b['x'],b['feature_mask'],b['phase'],0,hidden)
        return (*model(inputs,b['metadata'],b['mask']),None)
    if random_mask:
        n, length, _ = b['x'].shape
        channel = torch.randint(3, (n,), device=b['x'].device)
        hidden = torch.nn.functional.one_hot(channel, 3).float()[:, None].expand(-1, length, -1).clone()
        start = torch.randint(max(1, length * 3 // 4), (n,), device=b['x'].device)
        time = torch.arange(length, device=b['x'].device)[None]
        block = (time >= start[:, None]) & (time < start[:, None] + length // 4)
        hidden = torch.maximum(hidden, block[..., None].float())
        unmasked_probability = getattr(model,'unmasked_training_probability',0.)
        if unmasked_probability:
            use_mask = torch.rand(n,device=b['x'].device) >= unmasked_probability
            hidden = hidden * use_mask[:,None,None]
        inp = hidden_teacher_input(b['x'], b['feature_mask'], b['phase'], 0, hidden)
        denoise_std = getattr(model,'gas_denoise_std',0.)
        if denoise_std:
            inp[...,:3] = inp[...,:3] + torch.randn_like(inp[...,:3]) * denoise_std * inp[...,3:6]
        return (*model(inp, b['metadata'], b['mask']), hidden)
    curves, ends = [], []
    for channel in range(3):
        inp = hidden_teacher_input(b['x'], b['feature_mask'], b['phase'], channel)
        y, e = model(inp, b['metadata'], b['mask'])
        curves.append(y[..., channel])
        ends.append(e)
    return torch.stack(curves, -1), torch.stack(ends).mean(0), None


def objective(y, e, b, hidden=None):
    m = b['target_mask']
    if hidden is not None:
        m = m * hidden
    curve = masked_mse(y, b['target_curve'], m)
    dt = b['time_s'][:, 1:] - b['time_s'][:, :-1]
    contiguous = (dt > 0) & (dt <= 30)
    dm = m[:, 1:] * m[:, :-1] * contiguous[..., None]
    scale = dt.clamp_min(1)[..., None]
    shape = masked_mse((y[:, 1:] - y[:, :-1]) / scale,
                      (b['target_curve'][:, 1:] - b['target_curve'][:, :-1]) / scale, dm)
    at = masked_mse(e[:, 0], b['target_endpoint'][:, 0], b['endpoint_mask'][:, 0])
    peak = masked_mse(e[:, 1] / 10, b['target_endpoint'][:, 1] / 10, b['endpoint_mask'][:, 1])
    return curve + .1 * shape + .1 * at + .1 * peak


def prepare(cfg, root, limit=None):
    config_file = root / 'config.json'
    if config_file.exists() and json.loads(config_file.read_text()) != cfg:
        raise ValueError('Existing experiment config differs; use a new experiment root')
    cache = root / 'secure'
    cache.mkdir(parents=True, exist_ok=True)
    os.chmod(cache, 0o700)
    expected = json.loads(Path(cfg['split_path']).read_text())
    for split in ('train', 'validation', 'test'):
        out = cache / f'{split}.pt'
        ids = ids_from_split(Path(cfg['split_path']), split, limit)
        if not ids:
            raise ValueError(f'Empty or invalid split {split}')
        if out.exists():
            if len(load(out)['metadata']) != len(ids):
                raise ValueError('Existing cache population differs; use a new experiment root')
            continue
        ds = CpetSequenceDataset(SequenceConfig(Path(cfg['source_h5']), FEATURES, META,
                    max_len=cfg['max_len'], exam_ids=ids,
                    normalization_stats_path=Path(cfg['normalization_stats_path']),
                    require_frozen_normalization=True, require_supervision_policy=True))
        if len(ds) != len(ids):
            raise ValueError(f'Loader population changed: {split}: {len(ds)} vs {len(ids)}')
        b = {k: torch.stack([ds[i][k] for i in range(len(ds))]) for k in ds[0]}
        b['x'] = torch.nan_to_num(b['x'])
        b['time_s'] = b['x'][..., 6] * b['feature_std'][:, None, 6] + b['feature_mean'][:, None, 6]
        b['target_curve'] = b['x'][..., :3].clone()
        b['target_mask'] = b['feature_mask'][..., :3] * b['mask'][..., None] * b['trajectory_supervision_mask'][:, None, None]
        b['target_endpoint'] = torch.stack((b['at_norm'], b['vo2peak']), -1)
        b['endpoint_mask'] = torch.stack((b['at_supervision_mask'], b['vo2peak_mask']), -1)
        b['metric_endpoint_mask'] = torch.stack((b['at_metric_mask'], b['vo2peak_mask']), -1)
        b['exam_ids'] = [s['exam_id'] for s in ds.samples]
        b['centres'] = [s['institute'] for s in ds.samples]
        torch.save(b, out)
        os.chmod(out, 0o600)
        print(json.dumps({'prepared': split, 'n': len(ds), 'expected': len(expected[split])}), flush=True)
    dump_json(root / 'config.json', cfg)


@torch.no_grad()
def predict(model, data, role, spo2, cfg, device):
    ys, es = [], []
    model.eval()
    for start in range(0, len(data['metadata']), cfg['batch_size']):
        b = tensor_batch(data, slice(start, start + cfg['batch_size']), device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            if role == 'teacher':
                y, e, _ = hidden_prediction(model, b)
            else:
                y, e = model(restricted_input(b['x'], b['feature_mask'], b['phase'], spo2), b['metadata'], b['mask'])
        ys.append(y.float().cpu()); es.append(e.float().cpu())
    return torch.cat(ys), torch.cat(es)


def metrics(y, e, b):
    m = b['target_mask']
    sq = (y - b['target_curve']).square() * m
    per_channel = torch.sqrt(sq.sum((0,1)) / m.sum((0,1)).clamp_min(1))
    core = float(torch.sqrt(sq.sum() / m.sum().clamp_min(1)))
    em = b['metric_endpoint_mask']
    at_error = (e[:, 0] - b['target_endpoint'][:, 0]) * b['duration_s']
    peak_error = e[:, 1] - b['target_endpoint'][:, 1]
    result = {'core_rmse': core, 'channel_rmse': dict(zip(FEATURES[:3], per_channel.tolist())),
              'n': len(e), 'n_at': int(em[:,0].sum()), 'n_peak': int(em[:,1].sum())}
    for name, err, mask in [('at_mae_s', at_error, em[:,0]), ('peak_mae_ml_kg_min', peak_error, em[:,1])]:
        result[name] = float((err.abs() * mask).sum() / mask.sum().clamp_min(1))
    return result


def score(root, run, split, cfg, scalar=False):
    # This process owns measured held-out labels. It never sends labels to fit.
    ck = load(root / 'models' / run / 'candidate.pt' if scalar else root / 'models' / run / 'best.pt')
    b = load(root / 'secure' / f'{split}.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_direct_model(ck['input_dim'], ck['config'], ck['role']).to(device)
    model.load_state_dict(ck['state'])
    y, e = predict(model, b, ck['role'], ck['spo2'], cfg, device)
    result = metrics(y, e, b)
    if scalar:
        print(json.dumps({'core_rmse': result['core_rmse']}))
    else:
        out = root / 'eval' / run
        out.mkdir(parents=True, exist_ok=True)
        dump_json(out / f'{split}.json', result)
        torch.save({'predictions': y, 'endpoints': e, 'exam_ids': b['exam_ids'],
                    'target_mask': b['target_mask'], 'metric_endpoint_mask': b['metric_endpoint_mask'],
                    'curve_sse': ((y-b['target_curve']).square()*b['target_mask']).sum((1,2)),
                    'curve_n': b['target_mask'].sum((1,2)),
                    'at_abs_s': ((e[:,0]-b['target_endpoint'][:,0])*b['duration_s']).abs(),
                    'peak_abs': (e[:,1]-b['target_endpoint'][:,1]).abs()}, out / f'{split}_cases.pt')
        print(json.dumps({'run': run, 'split': split, **result}), flush=True)


def fit(root, role, seed, spo2, cfg):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run = 'teacher_seed42' if role == 'teacher' else f'{role}_{"spo2" if spo2 else "base"}_seed{seed}'
    out = root / 'models' / run
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        return
    training_path = root / ('package/train.pt' if role == 'package' else 'secure/train.pt')
    data = load(training_path)
    input_dim = (7 if cfg.get('teacher_architecture')=='current_input_sde' else 19) if role=='teacher' else (12 if spo2 else 10)
    model = build_direct_model(input_dim, cfg, role).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg['scheduler_patience'], factor=.5, min_lr=1e-6)
    best = float('inf'); stale = 0; best_epoch = 0; history = []
    dump_json(out / 'access_manifest.json', {'role': role, 'training_path': str(training_path),
          'measured_training_targets_accessed': role != 'package', 'validation': 'separate_scalar_evaluator',
          'input_dim': input_dim, 'parameters': sum(p.numel() for p in model.parameters()), 'seed': seed})
    for epoch in range(1, cfg['epochs'] + 1):
        model.train(); losses = []
        order = torch.randperm(len(data['metadata']))
        for start in range(0, len(order), cfg['batch_size']):
            b = tensor_batch(data, order[start:start+cfg['batch_size']], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                if role == 'teacher':
                    y, e, hidden = hidden_prediction(model, b, random_mask=True)
                    if cfg.get('teacher_architecture')=='current_input_sde':
                        hidden = hidden + .1 * (1-hidden)
                else:
                    inp = restricted_input(b['x'], b['feature_mask'], b['phase'], spo2)
                    y, e = model(inp, b['metadata'], b['mask']); hidden = None
                loss = objective(y, e, b, hidden)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss: {run} epoch {epoch}')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); losses.append(float(loss.detach()))
        checkpoint = {'state': {k:v.detach().cpu() for k,v in model.state_dict().items()},
                      'input_dim': input_dim, 'role': role, 'spo2': spo2, 'epoch': epoch, 'config': cfg}
        torch.save(checkpoint, out / 'candidate.pt')
        # CPU weights only cross process boundary; measured labels stay in scorer.
        result = subprocess.run([sys.executable, '-m', MODULE, '--config', str(root/'config.json'),
                    '--root', str(root), '--mode', 'score', '--run', run, '--scalar'],
                    capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f'Secure validation evaluator failed: {result.stderr}')
        val = json.loads(result.stdout.strip().splitlines()[-1])['core_rmse']
        if not math.isfinite(val):
            raise RuntimeError('Nonfinite validation RMSE')
        scheduler.step(val)
        improved = val < best - cfg['min_delta']
        if improved:
            best = val; best_epoch = epoch; stale = 0
            torch.save(checkpoint, out / 'best.pt')
        else:
            stale += 1
        row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'validation_core_rmse': val,
               'best_epoch': best_epoch, 'lr': optimizer.param_groups[0]['lr']}
        history.append(row); dump_json(out/'history.json', history)
        print(json.dumps({'run':run, **row}), flush=True)
        if stale >= cfg['patience']:
            break
    dump_json(out/'complete.json', {'best_epoch':best_epoch, 'best_rmse':best, 'stopped_epoch':epoch})


def export_package(root, cfg):
    ck = load(root/'models/teacher_seed42/best.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_direct_model(ck['input_dim'], ck['config'], 'teacher').to(device); model.load_state_dict(ck['state'])
    b = load(root/'secure/train.pt')
    y,e = predict(model,b,'teacher',False,cfg,device)
    # Exact allowlist: raw gas values and measured endpoints never enter package.
    p = {k:b[k] for k in ['metadata','mask','phase','time_s','duration_s','exam_ids','centres']}
    p['x'] = b['x'].clone(); p['x'][..., :3] = 0
    p['feature_mask'] = b['feature_mask'].clone(); p['feature_mask'][..., :3] = 0
    p['target_curve'] = y; p['target_endpoint'] = e
    p['target_mask'] = b['target_mask']; p['endpoint_mask'] = b['endpoint_mask']
    assert not p['x'][..., :3].any()
    assert not p['feature_mask'][..., :3].any()
    assert all(torch.isfinite(p[k]).all() for k in ('target_curve','target_endpoint'))
    package = root/'package/train.pt'; package.parent.mkdir(exist_ok=True)
    torch.save(p,package)
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    dump_json(root/'package/manifest.json', {'sha256':digest,'n':len(e),
       'keys':sorted(p),'teacher_checkpoint':str(root/'models/teacher_seed42/best.pt'),
       'teacher_architecture':cfg.get('teacher_architecture','direct_transformer'),
       'export':('full-current-input, deterministic single pass' if cfg.get('teacher_architecture')=='current_input_sde'
                 else 'whole-output-channel-hidden, endpoints mean of three passes'),
       'measured_trajectory_values':False,'measured_endpoint_values':False,
       'restricted_features':['HR','Time','Load_Phase','Power_Load','optional SpO2'], 'metadata':META})
    dump_json(root/'package/normalization.json', json.loads(Path(cfg['normalization_stats_path']).read_text()))
    print(json.dumps({'exported_package':str(package),'n':len(e)}),flush=True)


def paired_summary(root, cfg):
    rng = np.random.default_rng(20260929)
    results = {}
    pairs = [('package_base','measured_base'), ('package_spo2','measured_spo2'),
             ('package_spo2','package_base'), ('measured_spo2','measured_base')]
    if cfg.get('teacher_architecture') in {'sde','current_input_sde'} and cfg.get('reuse_measured_reference_root'):
        pairs += [('package_base','package_base'), ('package_spo2','package_spo2')]
    for split in ('validation','test'):
        for a, b in pairs:
            left = [load(root/f'eval/{a}_seed{s}/{split}_cases.pt') for s in cfg['seeds']]
            right_root = Path(cfg['reuse_measured_reference_root']) if a == b else root
            right = [load(right_root/f'eval/{b}_seed{s}/{split}_cases.pt') for s in cfg['seeds']]
            ids = left[0]['exam_ids']
            assert all(c['exam_ids'] == ids for c in left+right)
            for x,z in zip(left,right):
                assert torch.equal(x['metric_endpoint_mask'],z['metric_endpoint_mask'])
                assert torch.equal(x['curve_n'],z['curve_n'])
            def outcome(cells, indices):
                values = []
                for c in cells:
                    rmse = (c['curve_sse'][indices].sum() / c['curve_n'][indices].sum().clamp_min(1)).sqrt()
                    mask = c['metric_endpoint_mask'][indices]
                    at = (c['at_abs_s'][indices]*mask[:,0]).sum()/mask[:,0].sum().clamp_min(1)
                    peak = (c['peak_abs'][indices]*mask[:,1]).sum()/mask[:,1].sum().clamp_min(1)
                    values.append(torch.stack((rmse,at,peak)))
                return torch.stack(values).mean(0).numpy()
            point = outcome(left, slice(None))-outcome(right, slice(None))
            draws = []
            for _ in range(2000):
                indices = torch.from_numpy(rng.integers(0,len(ids),len(ids)))
                draws.append(outcome(left,indices)-outcome(right,indices))
            low,high=np.quantile(np.stack(draws),[.025,.975],axis=0)
            arch = cfg.get('teacher_architecture','sde')
            label = f'{arch}_teacher_minus_direct_teacher_{a}_{split}' if a == b else f'{a}_minus_{b}_{split}'
            results[label]={name:{'difference':float(point[i]),
                   'ci95':[float(low[i]),float(high[i])]} for i,name in enumerate(
                      ('core_rmse','at_mae_s','peak_mae_ml_kg_min'))}
    dump_json(root/'paired_differences.json',{'bootstrap_replicates':2000,
            'unit':'examination, paired across fixed fitted seeds',
            'limitation':'conditional on fitted models; repeated-subject clustering not included',
            'comparisons':results})


def pipeline(root, cfg, limit=None):
    reference = cfg.get('reuse_measured_reference_root')
    if reference:
        src = Path(reference)
        root.mkdir(parents=True, exist_ok=True)
        (root/'secure').mkdir(exist_ok=True)
        # Locked caches are immutable; no source-data preprocessing changes.
        source_cfg = json.loads((src/'config.json').read_text())
        for key in ['source_h5','split_path','normalization_stats_path','max_len','d_model',
                    'n_heads','n_layers','dropout','batch_size','learning_rate','epochs',
                    'patience','min_delta','scheduler_patience','seeds']:
            if source_cfg[key] != cfg[key]:
                raise ValueError(f'Reference mismatch: {key}')
        for split in ('train','validation','test'):
            dst = root/f'secure/{split}.pt'
            if not dst.exists():
                os.link(src/f'secure/{split}.pt', dst)
        for seed in cfg['seeds']:
            for arm in ('base','spo2'):
                name=f'measured_{arm}_seed{seed}'
                if not (src/f'models/{name}/complete.json').exists():
                    raise ValueError(f'Incomplete measured reference: {name}')
                for folder in ('models','eval'):
                    dst=root/f'{folder}/{name}'
                    if not dst.exists():
                        shutil.copytree(src/f'{folder}/{name}',dst)
        dump_json(root/'reference_reuse.json', {'root':str(src), 'measured_models_retrained':False,
                'comparison':'matched student fitting; teacher implementation changed'})
    prepare(cfg,root,limit)
    config = str(root/'config.json')
    def call(*args):
        subprocess.run([sys.executable,'-m',MODULE,'--root',str(root),'--config',config,*args],check=True)
    call('--mode','fit','--role','teacher','--seed','42')
    export_package(root,cfg)
    for seed in cfg['seeds']:
        for spo2 in (False,True):
            for role in ('measured','package'):
                args = ['--mode','fit','--role',role,'--seed',str(seed)] + (['--spo2'] if spo2 else [])
                call(*args)
                run=f'{role}_{"spo2" if spo2 else "base"}_seed{seed}'
                for split in ('validation','test'):
                    if role == 'measured' and reference and (root/f'eval/{run}/{split}.json').exists():
                        continue
                    call('--mode','score','--run',run,'--split',split)
    summary={}
    for role in ('measured','package'):
        for spo2 in ('base','spo2'):
            for split in ('validation','test'):
                cells=[json.loads((root/f'eval/{role}_{spo2}_seed{s}/{split}.json').read_text()) for s in cfg['seeds']]
                summary[f'{role}_{spo2}_{split}']={k:{'mean':float(np.mean([c[k] for c in cells])),
                     'sd':float(np.std([c[k] for c in cells],ddof=1)) if len(cells)>1 else 0.}
                     for k in ('core_rmse','at_mae_s','peak_mae_ml_kg_min')}
    dump_json(root/'summary.json',summary)
    paired_summary(root,cfg)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--mode',choices=['pipeline','prepare','fit','score','export'],default='pipeline')
    parser.add_argument('--role',choices=['teacher','measured','package'],default='teacher')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--spo2',action='store_true')
    parser.add_argument('--scalar',action='store_true')
    parser.add_argument('--run'); parser.add_argument('--split',default='validation')
    parser.add_argument('--limit',type=int)
    a=parser.parse_args(); cfg=yaml.safe_load(a.config.read_text())
    torch.set_num_threads(4)
    root=a.root.resolve(); root.mkdir(parents=True,exist_ok=True)
    if a.mode=='pipeline': pipeline(root,cfg,a.limit)
    elif a.mode=='prepare': prepare(cfg,root,a.limit)
    elif a.mode=='fit': fit(root,a.role,a.seed,a.spo2,cfg)
    elif a.mode=='score': score(root,a.run,a.split,cfg,a.scalar)
    else: export_package(root,cfg)


if __name__=='__main__':
    main()
