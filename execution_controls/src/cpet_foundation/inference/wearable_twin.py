from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cpet_foundation.data.sequences import (
    CpetSequenceDataset,
    SequenceConfig,
    normalization_options_from_config,
)
from cpet_foundation.data.wearable import (
    WearableSimulationConfig,
    simulate_action_conditioned_wearable_sample,
    simulate_wearable_sample,
)
from cpet_foundation.inference.digital_twin import SAFETY_NOTICE, denormalize_sequence
from cpet_foundation.model.wearable_student import build_wearable_student_from_config
from cpet_foundation.paths import PROJECT_ROOT


ACTION_CANDIDATES = (
    {"key": "continue_ramp", "label": "Continue ramp", "label_zh": "继续递增负荷"},
    {"key": "hold_workload", "label": "Hold workload", "label_zh": "维持当前负荷"},
    {"key": "step_down", "label": "Step down", "label_zh": "降负荷观察"},
    {"key": "stop_recovery", "label": "Stop / recovery", "label_zh": "停止进入恢复"},
)


def load_wearable_checkpoint(checkpoint_path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = build_wearable_student_from_config(
        config,
        n_wearable_features=len(checkpoint.get("watch_channels", checkpoint["wearable_channels"])),
        n_protocol_features=len(checkpoint.get("protocol_context_channels", config.get("protocol_context_channels", ()))),
        n_metadata=len(checkpoint["metadata_columns"]),
        n_target_features=len(checkpoint["feature_columns"]),
        max_len=int(config.get("wearable_max_len", config.get("max_len", 160))),
    )
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def infer_wearable_prefix(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    sample: dict[str, torch.Tensor],
    prefix_fraction: float = 0.6,
) -> dict[str, Any]:
    config = checkpoint["config"]
    wearable_config = _wearable_config(config)
    wearable = simulate_wearable_sample(sample, tuple(config["feature_columns"]), wearable_config)
    wearable_x = wearable.x.unsqueeze(0).clone()
    wearable_mask = wearable.mask.unsqueeze(0).clone()
    protocol_x = wearable.protocol_x.unsqueeze(0).clone()
    protocol_mask = wearable.protocol_mask.unsqueeze(0).clone()
    steps = wearable_x.shape[1]
    prefix_steps = max(2, min(steps, int(round(steps * prefix_fraction))))
    observed_input = wearable_x.clone()
    observed_mask = wearable_mask.clone()
    observed_input[:, prefix_steps:, :] = 0.0
    observed_mask[:, prefix_steps:, :] = 0.0
    for channel in ("Time", "Load_Phase", "RPM", "Power_Load"):
        if channel in wearable.channels:
            idx = wearable.channels.index(channel)
            observed_input[:, prefix_steps:, idx] = wearable_x[:, prefix_steps:, idx]
            observed_mask[:, prefix_steps:, idx] = wearable_mask[:, prefix_steps:, idx]
    sequence_mask = ((observed_mask.sum(dim=-1) + protocol_mask.sum(dim=-1)) > 0).float()
    phase = _phase_for_steps(sample["phase"].unsqueeze(0), steps)
    sample_count = int(config.get("future_sample_count", 0))
    outputs = model(
        observed_input,
        sample.get("metadata_raw", sample["metadata"]).unsqueeze(0),
        observed_mask,
        sequence_mask,
        phase,
        protocol_x=protocol_x,
        protocol_mask=protocol_mask,
        metadata_mask=sample.get("metadata_mask").unsqueeze(0) if sample.get("metadata_mask") is not None else None,
        body_mass_kg=_body_mass_from_sample(sample, config),
        sample_count=sample_count,
    )
    mu_standardized = outputs["trajectory_mu"].squeeze(0).cpu().numpy()
    sigma_standardized = (0.5 * outputs["trajectory_logvar"].squeeze(0)).exp().cpu().numpy()
    mu = _denormalize_target_sequence(mu_standardized, sample, config, checkpoint)
    sigma = _rescale_target_sigma(sigma_standardized, sample, config, checkpoint)
    trajectory_samples = outputs.get("trajectory_samples")
    if trajectory_samples is not None:
        sample_values = [
            _denormalize_target_sequence(sample_value, sample, config, checkpoint).tolist()
            for sample_value in trajectory_samples.squeeze(0).cpu().numpy()
        ]
    else:
        sample_values = []
    uncertainty = sigma_standardized.mean(axis=1)
    future_uncertainty = float(np.mean(uncertainty[prefix_steps:])) if prefix_steps < steps else 0.0
    current_uncertainty = float(np.mean(uncertainty[:prefix_steps]))
    information_gain = max(0.0, future_uncertainty - current_uncertainty)
    return {
        "prefix_fraction": prefix_fraction,
        "prefix_steps": prefix_steps,
        "at_norm": float(outputs["at_norm"].item()),
        "vo2peak": float(outputs["vo2peak"].item()),
        "trajectory_mu": mu.tolist(),
        "trajectory_sigma": sigma.tolist(),
        "trajectory_samples": sample_values,
        "uncertainty": uncertainty.tolist(),
        "future_uncertainty": future_uncertainty,
        "information_gain": information_gain,
        "recommendation": _recommendation(future_uncertainty, information_gain),
        "safety_notice": SAFETY_NOTICE,
    }


def load_wearable_demo_sample(config: dict[str, Any], source_h5: Path, exam_index: int = 0) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    dataset = CpetSequenceDataset(
        SequenceConfig(
            source_h5=source_h5,
            feature_columns=tuple(config["feature_columns"]),
            metadata_columns=tuple(config["metadata_columns"]),
            max_len=int(config["max_len"]),
            limit_exams=max(exam_index + 1, 1),
            **normalization_options_from_config(config, PROJECT_ROOT),
        )
    )
    sample = dataset[exam_index]
    raw_sample = dataset.samples[exam_index]
    return sample, {
        "feature_columns": list(config["feature_columns"]),
        "case": {
            "exam_id": str(raw_sample.get("exam_id", f"wearable-demo-{exam_index}")),
            "source_institute": str(raw_sample.get("institute", "unavailable")),
            "split": "wearable-demo",
            "identifier_policy": "pseudonymized_exam_id",
        }
    }


def write_wearable_demo_payload(checkpoint_path: Path, output_path: Path, prefix_fraction: float = 0.6) -> dict[str, Any]:
    model, checkpoint = load_wearable_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    root = checkpoint_path.resolve().parents[3]
    source_h5 = Path(config["source_h5"])
    if not source_h5.is_absolute():
        source_h5 = (root / source_h5).resolve()
    sample, meta = load_wearable_demo_sample(config, source_h5)
    payload = wearable_payload(model, checkpoint, sample, meta, prefix_fraction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def wearable_payload(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    sample: dict[str, torch.Tensor],
    meta: dict[str, Any],
    prefix_fraction: float = 0.6,
) -> dict[str, Any]:
    config = checkpoint["config"]
    result = infer_wearable_prefix(model, checkpoint, sample, prefix_fraction)
    display_sample = _sample_with_original_scale_x(sample)
    wearable = simulate_wearable_sample(display_sample, tuple(config["feature_columns"]), _wearable_config(config))
    observed_feature_columns = list(meta.get("feature_columns", checkpoint["feature_columns"]))
    original_cpet_series = denormalize_sequence(sample["x"].detach().cpu().numpy(), sample).tolist()
    original_cpet_mask = sample.get("mask")
    if original_cpet_mask is not None:
        original_cpet_mask = original_cpet_mask.detach().cpu().tolist()
    result = {**result, "observed": original_cpet_series}
    watch_payload = {
        "channels": list(wearable.channels),
        "series": wearable.x.tolist(),
        "mask": wearable.mask.tolist(),
        "sampling_rate_hz": wearable.sampling_rate_hz,
    }
    protocol_payload = {
        "channels": list(wearable.protocol_channels),
        "series": wearable.protocol_x.tolist(),
        "mask": wearable.protocol_mask.tolist(),
    }
    payload = {
        "model": "CPET-Wearable-Student",
        "architecture": "watch multimodal adapter + protocol context -> CPET-equivalent physiology",
        "workout_scenario": config.get("workout_scenario", "power_bike_ergometer"),
        "case": meta["case"],
        "features": observed_feature_columns,
        "observed_wearable": watch_payload,
        "watch_signals": watch_payload,
        "protocol_context": protocol_payload,
        "observed_cpet": {
            "features": observed_feature_columns,
            "series": original_cpet_series,
            "mask": original_cpet_mask,
        },
        "inferred_cpet": {
            "features": list(checkpoint["feature_columns"]),
            "trajectory_mu": result["trajectory_mu"],
            "trajectory_sigma": result["trajectory_sigma"],
            "trajectory_samples": result.get("trajectory_samples", []),
        },
        "result": result,
        "metrics": checkpoint.get("metrics", {}),
        "teacher_checkpoint": config.get("teacher_checkpoint"),
        "live_inference": True,
        "safety_notice": SAFETY_NOTICE,
    }
    return payload


def action_conditioned_payload(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    sample: dict[str, torch.Tensor],
    meta: dict[str, Any],
    prefix_fraction: float = 0.6,
) -> dict[str, Any]:
    observed_feature_columns = list(meta.get("feature_columns", checkpoint["config"].get("feature_columns", checkpoint["feature_columns"])))
    original_cpet_series = denormalize_sequence(sample["x"].detach().cpu().numpy(), sample).tolist()
    original_cpet_mask = sample.get("mask")
    if original_cpet_mask is not None:
        original_cpet_mask = original_cpet_mask.detach().cpu().tolist()
    forecasts = action_conditioned_forecasts(model, checkpoint, sample, prefix_fraction=prefix_fraction)
    selected = forecasts[0]
    result = {**selected["result"], "observed": original_cpet_series}
    return {
        "model": "CPET-Wearable-Student",
        "architecture": "action-conditioned watch adapter + planned protocol -> CPET-equivalent physiology",
        "forecast_mode": "leakage_safe_action_conditioned",
        "uses_future_observed_wearable": False,
        "uses_future_actual_protocol": False,
        "uses_planned_protocol": True,
        "workout_scenario": checkpoint["config"].get("workout_scenario", "power_bike_ergometer"),
        "case": meta["case"],
        "features": observed_feature_columns,
        "observed_cpet": {
            "features": observed_feature_columns,
            "series": original_cpet_series,
            "mask": original_cpet_mask,
        },
        "selected_action": selected["action"]["key"],
        "action_forecasts": forecasts,
        "observed_wearable": selected["observed_wearable"],
        "watch_signals": selected["observed_wearable"],
        "protocol_context": selected["protocol_context"],
        "inferred_cpet": selected["inferred_cpet"],
        "result": result,
        "metrics": checkpoint.get("metrics", {}),
        "teacher_checkpoint": checkpoint["config"].get("teacher_checkpoint"),
        "live_inference": True,
        "safety_notice": SAFETY_NOTICE,
    }


@torch.no_grad()
def action_conditioned_forecast(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    sample: dict[str, torch.Tensor],
    *,
    prefix_fraction: float,
    action: dict[str, str],
) -> dict[str, Any]:
    return action_conditioned_forecasts(
        model,
        checkpoint,
        sample,
        prefix_fraction=prefix_fraction,
        actions=(action,),
    )[0]


@torch.no_grad()
def action_conditioned_forecasts(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    sample: dict[str, torch.Tensor],
    *,
    prefix_fraction: float,
    actions: tuple[dict[str, str], ...] = ACTION_CANDIDATES,
) -> list[dict[str, Any]]:
    config = checkpoint["config"]
    wearable_config = _wearable_config(config)
    steps = int(config.get("wearable_max_len", config.get("max_len", sample["x"].shape[0])))
    prefix_steps = max(2, min(steps, int(round(steps * prefix_fraction))))
    wearable_samples = []
    for action in actions:
        protocol_plan = _planned_protocol(sample, config, tuple(wearable_config.protocol_channels), prefix_steps, action["key"])
        wearable_samples.append(
            simulate_action_conditioned_wearable_sample(
                sample,
                tuple(config["feature_columns"]),
                wearable_config,
                prefix_steps=prefix_steps,
                protocol_plan=protocol_plan,
            )
        )
    wearable_x = torch.stack([wearable.x for wearable in wearable_samples]).clone()
    wearable_mask = torch.stack([wearable.mask for wearable in wearable_samples]).clone()
    protocol_x = torch.stack([wearable.protocol_x for wearable in wearable_samples]).clone()
    protocol_mask = torch.stack([wearable.protocol_mask for wearable in wearable_samples]).clone()
    sequence_mask = ((wearable_mask.sum(dim=-1) + protocol_mask.sum(dim=-1)) > 0).float()
    phase = torch.stack([_phase_from_protocol_plan(wearable, sample, steps) for wearable in wearable_samples])
    sample_count = int(config.get("action_future_sample_count", 0))
    metadata = sample.get("metadata_raw", sample["metadata"]).unsqueeze(0).expand(len(wearable_samples), -1).clone()
    outputs = model(
        wearable_x,
        metadata,
        wearable_mask,
        sequence_mask,
        phase,
        protocol_x=protocol_x,
        protocol_mask=protocol_mask,
        metadata_mask=(sample["metadata_mask"].unsqueeze(0).expand(len(wearable_samples), -1)
                       if "metadata_mask" in sample else None),
        body_mass_kg=_body_mass_from_sample(sample, config).expand(len(wearable_samples)),
        sample_count=sample_count,
    )
    mu_standardized_batch = outputs["trajectory_mu"].detach().cpu().numpy()
    sigma_standardized_batch = (0.5 * outputs["trajectory_logvar"]).exp().detach().cpu().numpy()
    trajectory_samples = outputs.get("trajectory_samples")
    trajectory_sample_batch = trajectory_samples.detach().cpu().numpy() if trajectory_samples is not None else None
    at_norm = outputs["at_norm"].detach().cpu().numpy()
    vo2peak = outputs["vo2peak"].detach().cpu().numpy()
    forecasts = []
    for index, (action, wearable) in enumerate(zip(actions, wearable_samples, strict=True)):
        mu_standardized = mu_standardized_batch[index]
        sigma_standardized = sigma_standardized_batch[index]
        mu = _denormalize_target_sequence(mu_standardized, sample, config, checkpoint)
        sigma = _rescale_target_sigma(sigma_standardized, sample, config, checkpoint)
        sample_values = (
            [
                _denormalize_target_sequence(sample_value, sample, config, checkpoint).tolist()
                for sample_value in trajectory_sample_batch[index]
            ]
            if trajectory_sample_batch is not None
            else []
        )
        uncertainty = sigma_standardized.mean(axis=1)
        future_uncertainty = float(np.mean(uncertainty[prefix_steps:])) if prefix_steps < steps else 0.0
        current_uncertainty = float(np.mean(uncertainty[:prefix_steps]))
        information_gain = max(0.0, future_uncertainty - current_uncertainty)
        result = {
            "prefix_fraction": prefix_fraction,
            "prefix_steps": prefix_steps,
            "at_norm": float(at_norm[index]),
            "vo2peak": float(vo2peak[index]),
            "trajectory_mu": mu.tolist(),
            "trajectory_sigma": sigma.tolist(),
            "trajectory_samples": sample_values,
            "uncertainty": uncertainty.tolist(),
            "future_uncertainty": future_uncertainty,
            "information_gain": information_gain,
            "recommendation": _recommendation(future_uncertainty, information_gain),
            "action_score": _action_score(action["key"], future_uncertainty, information_gain),
            "safety_notice": SAFETY_NOTICE,
        }
        forecasts.append(
            {
                "action": action,
                "result": result,
                "observed_wearable": {
                    "channels": list(wearable.channels),
                    "series": wearable.x.tolist(),
                    "mask": wearable.mask.tolist(),
                    "sampling_rate_hz": wearable.sampling_rate_hz,
                },
                "protocol_context": {
                    "channels": list(wearable.protocol_channels),
                    "series": wearable.protocol_x.tolist(),
                    "mask": wearable.protocol_mask.tolist(),
                },
                "inferred_cpet": {
                    "features": list(checkpoint["feature_columns"]),
                    "trajectory_mu": result["trajectory_mu"],
                    "trajectory_sigma": result["trajectory_sigma"],
                    "trajectory_samples": result.get("trajectory_samples", []),
                },
            }
        )
    return forecasts


def _planned_protocol(
    sample: dict[str, torch.Tensor],
    config: dict[str, Any],
    protocol_channels: tuple[str, ...],
    prefix_steps: int,
    action_key: str,
) -> dict[str, np.ndarray]:
    columns = list(config["feature_columns"])
    values = sample["x"].detach().cpu().numpy().astype(np.float32, copy=False)
    raw_values = denormalize_sequence(values, sample)
    steps = values.shape[0]
    prefix_steps = max(1, min(steps, prefix_steps))
    plan: dict[str, np.ndarray] = {}
    for channel in protocol_channels:
        if channel == "ramp_slope":
            continue
        plan[channel] = _base_protocol_series(channel, columns, values, sample)

    if "Time" in protocol_channels:
        time_raw = _raw_feature_series("Time", columns, values, raw_values)
        dt = _typical_positive_step(time_raw[:prefix_steps], fallback=1.0)
        planned_time_raw = time_raw.copy()
        for step in range(prefix_steps, steps):
            planned_time_raw[step] = planned_time_raw[prefix_steps - 1] + dt * (step - prefix_steps + 1)
        plan["Time"] = _standardize_feature_series("Time", planned_time_raw, columns, sample)

    power_raw = _raw_feature_series("Power_Load", columns, values, raw_values)
    rpm_raw = _raw_feature_series("RPM", columns, values, raw_values)
    power_step = _typical_positive_step(power_raw[:prefix_steps], fallback=10.0)
    current_power = float(power_raw[prefix_steps - 1]) if power_raw.size else 0.0
    current_rpm = float(rpm_raw[prefix_steps - 1]) if rpm_raw.size else 60.0
    planned_power = power_raw.copy()
    planned_rpm = rpm_raw.copy()
    planned_phase = plan.get("Load_Phase", np.zeros(steps, dtype=np.float32)).copy()
    for step in range(prefix_steps, steps):
        offset = step - prefix_steps + 1
        if action_key == "continue_ramp":
            planned_power[step] = max(current_power + power_step * offset, 0.0)
            planned_rpm[step] = current_rpm if np.isfinite(current_rpm) and current_rpm > 0 else 60.0
            planned_phase[step] = 1.0
        elif action_key == "hold_workload":
            planned_power[step] = max(current_power, 0.0)
            planned_rpm[step] = current_rpm if np.isfinite(current_rpm) and current_rpm > 0 else 60.0
            planned_phase[step] = 1.0
        elif action_key == "step_down":
            planned_power[step] = max(current_power - 20.0, 0.0)
            planned_rpm[step] = max(current_rpm * 0.85, 0.0) if np.isfinite(current_rpm) else 50.0
            planned_phase[step] = 1.0
        elif action_key == "stop_recovery":
            planned_power[step] = 0.0
            planned_rpm[step] = 0.0
            planned_phase[step] = 2.0
    if "Power_Load" in protocol_channels:
        plan["Power_Load"] = _standardize_feature_series("Power_Load", planned_power, columns, sample)
    if "RPM" in protocol_channels:
        plan["RPM"] = _standardize_feature_series("RPM", planned_rpm, columns, sample)
    if "Load_Phase" in protocol_channels:
        plan["Load_Phase"] = planned_phase.astype(np.float32)
    if "ramp_slope" in protocol_channels:
        power = plan.get("Power_Load", np.zeros(steps, dtype=np.float32))
        plan["ramp_slope"] = np.gradient(power).astype(np.float32)
    return plan


def _base_protocol_series(
    channel: str,
    columns: list[str],
    values: np.ndarray,
    sample: dict[str, torch.Tensor],
) -> np.ndarray:
    if channel in columns:
        return values[:, columns.index(channel)].astype(np.float32, copy=True)
    if channel == "Load_Phase" and "phase" in sample:
        return sample["phase"].detach().cpu().numpy().astype(np.float32, copy=True)
    return np.zeros(values.shape[0], dtype=np.float32)


def _raw_feature_series(
    channel: str,
    columns: list[str],
    values: np.ndarray,
    raw_values: np.ndarray,
) -> np.ndarray:
    if channel in columns:
        return raw_values[:, columns.index(channel)].astype(np.float32, copy=True)
    return np.zeros(values.shape[0], dtype=np.float32)


def _standardize_feature_series(
    channel: str,
    raw: np.ndarray,
    columns: list[str],
    sample: dict[str, torch.Tensor],
) -> np.ndarray:
    if channel not in columns or "feature_mean" not in sample or "feature_std" not in sample:
        return raw.astype(np.float32, copy=True)
    index = columns.index(channel)
    mean = float(sample["feature_mean"][index].detach().cpu())
    std = float(sample["feature_std"][index].detach().cpu())
    if abs(std) <= 1e-6:
        std = 1.0
    return ((raw - mean) / std).astype(np.float32)


def _typical_positive_step(values: np.ndarray, fallback: float) -> float:
    if len(values) < 2:
        return fallback
    diffs = np.diff(values.astype(np.float32, copy=False))
    positive = diffs[np.isfinite(diffs) & (diffs > 1e-6)]
    if positive.size:
        return float(np.median(positive))
    return fallback


def _phase_from_protocol_plan(
    wearable: Any,
    sample: dict[str, torch.Tensor],
    steps: int,
) -> torch.Tensor:
    if "Load_Phase" in wearable.protocol_channels:
        phase_index = wearable.protocol_channels.index("Load_Phase")
        return wearable.protocol_x[:, phase_index].round().clamp(0, 3).long()
    return _phase_for_steps(sample["phase"].unsqueeze(0), steps).squeeze(0)


def _action_score(action_key: str, future_uncertainty: float, information_gain: float) -> float:
    risk_penalty = {"continue_ramp": 0.05, "hold_workload": 0.03, "step_down": 0.02, "stop_recovery": 0.0}.get(action_key, 0.0)
    return float(information_gain - 0.15 * future_uncertainty - risk_penalty)


def _sample_with_original_scale_x(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    raw = denormalize_sequence(sample["x"].detach().cpu().numpy(), sample)
    return {**sample, "x": torch.tensor(raw, dtype=sample["x"].dtype)}


def _denormalize_target_sequence(
    values: np.ndarray,
    sample: dict[str, torch.Tensor],
    config: dict[str, Any],
    checkpoint: dict[str, Any],
) -> np.ndarray:
    mean, std = _target_feature_stats(sample, config, checkpoint, values.shape[-1])
    return (values * std) + mean


def _rescale_target_sigma(
    values: np.ndarray,
    sample: dict[str, torch.Tensor],
    config: dict[str, Any],
    checkpoint: dict[str, Any],
) -> np.ndarray:
    _mean, std = _target_feature_stats(sample, config, checkpoint, values.shape[-1])
    return np.abs(values * std)


def _target_feature_stats(
    sample: dict[str, torch.Tensor],
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint_config = checkpoint.get("config", {})
    fixed_mean = np.asarray(checkpoint_config.get("output_feature_mean", ()), dtype=np.float32)
    fixed_std = np.asarray(checkpoint_config.get("output_feature_std", ()), dtype=np.float32)
    if fixed_mean.shape == (n_features,) and fixed_std.shape == (n_features,):
        return fixed_mean, np.where(np.abs(fixed_std) > 1e-6, fixed_std, 1.0).astype(np.float32)
    if "feature_mean" not in sample or "feature_std" not in sample:
        return np.zeros(n_features, dtype=np.float32), np.ones(n_features, dtype=np.float32)
    source_columns = list(config.get("feature_columns", ()))
    target_columns = list(checkpoint.get("feature_columns", config.get("target_feature_columns", source_columns)))
    if len(target_columns) != n_features:
        return np.zeros(n_features, dtype=np.float32), np.ones(n_features, dtype=np.float32)
    source_mean = sample["feature_mean"].detach().cpu().numpy().astype(np.float32, copy=False)
    source_std = sample["feature_std"].detach().cpu().numpy().astype(np.float32, copy=False)
    if len(source_columns) != source_mean.shape[0] or len(source_columns) != source_std.shape[0]:
        return np.zeros(n_features, dtype=np.float32), np.ones(n_features, dtype=np.float32)
    source_index = {name: index for index, name in enumerate(source_columns)}
    if any(name not in source_index for name in target_columns):
        return np.zeros(n_features, dtype=np.float32), np.ones(n_features, dtype=np.float32)
    indices = [source_index[name] for name in target_columns]
    mean = source_mean[indices]
    std = np.where(np.abs(source_std[indices]) > 1e-6, source_std[indices], 1.0).astype(np.float32)
    return mean, std


def _body_mass_from_sample(sample: dict[str, torch.Tensor], config: dict[str, Any]) -> torch.Tensor:
    metadata_columns = list(config.get("metadata_columns", ()))
    if "Weight_kg" in metadata_columns and "metadata_raw" in sample:
        index = metadata_columns.index("Weight_kg")
        return sample["metadata_raw"][index].reshape(1).to(dtype=torch.float32)
    return torch.tensor([70.0], dtype=torch.float32)


def _wearable_config(config: dict[str, Any]) -> WearableSimulationConfig:
    watch_channels = tuple(config.get("watch_channels", config.get("wearable_channels", ())))
    if not watch_channels:
        watch_channels = ("HR", "HR_smooth", "HR_slope", "HR_acceleration", "HR_rolling_std", "HR_reserve_pct", "active_energy_rate_proxy", "wrist_motion_proxy", "signal_quality")
    return WearableSimulationConfig(
        sampling_rate_hz=float(config.get("wearable_sampling_rate_hz", 1.0)),
        max_len=int(config.get("wearable_max_len", config.get("max_len", 160))),
        workout_scenario=str(config.get("workout_scenario", "power_bike_ergometer")),
        selected_channels=watch_channels,
        protocol_channels=tuple(config.get("protocol_context_channels", ("Time", "Load_Phase", "Power_Load", "RPM", "ramp_slope"))),
        lag_steps=int(config.get("wearable_lag_steps", 0)),
        noise_std=float(config.get("wearable_noise_std", 0.0)),
        dropout_probability=float(config.get("wearable_dropout_probability", 0.0)),
        quantization_step=config.get("wearable_quantization_step"),
        jitter_std=float(config.get("wearable_jitter_std", 0.0)),
        seed=int(config.get("seed", 42)),
    )


def _phase_for_steps(phase: torch.Tensor, steps: int) -> torch.Tensor:
    if phase.shape[1] == steps:
        return phase
    positions = torch.linspace(0, phase.shape[1] - 1, steps, device=phase.device).round().long()
    return phase.index_select(1, positions)


def _recommendation(future_uncertainty: float, information_gain: float) -> str:
    if future_uncertainty < 0.75 and information_gain < 0.15:
        return "consider-stop"
    if future_uncertainty > 1.25:
        return "continue"
    return "insufficient-confidence"
