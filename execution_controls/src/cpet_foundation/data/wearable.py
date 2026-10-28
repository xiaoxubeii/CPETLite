from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np
import torch


DEFAULT_WATCH_CHANNELS = (
    "HR",
    "HR_smooth",
    "HR_slope",
    "HR_acceleration",
    "HR_rolling_std",
    "HR_reserve_pct",
    "active_energy_rate_proxy",
    "wrist_motion_proxy",
    "signal_quality",
)
DEFAULT_PROTOCOL_CONTEXT_CHANNELS = ("Time", "Load_Phase", "Power_Load", "RPM", "ramp_slope")
DEFAULT_WEARABLE_CHANNELS = DEFAULT_WATCH_CHANNELS
DEFAULT_LAG_CHANNELS = ("HR", "HR_smooth", "HR_slope", "HR_acceleration", "HR_rolling_std")


@dataclass(frozen=True)
class WearableSimulationConfig:
    sampling_rate_hz: float = 1.0
    max_len: int | None = None
    workout_scenario: str = "power_bike_ergometer"
    selected_channels: tuple[str, ...] = DEFAULT_WEARABLE_CHANNELS
    protocol_channels: tuple[str, ...] = DEFAULT_PROTOCOL_CONTEXT_CHANNELS
    lag_steps: int = 0
    lag_channels: tuple[str, ...] = DEFAULT_LAG_CHANNELS
    noise_std: float = 0.0
    dropout_probability: float = 0.0
    quantization_step: float | None = None
    jitter_std: float = 0.0
    seed: int = 17

    def __post_init__(self) -> None:
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if self.max_len is not None and self.max_len <= 0:
            raise ValueError("max_len must be positive when provided")
        if self.lag_steps < 0:
            raise ValueError("lag_steps must be non-negative")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("dropout_probability must be between 0 and 1")
        if self.quantization_step is not None and self.quantization_step <= 0:
            raise ValueError("quantization_step must be positive when provided")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if self.jitter_std < 0:
            raise ValueError("jitter_std must be non-negative")


@dataclass(frozen=True)
class WearableSample:
    x: torch.Tensor
    mask: torch.Tensor
    channels: tuple[str, ...]
    sampling_rate_hz: float
    protocol_x: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 0), dtype=torch.float32))
    protocol_mask: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 0), dtype=torch.float32))
    protocol_channels: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)


def simulate_wearable_sample(
    sample: Mapping[str, Any],
    feature_columns: tuple[str, ...] | list[str],
    config: WearableSimulationConfig,
    exam_id: str | None = None,
    institute: str | None = None,
) -> WearableSample:
    values = _as_numpy(sample["x"]).astype(np.float32, copy=True)
    source_mask = _expand_mask(_as_numpy(sample.get("mask")), values.shape)
    time_mask = _time_support(sample, source_mask)
    source_mask = source_mask * time_mask[:, None]
    columns = tuple(feature_columns)
    indices = _downsample_indices(values.shape[0], config.max_len)

    rng = np.random.default_rng(config.seed)
    channel_values: list[np.ndarray] = []
    channel_masks: list[np.ndarray] = []
    channel_sources: dict[str, str | None] = {}

    for channel in config.selected_channels:
        series, series_mask, source = _select_watch_channel(channel, values, source_mask, columns, sample)
        series = series[indices].astype(np.float32, copy=True)
        series_mask = series_mask[indices].astype(np.float32, copy=True)
        if config.jitter_std > 0.0 and series.size:
            jitter = rng.normal(0.0, config.jitter_std, size=series.shape).astype(np.float32)
            series = series + jitter
        if config.lag_steps and channel in config.lag_channels:
            series, series_mask = _lag(series, series_mask, config.lag_steps)
        series_mask *= time_mask[indices]
        channel_values.append(series)
        channel_masks.append(series_mask)
        channel_sources[channel] = source

    wearable_x = np.stack(channel_values, axis=1).astype(np.float32)
    wearable_mask = np.stack(channel_masks, axis=1).astype(np.float32)
    wearable_x, wearable_mask = _apply_corruption(wearable_x, wearable_mask, config, rng)

    protocol_values: list[np.ndarray] = []
    protocol_masks: list[np.ndarray] = []
    for channel in config.protocol_channels:
        series, series_mask, source = _select_protocol_channel(channel, values, source_mask, columns, sample)
        series_mask = series_mask * time_mask
        series = np.where(series_mask > 0, series, 0.0)
        protocol_values.append(series[indices].astype(np.float32, copy=True))
        protocol_masks.append(series_mask[indices].astype(np.float32, copy=True))
        channel_sources[channel] = source
    protocol_x = np.stack(protocol_values, axis=1).astype(np.float32) if protocol_values else np.zeros((len(indices), 0), dtype=np.float32)
    protocol_mask = (
        np.stack(protocol_masks, axis=1).astype(np.float32) if protocol_masks else np.zeros((len(indices), 0), dtype=np.float32)
    )

    provenance = {
        "exam_id": exam_id if exam_id is not None else sample.get("exam_id"),
        "institute": institute if institute is not None else sample.get("institute"),
        "source": "simulated_from_cpet_sequence",
        "channel_sources": channel_sources,
    }
    manifest = wearable_manifest(config, columns)
    manifest["output_len"] = int(wearable_x.shape[0])

    return WearableSample(
        x=torch.tensor(wearable_x, dtype=torch.float32),
        mask=torch.tensor(wearable_mask, dtype=torch.float32),
        channels=tuple(config.selected_channels),
        sampling_rate_hz=config.sampling_rate_hz,
        protocol_x=torch.tensor(protocol_x, dtype=torch.float32),
        protocol_mask=torch.tensor(protocol_mask, dtype=torch.float32),
        protocol_channels=tuple(config.protocol_channels),
        provenance=provenance,
        manifest=manifest,
    )


def simulate_action_conditioned_wearable_sample(
    sample: Mapping[str, Any],
    feature_columns: tuple[str, ...] | list[str],
    config: WearableSimulationConfig,
    prefix_steps: int,
    protocol_plan: Mapping[str, np.ndarray | list[float]],
    exam_id: str | None = None,
    institute: str | None = None,
) -> WearableSample:
    values = _as_numpy(sample["x"]).astype(np.float32, copy=True)
    source_mask = _expand_mask(_as_numpy(sample.get("mask")), values.shape)
    steps = values.shape[0]
    prefix_steps = max(1, min(steps, int(prefix_steps)))
    causal_values = values.copy()
    causal_values[prefix_steps:, :] = causal_values[prefix_steps - 1, :]
    causal_mask = source_mask.copy()
    causal_mask[prefix_steps:, :] = 0.0
    causal_sample = {**dict(sample), "x": torch.tensor(causal_values, dtype=torch.float32), "mask": torch.tensor(causal_mask, dtype=torch.float32)}
    watch_config = replace(config, protocol_channels=())
    watch = simulate_wearable_sample(causal_sample, feature_columns, watch_config, exam_id=exam_id, institute=institute)
    wearable_x = watch.x.clone()
    wearable_mask = watch.mask.clone()
    wearable_x[prefix_steps:, :] = 0.0
    wearable_mask[prefix_steps:, :] = 0.0

    indices = _downsample_indices(steps, config.max_len)
    protocol_x, protocol_mask = _protocol_context_from_plan(
        values,
        source_mask,
        tuple(feature_columns),
        sample,
        config,
        protocol_plan,
        indices,
    )
    provenance = {
        **watch.provenance,
        "source": "causal_prefix_wearable_with_planned_protocol",
        "prefix_steps": prefix_steps,
        "uses_future_observed_wearable": False,
        "uses_future_actual_protocol": False,
        "uses_planned_protocol": True,
    }
    manifest = wearable_manifest(config, feature_columns)
    manifest["output_len"] = int(wearable_x.shape[0])
    manifest["forecast_mode"] = "action_conditioned_online"

    return WearableSample(
        x=wearable_x,
        mask=wearable_mask,
        channels=watch.channels,
        sampling_rate_hz=watch.sampling_rate_hz,
        protocol_x=torch.tensor(protocol_x, dtype=torch.float32),
        protocol_mask=torch.tensor(protocol_mask, dtype=torch.float32),
        protocol_channels=tuple(config.protocol_channels),
        provenance=provenance,
        manifest=manifest,
    )


def wearable_manifest(
    config: WearableSimulationConfig,
    feature_columns: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    return {
        "kind": "simulated_wearable_cpet",
        "workout_scenario": config.workout_scenario,
        "sampling_rate_hz": config.sampling_rate_hz,
        "max_len": config.max_len,
        "selected_channels": list(config.selected_channels),
        "watch_channels": list(config.selected_channels),
        "protocol_context_channels": list(config.protocol_channels),
        "lag_steps": config.lag_steps,
        "lag_channels": list(config.lag_channels),
        "noise_std": config.noise_std,
        "dropout_probability": config.dropout_probability,
        "quantization_step": config.quantization_step,
        "jitter_std": config.jitter_std,
        "seed": config.seed,
        "feature_columns": list(feature_columns),
    }


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=np.float32)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _expand_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.size == 0:
        return np.ones(shape, dtype=np.float32)
    if mask.ndim == 1:
        return np.repeat(mask.astype(np.float32)[:, None], shape[1], axis=1)
    if mask.shape == shape:
        return mask.astype(np.float32, copy=False)
    raise ValueError(f"mask shape {mask.shape} is incompatible with x shape {shape}")


def _time_support(sample: Mapping[str, Any], source_mask: np.ndarray) -> np.ndarray:
    """Time support is independent of individual signal availability."""
    if "time_mask" in sample:
        support = _as_numpy(sample["time_mask"])
        if support.shape != (source_mask.shape[0],):
            raise ValueError("time_mask must match the source time axis")
        return (np.isfinite(support) & (support > 0)).astype(np.float32)
    return (source_mask > 0).any(axis=1).astype(np.float32)


def _downsample_indices(source_len: int, max_len: int | None) -> np.ndarray:
    if source_len <= 0:
        raise ValueError("sample x must contain at least one timestep")
    if max_len is None or max_len == source_len:
        return np.arange(source_len, dtype=np.int64)
    if max_len == 1:
        return np.array([0], dtype=np.int64)
    return np.rint(np.linspace(0, source_len - 1, max_len)).astype(np.int64)


def _protocol_context_from_plan(
    values: np.ndarray,
    source_mask: np.ndarray,
    feature_columns: tuple[str, ...],
    sample: Mapping[str, Any],
    config: WearableSimulationConfig,
    protocol_plan: Mapping[str, np.ndarray | list[float]],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    protocol_values: list[np.ndarray] = []
    protocol_masks: list[np.ndarray] = []
    source_len = values.shape[0]
    for channel in config.protocol_channels:
        if channel in protocol_plan:
            series = np.asarray(protocol_plan[channel], dtype=np.float32)
            if series.shape[0] == source_len:
                series = series[indices]
            elif series.shape[0] != len(indices):
                raise ValueError(f"protocol plan channel {channel} has length {series.shape[0]}, expected {source_len} or {len(indices)}")
            mask = np.isfinite(series).astype(np.float32)
            protocol_values.append(np.nan_to_num(series, nan=0.0).astype(np.float32, copy=False))
            protocol_masks.append(mask)
            continue
        series, series_mask, _source = _select_protocol_channel(channel, values, source_mask, feature_columns, sample)
        protocol_values.append(series[indices].astype(np.float32, copy=True))
        protocol_masks.append(series_mask[indices].astype(np.float32, copy=True))
    if not protocol_values:
        return np.zeros((len(indices), 0), dtype=np.float32), np.zeros((len(indices), 0), dtype=np.float32)
    return np.stack(protocol_values, axis=1).astype(np.float32), np.stack(protocol_masks, axis=1).astype(np.float32)


def _select_channel(
    channel: str,
    values: np.ndarray,
    source_mask: np.ndarray,
    feature_columns: tuple[str, ...],
    sample: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, str | None]:
    if channel in feature_columns:
        column_idx = feature_columns.index(channel)
        series = values[:, column_idx].astype(np.float32, copy=True)
        finite = np.isfinite(series)
        mask = (source_mask[:, column_idx] > 0.0) & finite
        return np.where(mask, series, 0.0), mask.astype(np.float32), channel

    if channel == "Load_Phase" and "phase" in sample:
        phase = _as_numpy(sample["phase"]).astype(np.float32, copy=True)
        finite = np.isfinite(phase)
        observed = _as_numpy(sample.get("phase_mask", finite)).astype(bool)
        mask = finite & observed & (_time_support(sample, source_mask) > 0)
        return np.where(mask, phase, 0.0), mask.astype(np.float32), "phase"

    zeros = np.zeros(values.shape[0], dtype=np.float32)
    return zeros, zeros, None


def _select_watch_channel(
    channel: str,
    values: np.ndarray,
    source_mask: np.ndarray,
    feature_columns: tuple[str, ...],
    sample: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, str | None]:
    hr, hr_mask, hr_source = _select_channel("HR", values, source_mask, feature_columns, sample)
    if channel in feature_columns or channel in {"Time", "SpO2"}:
        return _select_channel(channel, values, source_mask, feature_columns, sample)
    if channel == "HR_smooth":
        series, mask = _neighborhood_feature(hr, hr_mask, _time_support(sample, source_mask), "mean")
        return series, mask, "derived_from_HR"
    if channel == "HR_slope":
        series, mask = _neighborhood_feature(hr, hr_mask, _time_support(sample, source_mask), "gradient")
        return series, mask, "derived_from_HR"
    if channel == "HR_acceleration":
        support = _time_support(sample, source_mask)
        slope, slope_mask = _neighborhood_feature(hr, hr_mask, support, "gradient")
        series, mask = _neighborhood_feature(slope, slope_mask, support, "gradient")
        return series, mask, "derived_from_HR"
    if channel == "HR_rolling_std":
        series, mask = _neighborhood_feature(hr, hr_mask, _time_support(sample, source_mask), "std")
        return series, mask, "derived_from_HR"
    if channel == "HR_reserve_pct":
        return _minmax(hr, hr_mask), hr_mask, "derived_from_HR"
    if channel == "HR_recovery_proxy":
        phase, phase_mask, _source = _select_protocol_channel("Load_Phase", values, source_mask, feature_columns, sample)
        peak = np.maximum.accumulate(np.where(hr_mask > 0, hr, -np.inf))
        recovery = np.where((phase >= 2.0) & (hr_mask > 0), peak - hr, 0.0).astype(np.float32)
        return _minmax(recovery, hr_mask), (hr_mask * phase_mask).astype(np.float32), "derived_from_HR_phase"
    if channel in {"active_energy_proxy", "active_energy_rate_proxy"}:
        phase, phase_mask, _source = _select_protocol_channel("Load_Phase", values, source_mask, feature_columns, sample)
        power, power_mask, _source = _select_protocol_channel("Power_Load", values, source_mask, feature_columns, sample)
        reserve = _minmax(hr, hr_mask)
        phase_component = _minmax(phase, phase_mask)
        power_component = _minmax(power, power_mask)
        proxy = (0.62 * reserve + 0.18 * phase_component + 0.20 * power_component).astype(np.float32)
        return proxy, np.maximum.reduce([hr_mask, phase_mask, power_mask]).astype(np.float32), "watch_proxy_from_HR_power_bike_protocol"
    if channel in {"motion_intensity_proxy", "wrist_motion_proxy"}:
        power, power_mask, _source = _select_protocol_channel("Power_Load", values, source_mask, feature_columns, sample)
        rpm, rpm_mask, _source = _select_protocol_channel("RPM", values, source_mask, feature_columns, sample)
        phase, phase_mask, _source = _select_protocol_channel("Load_Phase", values, source_mask, feature_columns, sample)
        mask = np.maximum.reduce([power_mask, rpm_mask, phase_mask]).astype(np.float32)
        proxy = (0.5 * _minmax(power, power_mask) + 0.3 * _minmax(rpm, rpm_mask) + 0.2 * _minmax(phase, phase_mask)).astype(np.float32)
        return proxy, mask, "watch_proxy_from_power_bike_protocol"
    if channel == "signal_quality":
        quality = hr_mask.astype(np.float32, copy=True)
        return quality, _time_support(sample, source_mask), hr_source or "derived_from_HR_mask"
    return _select_channel(channel, values, source_mask, feature_columns, sample)


def _select_protocol_channel(
    channel: str,
    values: np.ndarray,
    source_mask: np.ndarray,
    feature_columns: tuple[str, ...],
    sample: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    if channel == "Load_Phase":
        series, mask, _source = _select_channel(channel, values, source_mask, feature_columns, sample)
        return series, mask, "protocol_context"
    if channel == "ramp_slope":
        power, power_mask, _source = _select_channel("Power_Load", values, source_mask, feature_columns, sample)
        series, mask = _neighborhood_feature(power, power_mask, _time_support(sample, source_mask), "gradient")
        return series, mask, "protocol_context"
    series, mask, _source = _select_channel(channel, values, source_mask, feature_columns, sample)
    return series, mask, "protocol_context"


def _neighborhood_feature(
    values: np.ndarray, mask: np.ndarray, time_support: np.ndarray, kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Require the actual stencil, clipped to the sequence rather than batch padding."""
    output = np.zeros_like(values, dtype=np.float32)
    output_mask = np.zeros_like(mask, dtype=np.float32)
    supported = np.flatnonzero(time_support > 0)
    if not supported.size:
        return output, output_mask
    start, stop = int(supported[0]), int(supported[-1]) + 1
    observed = (mask[start:stop] > 0) & (time_support[start:stop] > 0) & np.isfinite(values[start:stop])
    clean = np.where(observed, values[start:stop], 0.0)
    if kind == "gradient":
        if len(clean) < 2:
            return output, output_mask
        result = _gradient(clean)
        valid = observed.copy()
        valid[1:] &= observed[:-1]
        valid[:-1] &= observed[1:]
    else:
        lo, hi, counts = _rolling_bounds(len(clean), 5)
        prefix = np.concatenate(([0], np.cumsum(observed)))
        valid = (prefix[hi] - prefix[lo]) == counts
        result = _rolling_mean(clean) if kind == "mean" else _rolling_std(clean)
    output[start:stop] = np.where(valid, result, 0.0)
    output_mask[start:stop] = valid.astype(np.float32)
    return output, output_mask


def _rolling_mean(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float32, copy=True)
    clean = values.astype(np.float64, copy=False)
    lo, hi, counts = _rolling_bounds(len(clean), window)
    prefix = np.concatenate(([0.0], np.cumsum(clean, dtype=np.float64)))
    output = (prefix[hi] - prefix[lo]) / counts
    return output.astype(np.float32)


def _rolling_std(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float32, copy=True)
    clean = values.astype(np.float64, copy=False)
    lo, hi, counts = _rolling_bounds(len(clean), window)
    prefix = np.concatenate(([0.0], np.cumsum(clean, dtype=np.float64)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(clean * clean, dtype=np.float64)))
    mean = (prefix[hi] - prefix[lo]) / counts
    mean_sq = (prefix_sq[hi] - prefix_sq[lo]) / counts
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(variance).astype(np.float32)


def _rolling_bounds(length: int, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = max(window // 2, 1)
    idx = np.arange(length)
    lo = np.maximum(0, idx - half)
    hi = np.minimum(length, idx + half + 1)
    counts = (hi - lo).astype(np.float64)
    return lo, hi, counts


def _gradient(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float32)
    return np.gradient(values.astype(np.float32, copy=False)).astype(np.float32)


def _minmax(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    observed = mask > 0.0
    if not observed.any():
        return np.zeros_like(values, dtype=np.float32)
    clean = values.astype(np.float32, copy=False)
    low = float(np.min(clean[observed]))
    high = float(np.max(clean[observed]))
    span = max(high - low, 1e-6)
    return np.where(observed, (clean - low) / span, 0.0).astype(np.float32)


def _lag(series: np.ndarray, mask: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    if steps == 0:
        return series, mask
    lagged = np.zeros_like(series)
    lagged_mask = np.zeros_like(mask)
    if steps < len(series):
        lagged[steps:] = series[:-steps]
        lagged_mask[steps:] = mask[:-steps]
    return lagged, lagged_mask


def _apply_corruption(
    values: np.ndarray,
    mask: np.ndarray,
    config: WearableSimulationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    corrupted = values.astype(np.float32, copy=True)
    corrupted_mask = mask.astype(np.float32, copy=True)
    observed = corrupted_mask > 0.0

    if config.noise_std > 0.0:
        noise = rng.normal(0.0, config.noise_std, size=corrupted.shape).astype(np.float32)
        corrupted = np.where(observed, corrupted + noise, corrupted)

    if config.quantization_step is not None:
        step = np.float32(config.quantization_step)
        corrupted = np.where(observed, np.round(corrupted / step) * step, corrupted)

    if config.dropout_probability > 0.0:
        keep = rng.random(corrupted.shape) >= config.dropout_probability
        corrupted_mask = np.where(keep, corrupted_mask, 0.0).astype(np.float32)

    corrupted = np.where(corrupted_mask > 0.0, corrupted, 0.0).astype(np.float32)
    return corrupted, corrupted_mask
