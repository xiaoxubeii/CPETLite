"""Endpoint-only baselines for CPETLite evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_FEATURE_COLUMNS = ("Time", "Load_Phase", "Power_Load", "RPM")


@dataclass(frozen=True)
class EndpointFeatureMatrix:
    features: np.ndarray
    at_norm: np.ndarray
    at_time_s: np.ndarray
    time_start_s: np.ndarray
    duration_s: np.ndarray
    at_supervision_eligible: np.ndarray
    at_metric_eligible: np.ndarray
    peak_vo2: np.ndarray
    peak_eligible: np.ndarray


def _array(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    return np.asarray(value)


def _scalar(sample: Mapping[str, object], key: str, default: float = 0.0) -> float:
    if key not in sample:
        return default
    return float(_array(sample[key]).reshape(-1)[0])


def build_endpoint_feature_matrix(
    samples: Iterable[Mapping[str, object]],
    feature_columns: Sequence[str],
    mode: str,
) -> EndpointFeatureMatrix:
    """Build metadata/protocol endpoint predictors without target trajectories."""
    if mode not in {"metadata", "protocol", "metadata_protocol"}:
        raise ValueError(f"Unknown endpoint baseline mode: {mode}")
    feature_index = {name: index for index, name in enumerate(feature_columns)}
    protocol_indices = [feature_index[name] for name in PROTOCOL_FEATURE_COLUMNS if name in feature_index]
    rows: list[list[float]] = []
    at_norm: list[float] = []
    at_time_s: list[float] = []
    time_start_s: list[float] = []
    duration_s: list[float] = []
    at_supervision_eligible: list[bool] = []
    at_metric_eligible: list[bool] = []
    peak_vo2: list[float] = []
    peak_eligible: list[bool] = []
    for sample in samples:
        values: list[float] = []
        if mode in {"metadata", "metadata_protocol"}:
            values.extend(_array(sample["metadata"]).astype(float).reshape(-1).tolist())
        if mode in {"protocol", "metadata_protocol"}:
            x = _array(sample["x"]).astype(float)
            valid = _array(sample["mask"]).reshape(-1) > 0
            for index in protocol_indices:
                series = x[valid, index]
                series = series[np.isfinite(series)]
                if series.size:
                    values.extend(
                        [
                            float(series.mean()),
                            float(series.std()),
                            float(series.min()),
                            float(series.max()),
                            float(series[-1]),
                        ]
                    )
                else:
                    values.extend([0.0] * 5)
        rows.append(values)
        at_norm.append(_scalar(sample, "at_norm"))
        at_time_s.append(_scalar(sample, "at_time_s"))
        time_start_s.append(_scalar(sample, "time_start_s"))
        duration_s.append(_scalar(sample, "duration_s", 1.0))
        at_supervision_eligible.append(_scalar(sample, "at_supervision_mask", 1.0) > 0)
        at_metric_eligible.append(_scalar(sample, "at_metric_mask", 1.0) > 0)
        peak_vo2.append(_scalar(sample, "vo2peak"))
        peak_eligible.append(_scalar(sample, "vo2peak_mask", 1.0) > 0)
    return EndpointFeatureMatrix(
        features=np.asarray(rows, dtype=np.float32),
        at_norm=np.asarray(at_norm, dtype=np.float32),
        at_time_s=np.asarray(at_time_s, dtype=np.float32),
        time_start_s=np.asarray(time_start_s, dtype=np.float32),
        duration_s=np.asarray(duration_s, dtype=np.float32),
        at_supervision_eligible=np.asarray(at_supervision_eligible, dtype=bool),
        at_metric_eligible=np.asarray(at_metric_eligible, dtype=bool),
        peak_vo2=np.asarray(peak_vo2, dtype=np.float32),
        peak_eligible=np.asarray(peak_eligible, dtype=bool),
    )
