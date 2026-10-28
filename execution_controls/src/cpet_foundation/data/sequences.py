from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cpet_foundation.data.h5_registry import load_pickled_dataset


SUBMAXIMAL_POST_AT_CENTERS = frozenset({"shanxi"})
SUPERVISION_POLICY_COLUMNS = (
    "trajectory_supervision_eligible",
    "peak_supervision_eligible",
    "at_supervision_eligible",
    "at_metric_eligible",
)


def examination_token(exam_id: str) -> tuple[int, int]:
    """Stable opaque join key; not an inferred subject identity or privacy proof."""
    digest = hashlib.sha256(str(exam_id).encode("utf-8")).digest()
    return tuple(int.from_bytes(digest[i:i + 8], "big") & ((1 << 63) - 1) for i in (0, 8))


@dataclass(frozen=True)
class SequenceConfig:
    source_h5: Path
    feature_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    max_len: int = 160
    target_column: str = "Time_at_AT"
    limit_exams: int | None = None
    exam_ids: tuple[str, ...] | None = None
    normalization_stats_path: Path | None = None
    require_frozen_normalization: bool = False
    require_supervision_policy: bool = False
    # "registry" retains the historical pickled-institute HDF5 reader.
    # "v5_derivative_split" reads one Standard-v5 derivative split, including
    # producer-supplied validity and task-eligibility masks.
    input_format: str = "registry"
    split_name: str | None = None


@dataclass(frozen=True)
class FrozenNormalizationStats:
    path: Path
    feature_stats: dict[str, tuple[float, float]]
    metadata_stats: dict[str, tuple[float, float]]
    fit_population: dict[str, Any]


def _metadata_validity(values: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
    valid = np.isfinite(values)
    for index, name in enumerate(columns):
        if name in {"Age", "Height_cm", "Weight_kg"}:
            valid[index] &= values[index] > 0
    return valid


def normalization_options_from_config(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    raw_path = config.get("normalization_stats_path")
    require_frozen = bool(config.get("require_frozen_normalization", False))
    require_supervision_policy = bool(config.get("require_supervision_policy", False))
    if raw_path in (None, ""):
        return {
            "normalization_stats_path": None,
            "require_frozen_normalization": require_frozen,
            "require_supervision_policy": require_supervision_policy,
        }
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    return {
        "normalization_stats_path": path,
        "require_frozen_normalization": require_frozen,
        "require_supervision_policy": require_supervision_policy,
    }


def _load_stats_table(
    payload: Mapping[str, Any],
    table_name: str,
    columns: tuple[str, ...],
    path: Path,
) -> dict[str, tuple[float, float]]:
    table = payload.get(table_name)
    if not isinstance(table, Mapping):
        raise ValueError(f"Normalization artifact {path} is missing the {table_name!r} table")
    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(f"Normalization artifact {path} is missing {table_name} columns: {missing}")

    result: dict[str, tuple[float, float]] = {}
    for column in columns:
        entry = table[column]
        if not isinstance(entry, Mapping):
            raise ValueError(f"Normalization artifact {path} has an invalid entry for {table_name}.{column}")
        if not bool(entry.get("standardize", True)):
            result[column] = (0.0, 1.0)
            continue
        try:
            mean = float(entry["mean"])
            std = float(entry["std_ddof1"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Normalization artifact {path} has invalid mean/std_ddof1 for {table_name}.{column}"
            ) from exc
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-6:
            raise ValueError(
                f"Normalization artifact {path} requires a finite mean and positive std_ddof1 for "
                f"{table_name}.{column}"
            )
        result[column] = (mean, std)
    return result


def load_frozen_normalization(config: SequenceConfig) -> FrozenNormalizationStats | None:
    path = config.normalization_stats_path
    if path is None:
        if config.require_frozen_normalization:
            raise ValueError("Strict frozen normalization requires normalization_stats_path")
        return None
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Normalization artifact does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read normalization artifact {path}: {exc}") from exc
    if payload.get("schema_version") not in {1, 2}:
        raise ValueError(
            f"Normalization artifact {path} has unsupported schema_version={payload.get('schema_version')!r}"
        )
    fit_population = payload.get("fit_population")
    if not isinstance(fit_population, dict) or not fit_population.get("split"):
        raise ValueError(f"Normalization artifact {path} is missing fit_population.split")
    return FrozenNormalizationStats(
        path=path,
        feature_stats=_load_stats_table(payload, "features", config.feature_columns, path),
        metadata_stats=_load_stats_table(payload, "metadata", config.metadata_columns, path),
        fit_population=dict(fit_population),
    )


def _resample_array(values: np.ndarray, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    values = values.astype(np.float32, copy=False)
    if len(values) == 0:
        return np.zeros((max_len, values.shape[1]), dtype=np.float32), np.zeros(max_len, dtype=np.float32)
    if len(values) == max_len:
        return values, np.ones(max_len, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, max_len)
    output = np.zeros((max_len, values.shape[1]), dtype=np.float32)
    for idx in range(values.shape[1]):
        col = values[:, idx]
        finite = np.isfinite(col)
        if finite.sum() < 2:
            fill = float(np.nanmean(col)) if finite.any() else 0.0
            output[:, idx] = fill
        else:
            output[:, idx] = np.interp(x_new, x_old[finite], col[finite]).astype(np.float32)
    return output, np.ones(max_len, dtype=np.float32)


def _resample_feature_availability(values: np.ndarray, max_len: int) -> np.ndarray:
    """Keep processed finite-value availability separate from interpolated values."""
    finite = np.isfinite(values)
    if len(values) == max_len:
        return finite.astype(np.float32)
    if len(values) == 0:
        return np.zeros((max_len, values.shape[1]), dtype=np.float32)
    nearest = np.rint(np.linspace(0, len(values) - 1, max_len)).astype(int)
    return finite[nearest].astype(np.float32)


def _as_h5_float(values: np.ndarray) -> np.ndarray:
    """Convert a numeric-or-string HDF5 vector without replacing missingness."""
    if values.dtype.kind in "OUS":
        return pd.to_numeric(pd.Series(values.astype(str)), errors="coerce").to_numpy(dtype=np.float32)
    return values.astype(np.float32, copy=False)


def _v5_scalers(
    split: h5py.Group,
    feature_columns: tuple[str, ...],
    metadata_columns: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read saved derivative scalers; the exporter guarantees train reuse."""
    root = split.get("metadata/standardization/tables")
    if root is None:
        raise ValueError(f"{split.name} is missing saved standardization metadata")

    def table(name: str, required: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        if name not in root:
            raise ValueError(f"{split.name} lacks saved scaler table {name!r}")
        node = root[name]
        columns = node["columns"].asstr()[:].tolist()
        mean, scale = node["mean"][:].astype(np.float32), node["scale"][:].astype(np.float32)
        if len(columns) != len(mean) or mean.shape != scale.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError(f"{split.name} has invalid saved scaler {name!r}")
        # Fields not selected for standardization (for example Load_Phase or
        # Gender) intentionally have identity parameters rather than a fitted
        # statistical transform.
        output_mean, output_scale = [], []
        for column in required:
            if column in columns:
                index = columns.index(column)
                output_mean.append(mean[index])
                output_scale.append(scale[index])
            else:
                output_mean.append(0.0)
                output_scale.append(1.0)
        return np.asarray(output_mean, dtype=np.float32), np.asarray(output_scale, dtype=np.float32)

    return {
        "features": table("features", feature_columns),
        "metadata": table("metadata_table", metadata_columns),
    }


def _resample_v5_without_imputation(
    values: np.ndarray,
    availability: np.ndarray,
    max_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select source rows only; unavailable cells remain false, never interpolated."""
    if values.ndim != 2 or availability.shape != values.shape or values.shape[0] == 0:
        raise ValueError("V5 values and availability require aligned nonempty [time, channel] arrays")
    if max_len < 1:
        raise ValueError("max_len must be positive")
    if len(values) <= max_len:
        selected = np.arange(len(values))
    else:
        selected = np.rint(np.linspace(0, len(values) - 1, max_len)).astype(np.int64)
    selected_values, selected_mask = values[selected].copy(), availability[selected].copy()
    output = np.zeros((max_len, values.shape[1]), dtype=np.float32)
    output_mask = np.zeros((max_len, values.shape[1]), dtype=bool)
    seq_mask = np.zeros(max_len, dtype=np.float32)
    count = len(selected)
    output[:count] = np.where(selected_mask, selected_values, 0.0)
    output_mask[:count] = selected_mask
    seq_mask[:count] = 1.0
    return output, output_mask, seq_mask


def _resample_phase(values: np.ndarray, max_len: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(max_len, dtype=np.int64)
    clean = pd.to_numeric(pd.Series(values), errors="coerce").ffill().bfill().fillna(0).to_numpy(dtype=np.float32)
    if len(clean) == max_len:
        return np.rint(clean).clip(0, 3).astype(np.int64)
    positions = np.linspace(0, max(len(clean) - 1, 0), max_len)
    nearest = np.rint(positions).astype(int).clip(0, len(clean) - 1)
    return np.rint(clean[nearest]).clip(0, 3).astype(np.int64)


def _standardize_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    skip_columns: set[str] | None = None,
    frozen_stats: Mapping[str, tuple[float, float]] | None = None,
    preserve_missing: bool = False,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    stats: dict[str, tuple[float, float]] = {}
    standardized = frame.copy()
    skip_columns = skip_columns or set()
    for col in columns:
        if frozen_stats is not None:
            mean, std = frozen_stats[col]
            values = pd.to_numeric(standardized[col], errors="coerce")
            result = (values - mean) / std
            standardized[col] = result if preserve_missing else result.fillna(0.0)
            stats[col] = (float(mean), float(std))
            continue
        if col in skip_columns:
            result = pd.to_numeric(standardized[col], errors="coerce")
            standardized[col] = result if preserve_missing else result.fillna(0.0)
            stats[col] = (0.0, 1.0)
            continue
        values = pd.to_numeric(standardized[col], errors="coerce")
        mean = float(values.mean()) if np.isfinite(values.mean()) else 0.0
        std = float(values.std()) if np.isfinite(values.std()) and values.std() > 1e-6 else 1.0
        result = (values - mean) / std
        standardized[col] = result if preserve_missing else result.fillna(0.0)
        stats[col] = (mean, std)
    return standardized, stats


class CpetSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, config: SequenceConfig) -> None:
        self.config = config
        self.frozen_normalization = load_frozen_normalization(config)
        self.samples = self._load_samples()

    def _load_samples(self) -> list[dict[str, np.ndarray | float | str]]:
        if self.config.input_format == "v5_derivative_split":
            return self._load_v5_derivative_split()
        if self.config.input_format != "registry":
            raise ValueError(f"Unsupported input_format={self.config.input_format!r}")
        samples: list[dict[str, np.ndarray | float | str]] = []
        allowed_ids = set(self.config.exam_ids) if self.config.exam_ids else None
        with h5py.File(self.config.source_h5, "r") as h5:
            for institute_name in h5["institutes"].keys():
                group = h5["institutes"][institute_name]
                features = load_pickled_dataset(group, "features")
                metadata = load_pickled_dataset(group, "metadata")
                targets = load_pickled_dataset(group["targets"], "raw")
                if allowed_ids is not None:
                    features = features[
                        features["Examination_ID"].astype(str).isin(allowed_ids)
                    ].copy()
                    metadata = metadata[
                        metadata["Examination_ID"].astype(str).isin(allowed_ids)
                    ].copy()
                    targets = targets[
                        targets["Examination_ID"].astype(str).isin(allowed_ids)
                    ].copy()
                    if features.empty:
                        continue
                missing_policy = [
                    column for column in SUPERVISION_POLICY_COLUMNS if column not in targets.columns
                ]
                if self.config.require_supervision_policy and missing_policy:
                    raise ValueError(
                        f"{institute_name}/targets/raw is missing required supervision policy columns: "
                        f"{missing_policy}"
                    )
                for col in self.config.feature_columns:
                    if col not in features.columns:
                        features[col] = np.nan
                raw_features = features.copy()
                for col in self.config.metadata_columns:
                    if col not in metadata.columns:
                        metadata[col] = np.nan
                raw_metadata = metadata.copy()
                feature_frozen = (
                    self.frozen_normalization.feature_stats if self.frozen_normalization is not None else None
                )
                metadata_frozen = (
                    self.frozen_normalization.metadata_stats if self.frozen_normalization is not None else None
                )
                features, feature_stats = _standardize_frame(
                    features,
                    self.config.feature_columns,
                    skip_columns={"Load_Phase"},
                    frozen_stats=feature_frozen,
                    preserve_missing=True,
                )
                metadata, _ = _standardize_frame(
                    metadata,
                    self.config.metadata_columns,
                    frozen_stats=metadata_frozen,
                )
                meta_map = metadata.drop_duplicates("Examination_ID").set_index("Examination_ID")
                raw_meta_map = raw_metadata.drop_duplicates("Examination_ID").set_index("Examination_ID")
                target_map = targets.drop_duplicates("Examination_ID").set_index("Examination_ID")
                raw_groups = raw_features.groupby("Examination_ID", sort=False)
                for exam_id, exam_frame in features.groupby("Examination_ID", sort=False):
                    exam_id_str = str(exam_id)
                    if exam_id not in target_map.index or exam_id not in meta_map.index:
                        continue
                    raw_exam_frame = raw_groups.get_group(exam_id)
                    raw_time = pd.to_numeric(raw_exam_frame["Time"], errors="coerce").to_numpy(dtype=np.float32)
                    finite_time = raw_time[np.isfinite(raw_time)]
                    if len(finite_time) < 2:
                        continue
                    time_start = float(np.nanmin(finite_time))
                    duration = float(np.nanmax(finite_time) - time_start)
                    if not np.isfinite(duration) or duration <= 0:
                        continue
                    at_seconds = pd.to_numeric(pd.Series([target_map.loc[exam_id, self.config.target_column]]), errors="coerce").iloc[0]
                    if not np.isfinite(at_seconds):
                        continue
                    raw_values = exam_frame[list(self.config.feature_columns)].to_numpy(dtype=np.float32)
                    sequence, mask = _resample_array(raw_values, self.config.max_len)
                    processed_availability = _resample_feature_availability(
                        raw_exam_frame[list(self.config.feature_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32),
                        self.config.max_len,
                    )
                    phase = _resample_phase(
                        raw_exam_frame.get("Load_Phase", pd.Series(np.zeros(len(raw_exam_frame)))).to_numpy(),
                        self.config.max_len,
                    )
                    at_norm = np.clip((float(at_seconds) - time_start) / duration, 0.0, 1.5)
                    vo2peak_source = raw_exam_frame["VO2_kg"] if "VO2_kg" in raw_exam_frame.columns else raw_exam_frame["VO2"]
                    finite_vo2peak = pd.to_numeric(vo2peak_source, errors="coerce").dropna()
                    vo2peak = float(finite_vo2peak.max()) if not finite_vo2peak.empty else float("nan")
                    target_row = target_map.loc[exam_id]
                    trajectory_supervision = _eligibility_value(
                        target_row,
                        "trajectory_supervision_eligible",
                        default=True,
                    )
                    peak_supervision = _eligibility_value(
                        target_row,
                        "peak_supervision_eligible",
                        default=np.isfinite(vo2peak),
                    )
                    at_supervision = _eligibility_value(
                        target_row,
                        "at_supervision_eligible",
                        default=np.isfinite(at_seconds),
                    )
                    at_metric = _eligibility_value(
                        target_row,
                        "at_metric_eligible",
                        default=np.isfinite(at_seconds),
                    )
                    is_submaximal_post_at = institute_name in SUBMAXIMAL_POST_AT_CENTERS
                    sequence_completeness = "submaximal_post_at" if is_submaximal_post_at else "complete_or_peak_available"
                    vo2peak_label_provenance = "observed_submax_vo2_max" if is_submaximal_post_at else "derived_sequence_peak"
                    vo2peak_mask = float(peak_supervision and not is_submaximal_post_at)
                    meta = meta_map.loc[exam_id, list(self.config.metadata_columns)].to_numpy(dtype=np.float32)
                    raw_meta = raw_meta_map.loc[exam_id, list(self.config.metadata_columns)].to_numpy(dtype=np.float32)
                    subject_value = raw_meta_map.loc[exam_id, "Subject_ID"] if "Subject_ID" in raw_meta_map else None
                    subject = str(subject_value).strip() if pd.notna(subject_value) else ""
                    group_key = f"{institute_name}:{subject}" if subject and subject.lower() != "nan" else f"{institute_name}:exam:{exam_id_str}"
                    probe_fold = int.from_bytes(hashlib.sha256(group_key.encode()).digest()[:8], "big") % 3
                    samples.append(
                        {
                            "exam_id": exam_id_str,
                            "institute": institute_name,
                            "x": sequence,
                            "mask": mask,
                            "feature_mask": processed_availability,
                            "probe_fold": probe_fold,
                            "phase": phase,
                            "metadata": np.nan_to_num(meta, nan=0.0),
                            "metadata_raw": np.nan_to_num(raw_meta, nan=0.0),
                            "metadata_mask": _metadata_validity(raw_meta, self.config.metadata_columns),
                            "at_norm": float(at_norm),
                            "at_time_s": float(at_seconds),
                            "time_start_s": float(time_start),
                            "duration_s": float(duration),
                            "time_fraction": np.interp(
                                np.linspace(0., 1., self.config.max_len),
                                np.linspace(0., 1., len(raw_time)),
                                (raw_time - time_start) / duration,
                            ).astype(np.float32),
                            "vo2peak": float(0.0 if not np.isfinite(vo2peak) else vo2peak),
                            "vo2peak_mask": float(vo2peak_mask),
                            "trajectory_supervision_mask": float(trajectory_supervision),
                            "at_supervision_mask": float(at_supervision),
                            "at_metric_mask": float(at_metric),
                            "sequence_completeness": sequence_completeness,
                            "vo2peak_label_provenance": vo2peak_label_provenance,
                            "feature_mean": np.array([feature_stats[col][0] for col in self.config.feature_columns], dtype=np.float32),
                            "feature_std": np.array([feature_stats[col][1] for col in self.config.feature_columns], dtype=np.float32),
                        }
                    )
                    if self.config.limit_exams and len(samples) >= self.config.limit_exams:
                        return samples
        return samples

    def _load_v5_derivative_split(self) -> list[dict[str, np.ndarray | float | str]]:
        """Load a v5 derivative split without fabricating unavailable values.

        The HDF5 payload remains raw; its saved split-local scaler is fitted on
        train and reused for held-out splits. Missing positions become numeric
        zero only after their false `feature_mask` has been retained.
        """
        split_name = self.config.split_name
        if split_name not in {"train", "val", "test"}:
            raise ValueError("v5_derivative_split requires split_name=train, val, or test")
        allowed_ids = set(self.config.exam_ids) if self.config.exam_ids else None
        samples: list[dict[str, np.ndarray | float | str]] = []
        with h5py.File(self.config.source_h5, "r") as h5:
            prefix = f"splits/{split_name}"
            if prefix not in h5:
                raise ValueError(f"{self.config.source_h5} has no v5 split {split_name!r}")
            split = h5[prefix]
            for node in ("features", "metadata_table", "targets/raw", "validity_masks/features", "validity_masks/targets"):
                if node not in split:
                    raise ValueError(f"{split.name} is missing required v5 node {node!r}")
            features, metadata, targets = split["features"], split["metadata_table"], split["targets/raw"]
            required = ["Examination_ID", "Time", "Load_Phase", *self.config.feature_columns]
            missing = [name for name in required if name not in features]
            if missing:
                raise ValueError(f"{split.name} is missing features: {missing}")
            if "Examination_ID" not in metadata or "Examination_ID" not in targets:
                raise ValueError(f"{split.name} lacks examination identities")
            for name in dict.fromkeys((*self.config.feature_columns, "Load_Phase")):
                if name not in split["validity_masks/features"]:
                    raise ValueError(f"{split.name} lacks feature validity mask: {name}")
                if len(split["validity_masks/features"][name]) != len(features["Examination_ID"]):
                    raise ValueError(f"{split.name} has invalid mask length: {name}")
            for name in (self.config.target_column, "VO2_kg_peak_phase1"):
                if name not in targets or name not in split["validity_masks/targets"]:
                    raise ValueError(f"{split.name} lacks target or validity mask: {name}")

            feature_ids = features["Examination_ID"].asstr()[:].astype(str)
            metadata_ids = metadata["Examination_ID"].asstr()[:].astype(str)
            target_ids = targets["Examination_ID"].asstr()[:].astype(str)
            if len(set(metadata_ids)) != len(metadata_ids) or len(set(target_ids)) != len(target_ids):
                raise ValueError(f"{split.name} has duplicate metadata or target identities")
            target_index = {exam: index for index, exam in enumerate(target_ids)}
            metadata_index = {exam: index for index, exam in enumerate(metadata_ids)}
            grouped_rows = pd.DataFrame({"exam": feature_ids}).groupby("exam", sort=True).indices
            metadata_values = {name: _as_h5_float(metadata[name][:]) if name in metadata else np.full(len(metadata_ids), np.nan)
                               for name in self.config.metadata_columns}
            scalar = {
                name: _as_h5_float(features[name][:]) for name in self.config.feature_columns
            }
            observed = {name: split["validity_masks/features"][name][:].astype(bool) for name in self.config.feature_columns}
            time = _as_h5_float(features["Time"][:])
            phase = _as_h5_float(features["Load_Phase"][:])
            phase_observed = split["validity_masks/features/Load_Phase"][:].astype(bool)
            target_values = {name: _as_h5_float(targets[name][:]) for name in (self.config.target_column, "VO2_kg_peak_phase1")}
            target_validity = {name: split["validity_masks/targets"][name][:].astype(bool) for name in target_values}
            eligibility = split.get("supervision_eligibility")
            at_eligible = eligibility["at"][:].astype(bool) if eligibility is not None and "at" in eligibility else np.ones(len(target_ids), dtype=bool)
            peak_eligible = eligibility["peak"][:].astype(bool) if eligibility is not None and "peak" in eligibility else np.ones(len(target_ids), dtype=bool)
            if at_eligible.shape != target_ids.shape or peak_eligible.shape != target_ids.shape:
                raise ValueError(f"{split.name} has invalid endpoint-eligibility lengths")
            scalers = _v5_scalers(split, self.config.feature_columns, self.config.metadata_columns)

            for exam_id, rows in grouped_rows.items():
                if allowed_ids is not None and exam_id not in allowed_ids:
                    continue
                if exam_id not in metadata_index or exam_id not in target_index:
                    raise ValueError(f"{split.name} feature identity lacks metadata/target: {exam_id}")
                order = np.argsort(np.nan_to_num(time[rows], nan=np.inf), kind="stable")
                rows = rows[order]
                finite_time = np.isfinite(time[rows])
                if finite_time.sum() < 2:
                    continue
                rows = rows[finite_time]
                raw = np.column_stack([scalar[name][rows] for name in self.config.feature_columns]).astype(np.float32)
                availability = np.column_stack([observed[name][rows] for name in self.config.feature_columns]).astype(bool)
                # Do not trust an impossible producer mask.
                availability &= np.isfinite(raw)
                phase_valid = phase_observed[rows] & np.isin(phase[rows], [0, 1, 2, 3])
                if "Load_Phase" in self.config.feature_columns:
                    availability[:, self.config.feature_columns.index("Load_Phase")] &= phase_valid
                means, scales = scalers["features"]
                normalized = (raw - means[None, :]) / scales[None, :]
                # Select and pad every field together, including the unscaled
                # categorical phase. Missing phase is never forward/back-filled.
                raw_time = time[rows]
                time_start, duration = float(raw_time.min()), float(raw_time.max() - raw_time.min())
                if not np.isfinite(duration) or duration <= 0:
                    continue
                aligned, aligned_mask, seq_mask = _resample_v5_without_imputation(
                    np.column_stack((normalized, (raw_time - time_start) / duration, phase[rows])),
                    np.column_stack((availability, np.ones(len(rows), dtype=bool), phase_valid)), self.config.max_len,
                )
                values, feature_mask = aligned[:, :-2], aligned_mask[:, :-2]
                time_fraction = aligned[:, -2]
                phase_codes, phase_mask = aligned[:, -1].astype(np.int64), aligned_mask[:, -1]
                target_row = target_index[exam_id]
                at_seconds = float(target_values[self.config.target_column][target_row])
                at_valid = bool(target_validity[self.config.target_column][target_row] and np.isfinite(at_seconds) and at_eligible[target_row])
                peak = float(target_values["VO2_kg_peak_phase1"][target_row])
                peak_valid = bool(target_validity["VO2_kg_peak_phase1"][target_row] and np.isfinite(peak) and peak_eligible[target_row])
                meta_row = metadata_index[exam_id]
                meta_raw = np.asarray([metadata_values[name][meta_row] for name in self.config.metadata_columns], dtype=np.float32)
                meta_valid = _metadata_validity(meta_raw, self.config.metadata_columns)
                producer_masks = split.get("validity_masks/metadata")
                if producer_masks is not None:
                    for index, name in enumerate(self.config.metadata_columns):
                        if name in producer_masks:
                            if len(producer_masks[name]) != len(metadata_ids):
                                raise ValueError(f"{split.name} has invalid metadata mask length: {name}")
                            meta_valid[index] &= bool(producer_masks[name][meta_row])
                meta_mean, meta_scale = scalers["metadata"]
                meta = (meta_raw - meta_mean) / meta_scale
                meta[~meta_valid | ~np.isfinite(meta)] = 0.0
                subject = metadata["Subject_ID"].asstr()[meta_row] if "Subject_ID" in metadata else ""
                institute = metadata["Institute_Name"].asstr()[meta_row] if "Institute_Name" in metadata else ""
                group_key = f"{institute}:{subject}" if subject else f"{institute}:exam:{exam_id}"
                samples.append({
                    "exam_id": exam_id, "institute": institute,
                    "x": values, "mask": seq_mask, "feature_mask": feature_mask.astype(np.float32),
                    "time_fraction": time_fraction,
                    "probe_fold": int.from_bytes(hashlib.sha256(group_key.encode()).digest()[:8], "big") % 3,
                    "phase": phase_codes, "phase_mask": phase_mask.astype(np.float32),
                    "metadata": meta.astype(np.float32), "metadata_raw": np.where(meta_valid, meta_raw, 0.0),
                    "metadata_mask": meta_valid,
                    "at_norm": float(np.clip((at_seconds - time_start) / duration, 0.0, 1.5)) if at_valid else 0.0,
                    "at_time_s": at_seconds if at_valid else 0.0, "time_start_s": time_start, "duration_s": duration,
                    "vo2peak": peak if peak_valid else 0.0, "vo2peak_mask": float(peak_valid),
                    "trajectory_supervision_mask": 1.0, "at_supervision_mask": float(at_valid), "at_metric_mask": float(at_valid),
                    "sequence_completeness": "v5_missing_preserved", "vo2peak_label_provenance": "processed_phase1_peak",
                    "feature_mean": means.astype(np.float32), "feature_std": scales.astype(np.float32), "subject_id": str(subject),
                })
                if self.config.limit_exams and len(samples) >= self.config.limit_exams:
                    break
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "sample_index": torch.tensor(index, dtype=torch.long),
            "sample_token": torch.tensor(examination_token(sample["exam_id"]), dtype=torch.long),
            "x": torch.tensor(sample["x"], dtype=torch.float32),
            "mask": torch.tensor(sample["mask"], dtype=torch.float32),
            "feature_mask": torch.tensor(sample["feature_mask"], dtype=torch.float32),
            "probe_fold": torch.tensor(sample["probe_fold"], dtype=torch.long),
            "phase": torch.tensor(sample["phase"], dtype=torch.long),
            "phase_mask": torch.tensor(sample.get("phase_mask", np.ones_like(sample["mask"])), dtype=torch.float32),
            "metadata": torch.tensor(sample["metadata"], dtype=torch.float32),
            "metadata_raw": torch.tensor(sample["metadata_raw"], dtype=torch.float32),
            "metadata_mask": torch.tensor(sample.get("metadata_mask", _metadata_validity(
                np.asarray(sample["metadata_raw"]), self.config.metadata_columns)), dtype=torch.float32),
            "at_norm": torch.tensor(sample["at_norm"], dtype=torch.float32),
            "at_time_s": torch.tensor(sample["at_time_s"], dtype=torch.float32),
            "time_start_s": torch.tensor(sample["time_start_s"], dtype=torch.float32),
            "duration_s": torch.tensor(sample["duration_s"], dtype=torch.float32),
            "time_fraction": torch.tensor(sample.get("time_fraction", np.linspace(0., 1., len(sample["mask"]))), dtype=torch.float32),
            "vo2peak": torch.tensor(sample["vo2peak"], dtype=torch.float32),
            "vo2peak_mask": torch.tensor(sample.get("vo2peak_mask", 1.0), dtype=torch.float32),
            "trajectory_supervision_mask": torch.tensor(
                sample.get("trajectory_supervision_mask", 1.0), dtype=torch.float32
            ),
            "at_supervision_mask": torch.tensor(
                sample.get("at_supervision_mask", 1.0), dtype=torch.float32
            ),
            "at_metric_mask": torch.tensor(sample.get("at_metric_mask", 1.0), dtype=torch.float32),
            "feature_mean": torch.tensor(sample["feature_mean"], dtype=torch.float32),
            "feature_std": torch.tensor(sample["feature_std"], dtype=torch.float32),
        }


def _eligibility_value(row: pd.Series, column: str, *, default: bool) -> bool:
    if column not in row.index:
        return bool(default)
    value = row[column]
    if pd.isna(value):
        raise ValueError(f"Supervision policy field {column!r} contains a missing value")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if float(value) in {0.0, 1.0}:
            return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Supervision policy field {column!r} has invalid value {value!r}")


def sequence_split_options(config: dict, root: Path, split_name: str, limit: int | None = None) -> dict:
    if config.get("input_format", "registry") == "v5_derivative_split":
        if config.get("normalization_stats_path") not in (None, ""):
            raise ValueError("V5 must use its saved HDF5 train scalers")
        name = {"validation": "val"}.get(split_name, split_name)
        if name not in {"train", "val", "test"}:
            raise ValueError(f"Unknown v5 split: {split_name}")
        return {"input_format": "v5_derivative_split", "split_name": name, "exam_ids": None}
    path = Path(config["split_path"])
    path = path if path.is_absolute() else root / path
    return {"exam_ids": ids_from_split(path, split_name, limit),
            **normalization_options_from_config(config, root)}


def ids_from_split(split_path: Path, split_name: str = "train", limit: int | None = None) -> tuple[str, ...] | None:
    if not split_path.exists():
        return None
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("policy") == "subject-disjoint-random":
        ids = _ids_from_subject_disjoint_policy(split_path, split, split_name)
        if not ids:
            return None
        return tuple(map(str, ids[:limit] if limit else ids))
    ids = split.get(split_name)
    if not ids:
        return None
    if limit:
        ids = ids[:limit]
    return tuple(map(str, ids))


def _resolve_split_policy_path(split_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    foundation_root = split_path.parents[2] if len(split_path.parents) >= 3 else split_path.parent
    rooted = foundation_root / candidate
    if rooted.exists():
        return rooted
    return split_path.parent / candidate


def _stable_fraction(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _exam_subject_keys(source_h5: Path) -> dict[str, str]:
    exam_to_subject: dict[str, str] = {}
    with h5py.File(source_h5, "r") as h5:
        for center in h5["institutes"].keys():
            metadata = pickle.loads(h5["institutes"][center]["metadata"][()].tobytes())
            for _, row in metadata.iterrows():
                exam_id = str(row["Examination_ID"])
                subject_id = row.get("Subject_ID")
                if pd.isna(subject_id):
                    subject_key = f"{center}:missing:{exam_id}"
                else:
                    subject_text = str(subject_id).strip()
                    if subject_text == "" or subject_text.lower() in {"nan", "none", "null"}:
                        subject_key = f"{center}:missing:{exam_id}"
                    else:
                        subject_key = f"{center}:{subject_text}"
                exam_to_subject[exam_id] = subject_key
    return exam_to_subject


def _ids_from_subject_disjoint_policy(split_path: Path, split: dict, split_name: str) -> list[str]:
    source_split_path = _resolve_split_policy_path(split_path, str(split["source_split"]))
    source_h5_path = _resolve_split_policy_path(split_path, str(split["source_h5"]))
    source_split = json.loads(source_split_path.read_text(encoding="utf-8"))
    source_ids = [str(exam_id) for key in ("train", "validation", "test") for exam_id in source_split.get(key, [])]
    exam_to_subject = _exam_subject_keys(source_h5_path)
    groups: dict[str, list[str]] = {}
    for exam_id in source_ids:
        groups.setdefault(exam_to_subject.get(exam_id, f"unknown:{exam_id}"), []).append(exam_id)

    target_counts = {key: int(value) for key, value in split.get("target_counts", {}).items()}
    if not target_counts:
        target_counts = {key: len(source_split.get(key, [])) for key in ("train", "validation", "test")}
    seed = int(split.get("seed", 42))
    ordered_groups = sorted(groups.items(), key=lambda item: _stable_fraction(seed, item[0]))
    split_order = tuple(split.get("split_order", ("validation", "test", "train")))
    partitions = {key: [] for key in ("train", "validation", "test")}
    for _, exam_ids in ordered_groups:
        assigned = False
        for key in split_order:
            if key == "train":
                continue
            if len(partitions[key]) < target_counts.get(key, 0):
                partitions[key].extend(exam_ids)
                assigned = True
                break
        if not assigned:
            partitions["train"].extend(exam_ids)
    return partitions.get(split_name, [])
