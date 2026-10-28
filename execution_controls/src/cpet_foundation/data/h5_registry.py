from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


DEFAULT_REQUIRED_FEATURES = [
    "Time",
    "Load_Phase",
    "VO2",
    "VCO2",
    "VE",
    "RER",
    "HR",
    "VO2_HR",
    "SpO2",
    "PetO2",
    "PetCO2",
    "Power_Load",
    "RPM",
]

REQUIRED_METADATA = [
    "Examination_ID",
    "Subject_ID",
    "Gender",
    "Age",
    "Height_cm",
    "Weight_kg",
    "Source_Device",
    "Institute_Name",
    "Raw_Phase0_End_Time",
    "Raw_Phase1_Start_Time",
    "Raw_Phase1_End_Time",
    "Raw_Phase2_Start_Time",
]


@dataclass(frozen=True)
class RegistryConfig:
    source_h5: Path
    output_dir: Path
    split_dir: Path
    required_features: tuple[str, ...] = tuple(DEFAULT_REQUIRED_FEATURES)
    target_column: str = "Time_at_AT"
    seed: int = 42


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickled_dataset(group: h5py.Group, dataset_name: str) -> pd.DataFrame:
    raw = bytes(group[dataset_name][:].tolist())
    value = pickle.loads(raw)
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{group.name}/{dataset_name} is {type(value)!r}, expected pandas.DataFrame")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def validate_source_h5(
    source_h5: Path,
    required_features: tuple[str, ...] = tuple(DEFAULT_REQUIRED_FEATURES),
    target_column: str = "Time_at_AT",
) -> dict[str, Any]:
    errors: list[str] = []
    institutes: dict[str, dict[str, Any]] = {}
    with h5py.File(source_h5, "r") as h5:
        if "institutes" not in h5:
            return {"valid": False, "errors": ["missing root group: institutes"], "institutes": {}}
        for institute_name in h5["institutes"].keys():
            group = h5["institutes"][institute_name]
            for name in ("exam_ids", "exam_row_counts", "features", "metadata", "targets"):
                if name not in group:
                    errors.append(f"{institute_name}: missing {name}")
            if "exam_row_counts" in group:
                if "counts" not in group["exam_row_counts"] or "ids" not in group["exam_row_counts"]:
                    errors.append(f"{institute_name}: exam_row_counts requires ids and counts")
            try:
                features = load_pickled_dataset(group, "features")
                metadata = load_pickled_dataset(group, "metadata")
                targets = load_pickled_dataset(group["targets"], "raw")
            except Exception as exc:
                errors.append(f"{institute_name}: unable to load pickled tables: {exc}")
                continue

            missing_features = [col for col in required_features if col not in features.columns]
            missing_metadata = [col for col in REQUIRED_METADATA if col not in metadata.columns]
            missing_targets = ["Examination_ID"] if "Examination_ID" not in targets.columns else []
            if target_column not in targets.columns:
                missing_targets.append(target_column)
            if missing_features:
                errors.append(f"{institute_name}: missing features {missing_features}")
            if missing_metadata:
                errors.append(f"{institute_name}: missing metadata {missing_metadata}")
            if missing_targets:
                errors.append(f"{institute_name}: missing targets {missing_targets}")

            counts = group["exam_row_counts"]["counts"][:] if "exam_row_counts" in group else np.array([])
            institutes[institute_name] = {
                "exam_count": int(len(group["exam_ids"])) if "exam_ids" in group else int(features["Examination_ID"].nunique()),
                "feature_rows": int(len(features)),
                "target_rows": int(len(targets)),
                "metadata_rows": int(len(metadata)),
                "row_count_min": int(counts.min()) if len(counts) else None,
                "row_count_median": float(np.median(counts)) if len(counts) else None,
                "row_count_max": int(counts.max()) if len(counts) else None,
                "feature_columns": list(map(str, features.columns)),
                "metadata_columns": list(map(str, metadata.columns)),
                "target_columns": list(map(str, targets.columns)),
            }
    return {"valid": not errors, "errors": errors, "institutes": institutes}


def build_manifest(config: RegistryConfig) -> dict[str, Any]:
    source_h5 = config.source_h5
    validation = validate_source_h5(source_h5, config.required_features, config.target_column)
    total_exams = sum(item["exam_count"] for item in validation["institutes"].values())
    total_rows = sum(item["feature_rows"] for item in validation["institutes"].values())
    manifest = {
        "format": "cpet_world_registry",
        "version": 1,
        "source_h5": str(source_h5),
        "source_sha256": file_sha256(source_h5),
        "schema_valid": validation["valid"],
        "schema_errors": validation["errors"],
        "target_column": config.target_column,
        "required_features": list(config.required_features),
        "total_exams": int(total_exams),
        "total_rows": int(total_rows),
        "institutes": validation["institutes"],
    }
    return manifest


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_safe) + "\n", encoding="utf-8")


def register_dataset(config: RegistryConfig) -> dict[str, Any]:
    manifest = build_manifest(config)
    write_json(config.output_dir / "manifest.json", manifest)
    return manifest


def manifest_is_stale(manifest_path: Path) -> bool:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_h5 = Path(manifest["source_h5"])
    if not source_h5.exists():
        return True
    return file_sha256(source_h5) != manifest.get("source_sha256")


def load_all_exam_records(source_h5: Path, target_column: str = "Time_at_AT") -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    with h5py.File(source_h5, "r") as h5:
        for institute_name in h5["institutes"].keys():
            group = h5["institutes"][institute_name]
            metadata = load_pickled_dataset(group, "metadata")
            targets = load_pickled_dataset(group["targets"], "raw")
            cols = ["Examination_ID", "Age", "Height_cm", "Weight_kg", "Gender", "Source_Device", "Institute_Name"]
            available_cols = [col for col in cols if col in metadata.columns]
            merged = metadata[available_cols].drop_duplicates("Examination_ID").merge(
                targets[["Examination_ID", target_column]].drop_duplicates("Examination_ID"),
                on="Examination_ID",
                how="inner",
            )
            merged["registry_institute"] = institute_name
            records.append(merged)
    return pd.concat(records, ignore_index=True)


def generate_splits(source_h5: Path, output_dir: Path, seed: int = 42, target_column: str = "Time_at_AT") -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    records = load_all_exam_records(source_h5, target_column)
    exam_ids = records["Examination_ID"].astype(str).to_numpy()
    shuffled = exam_ids.copy()
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    random_split = {
        "policy": "random",
        "seed": seed,
        "train": shuffled[:n_train].tolist(),
        "validation": shuffled[n_train : n_train + n_val].tolist(),
        "test": shuffled[n_train + n_val :].tolist(),
    }
    write_json(output_dir / f"split_random_seed{seed}.json", random_split)

    loco: dict[str, Any] = {"policy": "leave-one-center-out", "seed": seed, "folds": {}}
    for institute_name in sorted(records["registry_institute"].unique()):
        test_ids = records.loc[records["registry_institute"] == institute_name, "Examination_ID"].astype(str).tolist()
        train_ids = records.loc[records["registry_institute"] != institute_name, "Examination_ID"].astype(str).tolist()
        loco["folds"][institute_name] = {"train": train_ids, "test": test_ids}
    write_json(output_dir / f"split_loco_seed{seed}.json", loco)
    return {"random": random_split, "loco": loco}


def summarize_dataset(source_h5: Path, output_dir: Path, target_column: str = "Time_at_AT") -> dict[str, Any]:
    records = load_all_exam_records(source_h5, target_column)
    summary = {
        "exam_count": int(len(records)),
        "institutes": records["registry_institute"].value_counts().sort_index().to_dict(),
        "target_nonnull": int(records[target_column].notna().sum()),
        "age": records["Age"].describe().to_dict() if "Age" in records else {},
        "height_cm": records["Height_cm"].describe().to_dict() if "Height_cm" in records else {},
        "weight_kg": records["Weight_kg"].describe().to_dict() if "Weight_kg" in records else {},
        "gender": records["Gender"].astype(str).value_counts(dropna=False).to_dict() if "Gender" in records else {},
        "source_device": records["Source_Device"].astype(str).value_counts(dropna=False).head(20).to_dict()
        if "Source_Device" in records
        else {},
    }
    write_json(output_dir / "summary.json", summary)
    return summary
