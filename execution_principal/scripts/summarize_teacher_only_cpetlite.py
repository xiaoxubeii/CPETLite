#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cpet_foundation.data.distillation_package import audit_distillation_package
from cpet_foundation.model.wearable_student import TEACHER_ONLY_FORBIDDEN_BATCH_KEYS


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/eval/cpetlite_v1_teacher_only_external/comparison.json"
PACKAGE = ROOT / "artifacts/dataset/cpetlite_teacher_only_v1/distillation.h5"
PRIMARY = ("VO2", "VCO2", "VE")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_components(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = np.load(path)
    names = [str(value) for value in data["channel_names"]]
    indices = [names.index(name) for name in PRIMARY]
    prediction = data["trajectory_pred"][:, :, indices].astype(np.float64)
    target = data["trajectory_target"][:, :, indices].astype(np.float64)
    mask = (data["task_mask"][:, :, None] * data["target_channel_mask"][:, :, indices]).astype(np.float64)
    at_pred_s = data["time_start_s"] + data["at_pred_norm"] * data["duration_s"]
    return data["exam_id"].astype(str), {
        "core_sse": (np.square(prediction - target) * mask).sum(axis=(1, 2)),
        "core_n": mask.sum(axis=(1, 2)),
        "at_abs_error_s": np.abs(at_pred_s - data["at_target_s"]).astype(np.float64),
        "at_metric_mask": (
            data["at_metric_mask"].astype(np.float64)
            if "at_metric_mask" in data.files
            else np.ones(len(data["at_target_s"]), dtype=np.float64)
        ),
        "peak_abs_error": np.abs(data["vo2peak_pred"] - data["vo2peak_target"]).astype(np.float64),
        "peak_mask": data["vo2peak_mask"].astype(np.float64),
    }


def _metrics(values: dict[str, np.ndarray], indices: np.ndarray | None = None) -> np.ndarray:
    selected = slice(None) if indices is None else indices
    at_mask = values["at_metric_mask"][selected]
    return np.asarray(
        [
            np.sqrt(values["core_sse"][selected].sum() / values["core_n"][selected].sum()),
            (values["at_abs_error_s"][selected] * at_mask).sum() / at_mask.sum(),
            (values["peak_abs_error"][selected] * values["peak_mask"][selected]).sum()
            / values["peak_mask"][selected].sum(),
        ]
    )


def _paired_difference(
    teacher_only: dict[str, np.ndarray],
    supervised: dict[str, np.ndarray],
    *,
    draws: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(teacher_only["core_sse"])
    point = _metrics(teacher_only) - _metrics(supervised)
    samples = np.empty((draws, 3), dtype=np.float64)
    for draw in range(draws):
        indices = rng.integers(0, n, size=n)
        samples[draw] = _metrics(teacher_only, indices) - _metrics(supervised, indices)
    return {
        name: {
            "teacher_only_minus_direct_supervised": float(point[index]),
            "ci95_low": float(np.quantile(samples[:, index], 0.025)),
            "ci95_high": float(np.quantile(samples[:, index], 0.975)),
            "lower_is_better": True,
        }
        for index, name in enumerate(("primary_trajectory_rmse", "at_time_s_mae", "vo2peak_mae"))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    is_v2 = args.version == "v2"
    package = ROOT / (
        "artifacts/dataset/cpetlite_teacher_only_v2/distillation.h5"
        if is_v2
        else "artifacts/dataset/cpetlite_teacher_only_v1/distillation.h5"
    )
    package_audit = audit_distillation_package(package)
    exposed = sorted(TEACHER_ONLY_FORBIDDEN_BATCH_KEYS.intersection(package_audit["student_visible_fields"]))
    if exposed:
        raise ValueError(f"teacher-only package exposes forbidden fields: {exposed}")
    model_dir = ROOT / (
        "artifacts/models/direct_transformer_teacher_only_cpetlite_v2_seed42"
        if is_v2
        else "artifacts/models/direct_transformer_teacher_only_cpetlite_v1_seed42"
    )
    external_dir = ROOT / (
        "artifacts/eval/cpetlite_v2" if is_v2 else "artifacts/eval/cpetlite_v1_teacher_only_external"
    )
    supervised_dir = ROOT / "artifacts/models/direct_transformer_enhanced_cpetlite_v1_seed42"
    matched_external_dir = ROOT / (
        "artifacts/eval/cpetlite_v2" if is_v2 else "artifacts/eval/cpetlite_v1_matched_external"
    )
    paths = {
        "internal": {
            "teacher_only": (
                external_dir / "teacher_only_direct_transformer_internal_predictions.npz"
                if is_v2
                else model_dir / "secure_internal_predictions.npz"
            ),
            "supervised": (
                matched_external_dir / "direct_transformer_seed42_internal_predictions.npz"
                if is_v2
                else supervised_dir / "validation_predictions.npz"
            ),
        },
        "external": {
            "teacher_only": external_dir / (
                "teacher_only_direct_transformer_liuyuan_predictions.npz"
                if is_v2
                else "direct_transformer_predictions.npz"
            ),
            "supervised": matched_external_dir / (
                "direct_transformer_seed42_liuyuan_predictions.npz"
                if is_v2
                else "direct_transformer_predictions.npz"
            ),
        },
    }
    paired: dict[str, object] = {}
    for domain, domain_paths in paths.items():
        teacher_ids, teacher_components = _case_components(domain_paths["teacher_only"])
        supervised_ids, supervised_components = _case_components(domain_paths["supervised"])
        if not np.array_equal(teacher_ids, supervised_ids):
            raise ValueError(f"{domain} examination order differs")
        paired[domain] = _paired_difference(teacher_components, supervised_components)
    manifest = _read_json(model_dir / "manifest.json")
    report = {
        "comparison_contract": {
            "internal_validation_n": 1088 if is_v2 else 1740,
            "external_liuyuan_n": 2224,
            "internal_at_evaluable_n": 860 if is_v2 else 1740,
            "external_at_evaluable_n": 2224,
            "bootstrap_draws": 2000,
            "bootstrap_unit": "examination",
            "difference_orientation": "teacher_only minus direct_supervised; lower is better",
            "model_architecture": "direct_transformer",
            "seed": 42,
        },
        "data_isolation": {
            "supervision_mode": "teacher_only",
            "package_sha256": _sha256(package),
            "package_schema_version": package_audit["schema_version"],
            "package_splits": package_audit["splits"],
            "student_visible_fields": package_audit["student_visible_fields"],
            "forbidden_fields_present": exposed,
            "contains_patient_identifiers": package_audit["metadata"].get("contains_patient_identifiers"),
            "contains_measured_full_cpet_targets": package_audit["metadata"].get(
                "contains_measured_full_cpet_targets"
            ),
            "raw_cpet_opened_by_student_trainer": manifest["raw_cpet_opened_by_student_trainer"],
            "evidence_boundary": manifest["evidence_boundary"],
        },
        "teacher_only": {
            "checkpoint": str(model_dir / "model.pt"),
            "checkpoint_sha256": _sha256(model_dir / "model.pt"),
            "training_control": manifest["training_control"],
            "teacher_agreement": _read_json(model_dir / "metrics.json"),
            "internal": {
                key: _read_json(
                    external_dir / "teacher_only_direct_transformer_internal_metrics.json"
                    if is_v2
                    else model_dir / "secure_internal_metrics.json"
                )[key]
                for key in ("primary_trajectory_rmse", "reconstruction_trajectory_rmse", "at_time_s_mae", "vo2peak_mae")
            },
            "external": {
                key: _read_json(
                    external_dir / "teacher_only_direct_transformer_liuyuan_metrics.json"
                    if is_v2
                    else external_dir / "direct_transformer_metrics.json"
                )[key]
                for key in ("primary_trajectory_rmse", "reconstruction_trajectory_rmse", "at_time_s_mae", "vo2peak_mae")
            },
        },
        "direct_supervised": _read_json(
            matched_external_dir / ("matched_comparison.json" if is_v2 else "comparison.json")
        )["models"]["direct_transformer"],
        "paired_bootstrap": paired,
    }
    output = args.output.resolve() if args.output else (
        external_dir / ("teacher_only_comparison.json" if is_v2 else "comparison.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
