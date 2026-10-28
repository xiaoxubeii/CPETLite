#!/usr/bin/env python3
"""Audit the complete CPETLite v2 experiment grid and write a provenance report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_records(path: Path) -> list[dict[str, object]]:
    matrix = yaml.safe_load(path.read_text(encoding="utf-8"))
    existing = matrix.get("existing_outputs", {})
    records: list[dict[str, object]] = []
    for stage in matrix["stages"]:
        for seed in matrix["seeds"]:
            reused = existing.get(stage["id"], {}).get(seed, existing.get(stage["id"], {}).get(str(seed)))
            output = reused or matrix["output_dir_template"].format(stage=stage["id"], seed=seed)
            records.append(
                {
                    "group": matrix["experiment_id"],
                    "stage": stage["id"],
                    "seed": seed,
                    "model_dir": output,
                    "evaluation_dir": matrix["evaluation_dir"],
                    "reused": bool(reused),
                }
            )
    return records


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.is_file() and path.stat().st_size > 0,
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": _sha256(path) if path.is_file() and path.stat().st_size > 0 else None,
    }


def _prediction_contract(path: Path, *, expected_n: int, expected_at_n: int) -> list[str]:
    """Return provenance violations for a case-level prediction artifact."""
    issues: list[str] = []
    try:
        data = np.load(path)
        missing = {"exam_id", "at_metric_mask"}.difference(data.files)
        if missing:
            return [f"{path}: missing prediction fields {sorted(missing)}"]
        exam_ids = data["exam_id"].astype(str)
        at_mask = data["at_metric_mask"].astype(np.float64)
        if len(exam_ids) != expected_n:
            issues.append(f"{path}: examination count {len(exam_ids)} != {expected_n}")
        if len(set(exam_ids.tolist())) != len(exam_ids):
            issues.append(f"{path}: duplicate examination identifiers")
        if at_mask.shape != (len(exam_ids),):
            issues.append(f"{path}: AT metric mask shape {at_mask.shape} != {(len(exam_ids),)}")
        elif not np.isin(at_mask, (0.0, 1.0)).all():
            issues.append(f"{path}: AT metric mask is not binary")
        elif int(at_mask.sum()) != expected_at_n:
            issues.append(f"{path}: AT evaluable count {int(at_mask.sum())} != {expected_at_n}")
    except Exception as error:  # pragma: no cover - exercised by the strict artifact audit
        issues.append(f"{path}: cannot validate predictions ({error})")
    return issues


def _comparison_contract(path: Path, *, matrix: bool = False) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        contract = payload["contract" if matrix else "comparison_contract"]
        if matrix:
            observed = (
                int(contract["internal_at_evaluable_n"]),
                int(contract["external_at_evaluable_n"]),
            )
        else:
            counts = contract.get("at_evaluable_n")
            observed = (
                int(counts["internal"] if counts else contract["internal_at_evaluable_n"]),
                int(counts["external"] if counts else contract["external_at_evaluable_n"]),
            )
        issues = [] if observed == (860, 2224) else [f"{path}: AT denominators {observed} != (860, 2224)"]

        def visit(value: object, location: str) -> None:
            if isinstance(value, float) and not math.isfinite(value):
                issues.append(f"{path}: non-finite value at {location}")
            elif isinstance(value, dict):
                for key, child in value.items():
                    visit(child, f"{location}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, f"{location}[{index}]")

        visit(payload, "root")
        return issues
    except Exception as error:  # pragma: no cover - exercised by the strict artifact audit
        return [f"{path}: cannot validate comparison contract ({error})"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/eval/cpetlite_v2/experiment_audit.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    primary_models = [
        "artifacts/models/protocol_sde_cpetlite_v2_matched_seed42",
        "artifacts/models/wearable_student_core_base_cpetlite_v2",
        "artifacts/models/wearable_student_core_enhanced_cpetlite_v2",
        "artifacts/models/wearable_student_enhanced_cpetlite_v2_matched_seed42",
        "artifacts/models/wearable_student_enhanced_cpetlite_v2_no_distillation_seed42",
        "artifacts/models/direct_transformer_enhanced_cpetlite_v2_seed42",
        "artifacts/models/direct_tcn_enhanced_cpetlite_v2_seed42",
        "artifacts/models/wearable_student_signal_only_cpetlite_v2_seed42",
        "artifacts/models/wearable_student_high_noise_cpetlite_v2_seed42",
        "artifacts/models/wearable_student_prefix_cpetlite_v2_seed42",
        "artifacts/models/wearable_student_teacher_only_cpetlite_v2_seed42",
        "artifacts/models/direct_transformer_teacher_only_cpetlite_v2_seed42",
    ]
    matrix_paths = [
        ROOT / "configs/distillation_component_ablation_cpetlite_v2.yaml",
        ROOT / "configs/student_dynamics_comparison_cpetlite_v2.yaml",
        ROOT / "configs/gru_distillation_comparison_cpetlite_v2.yaml",
        ROOT / "configs/selective_distillation_comparison_cpetlite_v2.yaml",
    ]
    matrix_records = [record for path in matrix_paths for record in _matrix_records(path)]
    unique_model_dirs = sorted(set(primary_models + [str(record["model_dir"]) for record in matrix_records]))
    model_records: list[dict[str, object]] = []
    for relative in unique_model_dirs:
        directory = ROOT / relative
        files = [_file_record(directory / name) for name in ("model.pt", "metrics.json", "manifest.json")]
        model_records.append({"model_dir": relative, "complete": all(row["exists"] for row in files), "files": files})

    matrix_evaluations: list[dict[str, object]] = []
    for record in matrix_records:
        prefix = ROOT / str(record["evaluation_dir"]) / "external" / f"{record['stage']}_seed{record['seed']}"
        files = [_file_record(prefix.with_name(prefix.name + suffix)) for suffix in ("_metrics.json", "_manifest.json", "_predictions.npz")]
        matrix_evaluations.append({**record, "complete": all(row["exists"] for row in files), "files": files})

    required_evaluations = [
        "core_base_seed42_liuyuan_metrics.json",
        "core_enhanced_seed42_liuyuan_metrics.json",
        "matched_distilled_seed42_liuyuan_metrics.json",
        "matched_measured_only_seed42_liuyuan_metrics.json",
        "direct_transformer_seed42_liuyuan_metrics.json",
        "direct_tcn_seed42_liuyuan_metrics.json",
        "signal_only_seed42_liuyuan_metrics.json",
        "high_noise_seed42_liuyuan_metrics.json",
        "prefix_seed42_liuyuan_metrics.json",
        "teacher_only_neural_sde_internal_metrics.json",
        "teacher_only_neural_sde_liuyuan_metrics.json",
        "teacher_only_direct_transformer_internal_metrics.json",
        "teacher_only_direct_transformer_liuyuan_metrics.json",
        "primary_enhanced_internal_signal_masked_metrics.json",
        "primary_enhanced_internal_signal_shuffled_metrics.json",
        "primary_enhanced_external_signal_masked_metrics.json",
        "primary_enhanced_external_signal_shuffled_metrics.json",
        "endpoint_baselines_internal.json",
        "endpoint_baselines_external_liuyuan.json",
        "teacher_liuyuan_metrics.json",
        "matched_comparison.json",
        "base_enhanced_comparison.json",
        "teacher_only_comparison.json",
    ]
    evaluation_records = [_file_record(ROOT / "artifacts/eval/cpetlite_v2" / name) for name in required_evaluations]
    primary_prediction_names = [
        (f"{name}_{split}_predictions.npz", split)
        for name in (
            "core_base_seed42",
            "core_enhanced_seed42",
            "matched_distilled_seed42",
            "matched_measured_only_seed42",
            "direct_transformer_seed42",
            "direct_tcn_seed42",
        )
        for split in ("internal", "liuyuan")
    ]
    evaluation_records.extend(
        _file_record(ROOT / "artifacts/eval/cpetlite_v2" / name) for name, _ in primary_prediction_names
    )
    teacher_only_prediction_names = [
        (f"teacher_only_{architecture}_{split}_predictions.npz", split)
        for architecture in ("neural_sde", "direct_transformer")
        for split in ("internal", "liuyuan")
    ]
    evaluation_records.extend(
        _file_record(ROOT / "artifacts/eval/cpetlite_v2" / name)
        for name, _ in teacher_only_prediction_names
    )
    evaluation_records.extend(
        _file_record(ROOT / directory / "comparison.json")
        for directory in (
            "artifacts/eval/cpetlite_v2_distillation_component_ablation",
            "artifacts/eval/cpetlite_v2_student_dynamics_comparison",
            "artifacts/eval/cpetlite_v2_gru_distillation_comparison",
            "artifacts/eval/cpetlite_v2_selective_distillation_comparison",
        )
    )
    evaluation_records.append(
        _file_record(
            ROOT
            / "artifacts/eval/cpetlite_v2_student_dynamics_comparison/gru_vs_direct_paired.json"
        )
    )
    evaluation_records.append(
        _file_record(
            ROOT
            / "artifacts/eval/cpetlite_v2_student_dynamics_comparison/inference_benchmark.json"
        )
    )
    evaluation_records.append(
        _file_record(
            ROOT
            / "artifacts/eval/cpetlite_v2_selective_distillation_comparison/inference_benchmark.json"
        )
    )
    quality_issues: list[str] = []
    for name, split in primary_prediction_names + teacher_only_prediction_names:
        path = ROOT / "artifacts/eval/cpetlite_v2" / name
        if path.is_file():
            quality_issues.extend(
                _prediction_contract(
                    path,
                    expected_n=1088 if split == "internal" else 2224,
                    expected_at_n=860 if split == "internal" else 2224,
                )
            )
    for row in matrix_evaluations:
        prediction = next(Path(str(item["path"])) for item in row["files"] if str(item["path"]).endswith("_predictions.npz"))
        path = ROOT / prediction
        if path.is_file():
            quality_issues.extend(_prediction_contract(path, expected_n=2224, expected_at_n=2224))

    comparison_paths = [
        (ROOT / "artifacts/eval/cpetlite_v2/matched_comparison.json", False),
        (ROOT / "artifacts/eval/cpetlite_v2/base_enhanced_comparison.json", False),
        (ROOT / "artifacts/eval/cpetlite_v2/teacher_only_comparison.json", False),
        (ROOT / "artifacts/eval/cpetlite_v2_distillation_component_ablation/comparison.json", True),
        (ROOT / "artifacts/eval/cpetlite_v2_student_dynamics_comparison/comparison.json", True),
        (ROOT / "artifacts/eval/cpetlite_v2_gru_distillation_comparison/comparison.json", True),
        (ROOT / "artifacts/eval/cpetlite_v2_selective_distillation_comparison/comparison.json", True),
    ]
    for path, is_matrix in comparison_paths:
        if path.is_file():
            quality_issues.extend(_comparison_contract(path, matrix=is_matrix))

    missing = [row["model_dir"] for row in model_records if not row["complete"]]
    missing += [f"{row['group']}:{row['stage']}:seed{row['seed']} external" for row in matrix_evaluations if not row["complete"]]
    missing += [str(row["path"]) for row in evaluation_records if not row["exists"]]
    payload = {
        "schema_version": 1,
        "complete": not missing and not quality_issues,
        "unique_model_count": len(model_records),
        "matrix_cell_count": len(matrix_records),
        "matrix_external_evaluation_count": len(matrix_evaluations),
        "models": model_records,
        "matrix_evaluations": matrix_evaluations,
        "required_evaluations": evaluation_records,
        "missing": missing,
        "quality_issues": quality_issues,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": not missing and not quality_issues, "missing_count": len(missing), "quality_issue_count": len(quality_issues), "output": str(output)}, indent=2))
    if args.strict and (missing or quality_issues):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
