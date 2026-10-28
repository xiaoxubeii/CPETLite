#!/usr/bin/env python3
"""Re-evaluate a wearable-student checkpoint with task-specific metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cpet_foundation.data.sequences import (
    CpetSequenceDataset,
    SequenceConfig,
    ids_from_split,
    normalization_options_from_config,
    sequence_split_options,
)
from cpet_foundation.inference.digital_twin import load_checkpoint
from cpet_foundation.inference.wearable_twin import load_wearable_checkpoint
from cpet_foundation.training.runner import (
    evaluate_wearable_student,
    load_yaml,
    resolve_wearable_task_mode,
    shape_feature_indices_from_config,
    set_seed,
    wearable_simulation_config_from_training_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help="Secure evaluation config supplying raw-data paths absent from a teacher-only checkpoint",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split-path", type=Path)
    parser.add_argument("--split-name", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--center")
    parser.add_argument("--evaluation-name")
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument(
        "--input-perturbation",
        default="none",
        choices=("none", "signal_masked", "signal_shuffled"),
    )
    parser.add_argument("--perturbation-seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    student, checkpoint = load_wearable_checkpoint(checkpoint_path)
    config = dict(checkpoint["config"])
    if args.config is not None:
        evaluation_config = load_yaml(args.config.resolve())
        config.update(evaluation_config)
    evaluation_seed = int(config.get("seed", 42))
    set_seed(evaluation_seed)
    source_h5 = _resolve(repo_root, config["source_h5"])
    is_v5 = config.get("input_format") == "v5_derivative_split"
    if is_v5 and args.split_path:
        raise ValueError("V5 evaluation uses HDF5-owned partitions, not --split-path")
    split_path = source_h5 if is_v5 else (args.split_path.resolve() if args.split_path else _resolve(repo_root, config["split_path"]))
    if args.split_path:
        config["split_path"] = str(split_path)
    input_options = sequence_split_options(config, repo_root, args.split_name, args.limit)
    dataset = CpetSequenceDataset(
        SequenceConfig(
            source_h5=source_h5,
            feature_columns=tuple(config["feature_columns"]),
            metadata_columns=tuple(config["metadata_columns"]),
            max_len=int(config.get("max_len", 160)),
            limit_exams=args.limit,
            **input_options,
        )
    )
    evaluation_ids = tuple(sample["exam_id"] for sample in dataset.samples)
    if not evaluation_ids:
        raise ValueError("Empty evaluation population")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = student.to(device)
    configured_teacher = args.teacher_checkpoint or config.get("teacher_checkpoint")
    if configured_teacher is None:
        raise ValueError("Evaluation requires --teacher-checkpoint when the student config has no teacher checkpoint")
    teacher_path = _resolve(repo_root, configured_teacher)
    teacher, teacher_checkpoint = load_checkpoint(teacher_path)
    teacher = teacher.to(device)
    config["teacher_metadata_columns"] = list(
        teacher_checkpoint.get("config", {}).get("metadata_columns", config.get("metadata_columns", ()))
    )
    target_names = tuple(config.get("target_feature_columns", config["feature_columns"]))
    source_index = {name: index for index, name in enumerate(config["feature_columns"])}
    target_indices = tuple(source_index[name] for name in target_names)
    output_path = args.output.resolve() if args.output else checkpoint_path.with_name("metrics.json")
    predictions_output_path = args.predictions_output.resolve() if args.predictions_output else None
    metrics = evaluate_wearable_student(
        student,
        teacher,
        DataLoader(dataset, batch_size=int(config.get("batch_size", 64)), pin_memory=device.type == "cuda"),
        tuple(config["feature_columns"]),
        target_indices,
        shape_feature_indices_from_config(config, target_names),
        float(config.get("shape_loss_prefix_fraction", 0.6)),
        wearable_simulation_config_from_training_config(config),
        device,
        task_mode=resolve_wearable_task_mode(config),
        online_prefix_min_fraction=float(config.get("online_prefix_min_fraction", 0.35)),
        online_prefix_max_fraction=float(config.get("online_prefix_max_fraction", 0.75)),
        online_action_key=str(config.get("online_action_key", "continue_ramp")),
        training_config=config,
        prediction_output_path=predictions_output_path,
        input_perturbation=args.input_perturbation,
        perturbation_seed=args.perturbation_seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    manifest_path = (
        args.manifest_output.resolve()
        if args.manifest_output
        else output_path.with_name(f"{output_path.stem}_manifest.json")
    )
    normalization_path = None if is_v5 else input_options["normalization_stats_path"]
    normalization_payload = (
        json.loads(normalization_path.read_text(encoding="utf-8")) if normalization_path is not None else {}
    )
    manifest = {
        "schema_version": 1,
        "evaluation_name": args.evaluation_name or f"{args.center or 'checkpoint'}_{args.split_name}",
        "evaluation_type": "external-centre" if args.center else "checkpoint-split",
        "center": args.center,
        "split_name": args.split_name,
        "split_path": str(split_path),
        "split_sha256": _sha256(split_path),
        "requested_examinations": len(evaluation_ids),
        "loaded_examinations": len(dataset),
        "evaluation_seed": evaluation_seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "normalization_stats_path": str(normalization_path) if normalization_path is not None else None,
        "normalization_stats_sha256": _sha256(normalization_path) if normalization_path is not None else None,
        "normalization_fit_population": "internal_training_hdf5" if is_v5 else normalization_payload.get("fit_population"),
        "input_format": config.get("input_format", "registry"),
        "source_h5_sha256": _sha256(source_h5),
        "metrics_path": str(output_path),
        "predictions_path": str(predictions_output_path) if predictions_output_path is not None else None,
        "input_boundary": config.get("input_boundary", "legacy"),
        "teacher_checkpoint": str(teacher_path),
        "task_mode": resolve_wearable_task_mode(config),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "output": str(output_path),
                "manifest": str(manifest_path),
                "predictions": str(predictions_output_path) if predictions_output_path is not None else None,
                "split": args.split_name,
                "samples": len(dataset),
                "metrics": metrics,
            },
            indent=2,
        )
    )


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
