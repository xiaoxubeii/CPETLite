from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader

from cpet_foundation.data.distillation_package import write_distillation_package
from cpet_foundation.data.matched_inputs import input_contract
from cpet_foundation.data.sequences import (
    CpetSequenceDataset,
    SequenceConfig,
    ids_from_split,
    normalization_options_from_config,
    sequence_split_options,
)
from cpet_foundation.model.wearable_student import DERIVED_CPET_FEATURES
from cpet_foundation.training.runner import (
    _deterministic_base_feature_columns,
    _device,
    _optional_int,
    _resolve_config_path,
    _target_feature_indices,
    _training_metadata_scaler,
    _training_output_scaler,
    build_wearable_student_batch,
    load_yaml,
    resolve_wearable_task_mode,
    shape_feature_indices_from_config,
    wearable_simulation_config_from_training_config,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_teacher_only_distillation_package(
    config_path: Path,
    output_path: Path,
    *,
    limit_exams: int | None = None,
) -> dict[str, object]:
    from cpet_foundation.inference.digital_twin import load_checkpoint

    config_path = config_path.resolve()
    config = load_yaml(config_path)
    root = config_path.parents[1]
    source_h5 = _resolve_config_path(root, str(config["source_h5"]))
    is_v5 = config.get("input_format") == "v5_derivative_split"
    split_path = source_h5 if is_v5 else _resolve_config_path(root, str(config["split_path"]))
    stats_path = source_h5 if is_v5 else _resolve_config_path(root, str(config["normalization_stats_path"]))
    teacher_path = _resolve_config_path(root, str(config["teacher_checkpoint"]))
    configured_limit = _optional_int(config.get("limit_exams", None))
    effective_limit = limit_exams if limit_exams is not None else configured_limit
    validation_limit = None if effective_limit is None else max(1, effective_limit // 5)

    datasets = {
        "train": CpetSequenceDataset(
            SequenceConfig(
                source_h5=source_h5,
                feature_columns=tuple(config["feature_columns"]),
                metadata_columns=tuple(config["metadata_columns"]),
                max_len=int(config.get("max_len", 160)),
                limit_exams=effective_limit,
                **sequence_split_options(config, root, "train", effective_limit),
            )
        ),
        "validation": CpetSequenceDataset(
            SequenceConfig(
                source_h5=source_h5,
                feature_columns=tuple(config["feature_columns"]),
                metadata_columns=tuple(config["metadata_columns"]),
                max_len=int(config.get("max_len", 160)),
                limit_exams=validation_limit,
                **sequence_split_options(config, root, "validation", validation_limit),
            )
        ),
    }
    device = _device()
    teacher_model, teacher_checkpoint = load_checkpoint(teacher_path)
    teacher_model = teacher_model.to(device).eval()
    config["teacher_metadata_columns"] = list(
        teacher_checkpoint.get("config", {}).get("metadata_columns", config.get("metadata_columns", ()))
    )

    target_feature_columns = tuple(config.get("target_feature_columns", config["feature_columns"]))
    feature_index = {name: index for index, name in enumerate(config["feature_columns"])}
    target_indices = tuple(feature_index[name] for name in target_feature_columns)
    shape_feature_indices = shape_feature_indices_from_config(config, target_feature_columns)
    direct_loss_feature_indices = _target_feature_indices(
        config, "direct_loss_feature_columns", target_feature_columns, ("VO2", "VO2_kg", "VCO2", "VE")
    )
    core_derived_feature_indices = _target_feature_indices(
        config, "core_derived_feature_columns", target_feature_columns, ()
    )
    peak_loss_feature_indices = _target_feature_indices(
        config,
        "peak_loss_feature_columns",
        target_feature_columns,
        tuple(config.get("shape_loss_feature_columns", ("VO2", "VO2_kg", "VCO2", "VE"))),
    )
    derived_feature_indices = _target_feature_indices(
        config,
        "derived_consistency_feature_columns",
        target_feature_columns,
        ("VO2", "VO2_kg", "VCO2", "VCO2_kg", "VE", "RER", "VE_VO2", "VE_VCO2", "METS", "HR", "VO2_HR"),
    )
    metadata_columns = tuple(config["metadata_columns"])
    body_mass_metadata_index = metadata_columns.index("Weight_kg") if "Weight_kg" in metadata_columns else None
    deterministic_outputs = bool(config.get("deterministic_derived_channels", False))
    base_feature_columns = (
        _deterministic_base_feature_columns(config) if deterministic_outputs else target_feature_columns
    )
    output_feature_mean, output_feature_std = _training_output_scaler(
        datasets["train"],
        tuple(config["feature_columns"]),
        target_feature_columns,
        body_mass_metadata_index,
        config,
    )
    metadata_feature_mean, metadata_feature_std = _training_metadata_scaler(
        datasets["train"], metadata_columns
    )
    config["target_feature_names"] = list(target_feature_columns)
    config["base_feature_names"] = list(base_feature_columns)
    config["output_feature_mean"] = output_feature_mean.tolist()
    config["output_feature_std"] = output_feature_std.tolist()
    config["metadata_feature_mean"] = metadata_feature_mean.tolist()
    config["metadata_feature_std"] = metadata_feature_std.tolist()
    distillation_feature_indices = tuple(target_feature_columns.index(name) for name in base_feature_columns)
    distillation_relation_feature_indices = _target_feature_indices(
        config,
        "distillation_relation_feature_columns",
        target_feature_columns,
        ("VO2", "VCO2", "VE"),
    )
    derived_loss_feature_indices = tuple(target_feature_columns.index(name) for name in DERIVED_CPET_FEATURES)
    shape_loss_prefix_fraction = float(config.get("shape_loss_prefix_fraction", 0.6))
    task_mode = resolve_wearable_task_mode(config)
    wearable_config = wearable_simulation_config_from_training_config(config)

    def batches_for(split_name: str) -> Iterator[dict[str, torch.Tensor]]:
        loader = DataLoader(
            datasets[split_name],
            batch_size=int(config.get("batch_size", 64)),
            shuffle=False,
            num_workers=int(config.get("num_workers", 0)),
            pin_memory=device.type == "cuda",
        )
        for raw_batch in loader:
            raw_batch = {name: value.to(device) for name, value in raw_batch.items()}
            student_batch = build_wearable_student_batch(
                raw_batch,
                feature_columns=tuple(config["feature_columns"]),
                target_indices=target_indices,
                shape_feature_indices=shape_feature_indices,
                direct_loss_feature_indices=direct_loss_feature_indices,
                core_derived_feature_indices=core_derived_feature_indices,
                slope_loss_feature_indices=shape_feature_indices,
                peak_loss_feature_indices=peak_loss_feature_indices,
                derived_feature_indices=derived_feature_indices,
                derived_loss_feature_indices=derived_loss_feature_indices,
                distillation_feature_indices=distillation_feature_indices,
                distillation_relation_feature_indices=distillation_relation_feature_indices,
                body_mass_metadata_index=body_mass_metadata_index,
                target_feature_columns=target_feature_columns,
                output_feature_mean=output_feature_mean.to(device),
                output_feature_std=output_feature_std.to(device),
                deterministic_config=config,
                shape_loss_prefix_fraction=shape_loss_prefix_fraction,
                simulation_config=wearable_config,
                teacher_model=teacher_model,
                device=device,
                task_mode=task_mode,
                online_safe_training=task_mode == "prefix_forecast",
                online_prefix_min_fraction=float(
                    config.get("online_prefix_min_fraction", shape_loss_prefix_fraction)
                ),
                online_prefix_max_fraction=float(
                    config.get("online_prefix_max_fraction", shape_loss_prefix_fraction)
                ),
                online_action_key=str(config.get("online_action_key", "continue_ramp")),
                supervision_mode="teacher_only",
            )
            yield {name: value.detach().cpu() for name, value in student_batch.items()}

    model_contract_keys = (
        "feature_columns",
        "target_feature_columns",
        "metadata_columns",
        "watch_channels",
        "protocol_context_channels",
        "target_feature_names",
        "base_feature_names",
        "output_feature_mean",
        "output_feature_std",
        "metadata_feature_mean",
        "metadata_feature_std",
        "deterministic_derived_channels",
        "joint_gas_exchange_decoder",
        "d_model",
        "n_heads",
        "n_layers",
        "dropout",
        "max_len",
        "wearable_max_len",
    )
    metadata = {
        "source_h5_sha256": _sha256(source_h5),
        "split_sha256": _sha256(split_path),
        "normalization_stats_sha256": _sha256(stats_path),
        "teacher_checkpoint_sha256": _sha256(teacher_path),
        "source_config_sha256": _sha256(config_path),
        "input_format": config.get("input_format", "registry"),
        "matched_input_contract": input_contract(config),
        "model_contract": {key: config[key] for key in model_contract_keys if key in config},
        "contains_patient_identifiers": False,
        "contains_measured_full_cpet_targets": False,
        "evidence_boundary": "data-minimized clinical-derived teacher package; not a formal privacy guarantee",
    }
    return write_distillation_package(
        output_path.resolve(),
        {"train": batches_for("train"), "validation": batches_for("validation")},
        metadata=metadata,
    )
