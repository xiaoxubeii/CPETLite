from __future__ import annotations

import json
import hashlib
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from cpet_foundation.data.sequences import (
    CpetSequenceDataset,
    SequenceConfig,
    ids_from_split,
    normalization_options_from_config,
)
from cpet_foundation.data.wearable import (
    WearableSample,
    WearableSimulationConfig,
    simulate_action_conditioned_wearable_sample,
    simulate_wearable_sample,
    wearable_manifest,
)
from cpet_foundation.eval.metrics import at_time_metrics, regression_metrics, trajectory_metrics
from cpet_foundation.model.losses import total_loss
from cpet_foundation.model.wearable_student import (
    DERIVED_CPET_FEATURES,
    build_wearable_student_from_config,
    derive_cpet_channels,
    wearable_student_loss,
)
from cpet_foundation.model.world_model import PROTOCOL_CONDITIONED_ARCHITECTURES, build_model


@dataclass
class EarlyStoppingTracker:
    mode: str = "min"
    patience: int = 8
    min_delta: float = 0.001
    best_value: float | None = None
    best_epoch: int = 0
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        self.mode = str(self.mode).lower()
        if self.mode not in {"min", "max"}:
            raise ValueError("early stopping mode must be 'min' or 'max'")
        if int(self.patience) < 1:
            raise ValueError("early stopping patience must be at least 1")
        if float(self.min_delta) < 0:
            raise ValueError("early stopping min_delta must be non-negative")

    def update(self, epoch: int, value: float) -> bool:
        current = float(value)
        if not np.isfinite(current):
            self.bad_epochs += 1
            return False
        if self.best_value is None:
            improved = True
        elif self.mode == "min":
            improved = (self.best_value - current) >= float(self.min_delta)
        else:
            improved = (current - self.best_value) >= float(self.min_delta)
        if improved:
            self.best_value = current
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= int(self.patience)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _deep_merge_config(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    resolved = path.resolve()
    if resolved in _seen:
        chain = " -> ".join(str(item) for item in (*_seen, resolved))
        raise ValueError(f"cyclic config inheritance: {chain}")
    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"training config must be a mapping: {resolved}")
    parent = loaded.pop("extends", None)
    if parent is None:
        return loaded
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    return _deep_merge_config(load_yaml(parent_path, (*_seen, resolved)), loaded)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def protocol_indices_from_config(config: dict) -> tuple[int, ...]:
    architecture = str(config.get("architecture", "masked_completion")).lower()
    if architecture not in PROTOCOL_CONDITIONED_ARCHITECTURES:
        return ()
    if config.get("protocol_feature_indices") is not None:
        return tuple(int(index) for index in config["protocol_feature_indices"])
    protocol_columns = set(config.get("protocol_feature_columns", ("Time", "Load_Phase", "Power_Load", "RPM")))
    return tuple(index for index, name in enumerate(config["feature_columns"]) if name in protocol_columns)


def shape_feature_indices_from_config(
    config: dict,
    target_feature_columns: tuple[str, ...],
) -> tuple[int, ...]:
    shape_columns = tuple(
        config.get(
            "slope_loss_feature_columns",
            config.get("shape_loss_feature_columns", ("VO2", "VO2_kg", "VCO2", "VE")),
        )
    )
    target_index = {name: index for index, name in enumerate(target_feature_columns)}
    return tuple(target_index[name] for name in shape_columns if name in target_index)


def _target_feature_indices(
    config: dict,
    key: str,
    target_feature_columns: tuple[str, ...],
    default: tuple[str, ...],
) -> tuple[int, ...]:
    selected = tuple(config.get(key, default))
    target_index = {name: index for index, name in enumerate(target_feature_columns)}
    return tuple(target_index[name] for name in selected if name in target_index)


def reconstruction_feature_indices(
    config: dict,
    target_feature_columns: tuple[str, ...],
    *,
    primary: bool = False,
) -> tuple[int, ...]:
    """Return configured reconstructed outcomes, excluding observed inputs."""
    key = "primary_reconstruction_feature_columns" if primary else "reconstruction_metric_feature_columns"
    fallback = ("VO2", "VCO2", "VE") if primary else target_feature_columns
    selected = tuple(config.get(key, fallback))
    observed = set(config.get("watch_channels", config.get("wearable_channels", ())))
    observed.update(config.get("protocol_context_channels", ()))
    target_index = {name: index for index, name in enumerate(target_feature_columns)}
    return tuple(target_index[name] for name in selected if name in target_index and name not in observed)


def _deterministic_base_feature_columns(config: dict) -> tuple[str, ...]:
    return tuple(
        config.get(
            "base_output_feature_columns",
            ("VO2", "VCO2", "VE", "HR", "SpO2", "PetO2", "PetCO2", "Power_Load", "RPM"),
        )
    )


def _derive_thresholds(config: dict) -> tuple[float, float, float]:
    return (
        float(config.get("derived_min_vo2", 50.0)),
        float(config.get("derived_min_vco2", 50.0)),
        float(config.get("derived_min_hr", 30.0)),
    )


def _observed_target_validity(source_mask: torch.Tensor, target_names: tuple[str, ...]) -> torch.Tensor:
    """Use source availability for decoded channels and their inputs for algebraic ones."""
    observed = source_mask.to(torch.bool).clone()
    index = {name: position for position, name in enumerate(target_names)}
    dependencies = {
        "VO2_kg": ("VO2",),
        "VCO2_kg": ("VCO2",),
        "RER": ("VO2", "VCO2"),
        "VE_VO2": ("VE", "VO2"),
        "VE_VCO2": ("VE", "VCO2"),
        "METS": ("VO2",),
        "VO2_HR": ("VO2", "HR"),
    }
    for name, inputs in dependencies.items():
        if name in index and all(input_name in index for input_name in inputs):
            valid = observed[..., index[inputs[0]]]
            for input_name in inputs[1:]:
                valid = valid & observed[..., index[input_name]]
            observed[..., index[name]] = valid
    return observed


def _training_output_scaler(
    dataset: CpetSequenceDataset,
    feature_columns: tuple[str, ...],
    target_feature_columns: tuple[str, ...],
    body_mass_metadata_index: int | None,
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_index = {name: index for index, name in enumerate(feature_columns)}
    target_source_indices = [source_index[name] for name in target_feature_columns]
    width = len(target_feature_columns)
    total = torch.zeros(width, dtype=torch.float64)
    total_sq = torch.zeros(width, dtype=torch.float64)
    count = torch.zeros(width, dtype=torch.float64)
    min_vo2, min_vco2, min_hr = _derive_thresholds(config)
    for sample in dataset.samples:
        x = torch.as_tensor(sample["x"], dtype=torch.float32)
        source_mean = torch.as_tensor(sample["feature_mean"], dtype=torch.float32)
        source_std = torch.as_tensor(sample["feature_std"], dtype=torch.float32).clamp_min(1e-6)
        physical = x[:, target_source_indices] * source_std[target_source_indices] + source_mean[target_source_indices]
        raw_metadata = torch.as_tensor(sample["metadata_raw"], dtype=torch.float32)
        mass = raw_metadata[body_mass_metadata_index].reshape(1) if body_mass_metadata_index is not None else None
        if bool(config.get("deterministic_derived_channels", False)):
            physical, channel_valid = derive_cpet_channels(
                physical.unsqueeze(0),
                target_feature_columns,
                mass,
                min_vo2=min_vo2,
                min_vco2=min_vco2,
                min_hr=min_hr,
            )
            physical = physical.squeeze(0)
            channel_valid = channel_valid.squeeze(0)
        else:
            channel_valid = torch.ones_like(physical, dtype=torch.bool)
        sequence_valid = torch.as_tensor(sample["mask"], dtype=torch.bool).unsqueeze(-1)
        source_valid = torch.as_tensor(sample.get("feature_mask", torch.ones_like(x)), dtype=torch.bool)
        source_valid = source_valid[:, target_source_indices]
        if bool(config.get("deterministic_derived_channels", False)):
            source_valid = _observed_target_validity(source_valid, target_feature_columns)
        valid = sequence_valid & source_valid & channel_valid & torch.isfinite(physical)
        values = physical.to(torch.float64)
        total += torch.where(valid, values, 0.0).sum(dim=0)
        total_sq += torch.where(valid, values.square(), 0.0).sum(dim=0)
        count += valid.sum(dim=0)
    safe_count = count.clamp_min(1.0)
    mean = total / safe_count
    variance = (total_sq / safe_count - mean.square()).clamp_min(1e-6)
    std = variance.sqrt()
    return mean.to(torch.float32), std.to(torch.float32)


def _training_metadata_scaler(
    dataset: CpetSequenceDataset,
    metadata_columns: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    width = len(metadata_columns)
    rows = [torch.as_tensor(sample["metadata_raw"], dtype=torch.float32) for sample in dataset.samples]
    if not rows:
        return torch.zeros(width, dtype=torch.float32), torch.ones(width, dtype=torch.float32)
    values = torch.stack(rows)
    finite = torch.isfinite(values)
    finite &= torch.stack([torch.as_tensor(sample.get("metadata_mask", torch.ones(width)),
                                          dtype=torch.bool) for sample in dataset.samples])
    for index, name in enumerate(metadata_columns):
        if name in {"Age", "Height_cm", "Weight_kg"}:
            finite[:, index] &= values[:, index] > 0
    count = finite.sum(dim=0).clamp_min(1)
    clean = torch.where(finite, values, 0.0)
    mean = clean.sum(dim=0) / count
    variance = (torch.where(finite, (values - mean).square(), 0.0).sum(dim=0) / count).clamp_min(1e-6)
    return mean, variance.sqrt()


def _deterministic_student_targets(
    batch: dict[str, torch.Tensor],
    teacher_outputs: dict[str, torch.Tensor],
    *,
    target_indices: tuple[int, ...],
    target_feature_columns: tuple[str, ...],
    output_mean: torch.Tensor,
    output_std: torch.Tensor,
    body_mass_kg: torch.Tensor | None,
    config: dict,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_index_tensor = torch.tensor(target_indices, device=batch["x"].device, dtype=torch.long)
    source_mean = batch["feature_mean"].index_select(-1, target_index_tensor).unsqueeze(1)
    source_std = batch["feature_std"].index_select(-1, target_index_tensor).clamp_min(1e-6).unsqueeze(1)
    target_old_z = _resize_steps(batch["x"].index_select(-1, target_index_tensor), steps)
    teacher_old_z = _resize_steps(teacher_outputs["trajectory_mu"].index_select(-1, target_index_tensor), steps)
    teacher_old_logvar = _resize_steps(teacher_outputs["trajectory_logvar"].index_select(-1, target_index_tensor), steps)
    target_physical = target_old_z * source_std + source_mean
    teacher_physical = teacher_old_z * source_std + source_mean
    min_vo2, min_vco2, min_hr = _derive_thresholds(config)
    rer_confidence = _rer_denominator_confidence(
        target_physical,
        target_feature_columns,
        min_vo2=min_vo2,
        min_vco2=min_vco2,
        soft_scale=float(config.get("rer_denominator_soft_scale", 150.0)),
    )
    target_physical, target_channel_valid = derive_cpet_channels(
        target_physical,
        target_feature_columns,
        body_mass_kg,
        min_vo2=min_vo2,
        min_vco2=min_vco2,
        min_hr=min_hr,
    )
    teacher_physical, _ = derive_cpet_channels(
        teacher_physical,
        target_feature_columns,
        body_mass_kg,
        min_vo2=min_vo2,
        min_vco2=min_vco2,
        min_hr=min_hr,
    )
    mean = output_mean.to(target_physical).reshape(1, 1, -1)
    std = output_std.to(target_physical).clamp_min(1e-6).reshape(1, 1, -1)
    target_z = (target_physical - mean) / std
    teacher_z = (teacher_physical - mean) / std
    teacher_logvar_z = teacher_old_logvar + 2.0 * torch.log(source_std / std)
    source_mask = batch.get("feature_mask")
    if source_mask is not None:
        observed = _resize_steps(source_mask.index_select(-1, target_index_tensor), steps) > 0
        observed = _observed_target_validity(observed, target_feature_columns)
        target_channel_valid = target_channel_valid & observed
    return target_z, target_channel_valid, teacher_z, teacher_logvar_z, rer_confidence


def _rer_denominator_confidence(
    target_physical: torch.Tensor,
    target_feature_columns: tuple[str, ...],
    *,
    min_vo2: float,
    min_vco2: float,
    soft_scale: float,
) -> torch.Tensor:
    """Ramp RER supervision from the hard validity threshold to full confidence."""
    if "VO2" not in target_feature_columns or "VCO2" not in target_feature_columns:
        return torch.ones(target_physical.shape[:2], device=target_physical.device, dtype=target_physical.dtype)
    scale = max(float(soft_scale), 1e-6)
    vo2 = target_physical[..., target_feature_columns.index("VO2")]
    vco2 = target_physical[..., target_feature_columns.index("VCO2")]
    vo2_confidence = ((vo2 - float(min_vo2)) / scale).clamp(0.0, 1.0)
    vco2_confidence = ((vco2 - float(min_vco2)) / scale).clamp(0.0, 1.0)
    confidence = torch.minimum(vo2_confidence, vco2_confidence)
    return torch.where(torch.isfinite(confidence), confidence, torch.zeros_like(confidence))


def model_name_for_architecture(architecture: str) -> str:
    if architecture == "wearable_student":
        return "cpet-world-wearable-student"
    if architecture == "protocol_direct_transformer":
        return "cpet-world-protocol-direct-transformer"
    if architecture in {"protocol_latent_sde", "protocol_sde", "protocol_conditioned_sde"}:
        return "cpet-world-protocol-sde"
    if architecture == "protocol_lss":
        return "cpet-world-protocol-lss"
    return "cpet-world"


def is_wearable_architecture(architecture: str) -> bool:
    return architecture == "wearable_student"


def apply_prefix_mask(
    batch: dict[str, torch.Tensor],
    min_fraction: float = 0.35,
    max_fraction: float = 0.75,
    protocol_indices: tuple[int, ...] = (),
) -> dict[str, torch.Tensor]:
    x = batch["x"].clone()
    mask = batch["mask"].clone()
    feature_mask = batch.get("feature_mask", torch.ones_like(x)).clone()
    original_x = x.clone()
    steps = x.shape[1]
    fractions = torch.empty(x.shape[0], device=x.device).uniform_(min_fraction, max_fraction)
    for row, fraction in enumerate(fractions):
        prefix_steps = max(2, min(steps, int(round(float(fraction) * steps))))
        x[row, prefix_steps:, :] = 0.0
        feature_mask[row, prefix_steps:, :] = 0.0
        if protocol_indices:
            x[row, prefix_steps:, list(protocol_indices)] = original_x[row, prefix_steps:, list(protocol_indices)]
            feature_mask[row, prefix_steps:, list(protocol_indices)] = batch.get(
                "feature_mask", torch.ones_like(x)
            )[row, prefix_steps:, list(protocol_indices)]
        mask[row, prefix_steps:] = 0.0
    masked = dict(batch)
    masked["x_input"] = x
    masked["mask_input"] = mask
    masked["feature_mask_input"] = feature_mask
    return masked


def prepare_world_model_input(
    batch: dict[str, torch.Tensor],
    *,
    mode: str = "prefix",
    protocol_indices: tuple[int, ...] = (),
) -> dict[str, torch.Tensor]:
    if mode == "full":
        return {
            **batch,
            "x_input": batch["x"],
            "mask_input": batch["mask"],
            "feature_mask_input": batch.get("feature_mask", torch.ones_like(batch["x"])),
        }
    if mode == "prefix":
        return apply_prefix_mask(batch, protocol_indices=protocol_indices)
    raise ValueError(f"unsupported world-model input mode: {mode!r}")


def train_from_config(config_path: Path) -> dict:
    config = load_yaml(config_path)
    set_seed(int(config.get("seed", 42)))
    root = config_path.resolve().parents[1]
    output_dir = (root / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(config.get("supervision_mode", "hybrid")).strip().lower() == "teacher_only":
        return train_wearable_student_from_package(
            config=config,
            root=root,
            output_dir=output_dir,
            device=_device(),
        )
    source_h5 = (root / config["source_h5"]).resolve() if not Path(config["source_h5"]).is_absolute() else Path(config["source_h5"])
    input_format = str(config.get("input_format", "registry"))
    split_path: Path | None = None

    limit_exams = _optional_int(config.get("limit_exams", 256))
    val_limit = None if limit_exams is None else max(1, limit_exams // 5)
    if input_format == "v5_derivative_split":
        if config.get("normalization_stats_path") not in (None, ""):
            raise ValueError("v5_derivative_split uses the producer-saved HDF5 scaler, not normalization_stats_path")
        train_ids = val_ids = None
        normalization_options: dict[str, object] = {}
    elif input_format == "registry":
        if "split_path" not in config:
            raise ValueError("registry input requires split_path")
        split_path = (root / config["split_path"]).resolve()
        train_ids = ids_from_split(split_path, "train", limit_exams)
        val_ids = ids_from_split(split_path, "validation", val_limit)
        normalization_options = normalization_options_from_config(config, root)
    else:
        raise ValueError(f"Unsupported input_format={input_format!r}")
    train_ds = CpetSequenceDataset(
        SequenceConfig(
            source_h5=source_h5,
            feature_columns=tuple(config["feature_columns"]),
            metadata_columns=tuple(config["metadata_columns"]),
            max_len=int(config.get("max_len", 160)),
            limit_exams=limit_exams,
            exam_ids=train_ids,
            input_format=input_format,
            split_name="train" if input_format == "v5_derivative_split" else None,
            **normalization_options,
        )
    )
    val_ds = CpetSequenceDataset(
        SequenceConfig(
            source_h5=source_h5,
            feature_columns=tuple(config["feature_columns"]),
            metadata_columns=tuple(config["metadata_columns"]),
            max_len=int(config.get("max_len", 160)),
            limit_exams=val_limit,
            exam_ids=val_ids,
            input_format=input_format,
            split_name="val" if input_format == "v5_derivative_split" else None,
            **normalization_options,
        )
    )
    if len(train_ds) < 4 or len(val_ds) < 1:
        raise RuntimeError(f"Not enough samples for training/validation: train={len(train_ds)} val={len(val_ds)}")
    device = _device()
    loader_kwargs = {
        "batch_size": int(config.get("batch_size", 16)),
        "num_workers": int(config.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    fitting_train, fitting_val = train_ds, val_ds
    if config.get("matched_input_package"):
        from cpet_foundation.data.matched_inputs import MatchedInputDataset
        if not is_wearable_architecture(str(config.get("architecture", "")).lower()):
            raise ValueError("matched_input_package requires a wearable student")
        package = _resolve_config_path(root, str(config["matched_input_package"]))
        fitting_train = MatchedInputDataset(train_ds, package, "train", config=config)
        fitting_val = MatchedInputDataset(val_ds, package, "validation", config=config)
        config["matched_input_provenance"] = {
            "train": fitting_train.provenance, "validation": fitting_val.provenance,
        }
    loader = DataLoader(fitting_train, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(fitting_val, **loader_kwargs)
    architecture = str(config.get("architecture", "masked_completion")).lower()
    if is_wearable_architecture(architecture):
        return train_wearable_student(
            config=config,
            root=root,
            output_dir=output_dir,
            source_h5=source_h5,
            split_path=split_path,
            train_ds=train_ds,
            val_ds=val_ds,
            loader=loader,
            val_loader=val_loader,
            device=device,
        )
    return _train_world_model(
        config=config,
        output_dir=output_dir,
        source_h5=source_h5,
        split_path=split_path,
        train_ds=train_ds,
        val_ds=val_ds,
        loader=loader,
        val_loader=val_loader,
        device=device,
    )


def _train_world_model(
    *,
    config: dict,
    output_dir: Path,
    source_h5: Path,
    split_path: Path | None,
    train_ds: CpetSequenceDataset,
    val_ds: CpetSequenceDataset,
    loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    model = build_model(
        config,
        len(config["feature_columns"]),
        len(config["metadata_columns"]),
        int(config.get("max_len", 160)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=1e-4,
    )
    feature_index = {name: idx for idx, name in enumerate(config["feature_columns"])}
    protocol_indices = protocol_indices_from_config(config)
    history = []
    for epoch in range(int(config.get("epochs", 1))):
        model.train()
        epoch_losses = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            batch = prepare_world_model_input(
                batch,
                mode=str(config.get("teacher_input_mode", "prefix")),
                protocol_indices=protocol_indices,
            )
            outputs = model(
                batch["x_input"], batch["metadata"], batch["mask_input"], batch["phase"],
                feature_mask=batch["feature_mask_input"],
                phase_mask=batch.get("phase_mask"),
            )
            loss, parts = total_loss(outputs, batch, feature_index)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            history.append(parts)
            epoch_losses.append(parts["loss"])
        print(f"epoch={epoch + 1} train_loss={float(np.mean(epoch_losses)):.4f}", flush=True)

    metrics = evaluate_model(
        model, val_loader, feature_index, device,
        protocol_indices=protocol_indices,
        input_mode=str(config.get("teacher_input_mode", "prefix")),
    )
    checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
        "feature_columns": config["feature_columns"],
        "metadata_columns": config["metadata_columns"],
        "metrics": metrics,
    }
    torch.save(checkpoint, output_dir / "model.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history[-100:], indent=2) + "\n", encoding="utf-8")
    architecture = str(config.get("architecture", "masked_completion")).lower()
    manifest = {
        "model": model_name_for_architecture(architecture),
        "architecture": config.get("architecture", "masked_completion"),
        "protocol_feature_columns": config.get("protocol_feature_columns", []),
        "run_type": output_dir.name,
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "device": str(device),
        "checkpoint": str(output_dir / "model.pt"),
        "metrics": str(output_dir / "metrics.json"),
        "feature_columns": config["feature_columns"],
        "metadata_columns": config["metadata_columns"],
        "source_h5": str(source_h5),
        "split_path": str(split_path) if split_path is not None else None,
        "input_format": config.get("input_format", "registry"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "metrics": metrics,
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "device": str(device),
    }


def train_wearable_student_from_package(
    *,
    config: dict,
    root: Path,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Train a student from teacher outputs without resolving raw CPET paths."""
    import hashlib

    from cpet_foundation.data.distillation_package import (
        DistillationPackageDataset,
        audit_distillation_package,
    )

    config = dict(config)
    if str(config.get("architecture", "")).lower() != "wearable_student":
        raise ValueError("teacher-only package training requires architecture=wearable_student")
    package_value = config.get("distillation_package_path")
    if not package_value:
        raise ValueError("teacher-only package training requires distillation_package_path")
    package_path = _resolve_config_path(root, str(package_value))
    audit = audit_distillation_package(package_path)
    if audit["supervision_mode"] != "teacher_only":
        raise ValueError("distillation package supervision_mode must be teacher_only")
    from cpet_foundation.data.matched_inputs import compute_precision, TRAINING_SEMANTICS_VERSION
    precision = compute_precision(config)
    matched_contract = audit.get("metadata", {}).get("matched_input_contract")
    if config.get("require_matched_input_contract"):
        from cpet_foundation.data.matched_inputs import checked_matched_package
        checked_matched_package(package_path, config=config)
    if matched_contract is not None and matched_contract.get("compute_precision") != precision:
        raise ValueError("teacher-only compute precision conflicts with matched input contract")
    config["mixed_precision"] = precision
    contract = dict(audit.get("metadata", {}).get("model_contract", {}))
    for key, value in contract.items():
        if key in config and config[key] not in (None, value):
            raise ValueError(f"teacher-only config conflicts with package model contract: {key}")
        config[key] = value
    config["supervision_mode"] = "teacher_only"

    preload_package = bool(config.get("preload_distillation_package", True))
    train_ds = DistillationPackageDataset(package_path, "train", preload=preload_package)
    val_ds = DistillationPackageDataset(package_path, "validation", preload=preload_package)
    if len(train_ds) < 4 or len(val_ds) < 1:
        raise RuntimeError(
            f"Not enough package samples for training/validation: train={len(train_ds)} val={len(val_ds)}"
        )
    batch_size = int(config.get("batch_size", 64))
    loader_kwargs = {"num_workers": 0, "pin_memory": device.type == "cuda"}
    if bool(config.get("vectorized_distillation_batches", True)):
        train_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(train_ds), batch_size, drop_last=False
        )
        val_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(val_ds), batch_size, drop_last=False
        )
        loader = DataLoader(train_ds, batch_size=None, sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=None, sampler=val_sampler, **loader_kwargs)
    else:
        loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_ds.collate, **loader_kwargs
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=val_ds.collate, **loader_kwargs
        )
    example = train_ds[0]
    target_features = int(example["teacher_trajectory_mu"].shape[-1])
    teacher_latent_width = int(example["teacher_latent_rollout"].shape[-1])
    if int(config.get("d_model", teacher_latent_width)) != teacher_latent_width:
        raise ValueError(
            "teacher-only latent alignment requires student d_model to match teacher latent width: "
            f"student={config.get('d_model')} teacher={teacher_latent_width}"
        )
    model = build_wearable_student_from_config(
        config,
        n_wearable_features=int(example["wearable_x"].shape[-1]),
        n_protocol_features=int(example["protocol_x"].shape[-1]),
        n_metadata=int(example["metadata"].shape[-1]),
        n_target_features=target_features,
        max_len=int(example["wearable_x"].shape[0]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=1e-4,
    )
    early_config = dict(config.get("early_stopping") or {})
    monitor_name = str(early_config.get("monitor", "teacher_validation_loss"))
    if monitor_name not in {"teacher_validation_loss", "teacher_core_trajectory_rmse"}:
        raise ValueError("unsupported teacher-only package training monitor")
    tracker = EarlyStoppingTracker(
        mode=str(early_config.get("mode", "min")),
        patience=int(early_config.get("patience", 20)),
        min_delta=float(early_config.get("min_delta", 0.001)),
    )
    scheduler_config = dict(config.get("lr_scheduler") or {})
    scheduler = None
    if str(scheduler_config.get("type", "none")).lower() == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_config.get("mode", "min")),
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 6)),
            min_lr=float(scheduler_config.get("min_lr", 1e-6)),
        )

    def move(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: value.to(device) for name, value in batch.items()}

    def forward_loss(package_batch):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda" and precision == "bfloat16"):
            outputs = model(
                package_batch["wearable_x"], package_batch["metadata"], package_batch["wearable_mask"],
                package_batch["mask"], package_batch["phase"], protocol_x=package_batch["protocol_x"],
                protocol_mask=package_batch["protocol_mask"], phase_mask=package_batch.get("phase_mask"),
                metadata_mask=package_batch.get("metadata_mask"),
                body_mass_kg=package_batch.get("body_mass_kg"),
            )
            loss, _ = wearable_student_loss(outputs, package_batch, config.get("loss_weights"),
                                            supervision_mode="teacher_only",
                                            trajectory_distillation_mode=str(config.get("trajectory_distillation_mode", "gaussian_nll")))
        return outputs, loss

    @torch.no_grad()
    def validate() -> dict[str, float | int | None]:
        model.eval()
        loss_values: list[float] = []
        trajectory_squared_error = 0.0
        trajectory_count = 0.0
        core_squared_error = 0.0
        core_count = 0.0
        at_absolute_error = 0.0
        peak_absolute_error = 0.0
        at_count = 0
        peak_count = 0
        for package_batch in val_loader:
            package_batch = move(package_batch)
            outputs, loss = forward_loss(package_batch)
            loss_values.append(float(loss.detach().cpu()))
            steps_mask = package_batch["mask"].unsqueeze(-1).to(outputs["trajectory_mu"].dtype)
            delta = outputs["trajectory_mu"] - package_batch["teacher_trajectory_mu"]
            valid_predictions = (steps_mask > 0) & torch.isfinite(package_batch["teacher_trajectory_mu"])
            if bool((valid_predictions & ~torch.isfinite(outputs["trajectory_mu"])).any()):
                raise FloatingPointError("Non-finite prediction at an eligible teacher trajectory target")
            trajectory_squared_error += float((delta.square() * steps_mask).sum().cpu())
            trajectory_count += float(steps_mask.sum().cpu()) * float(delta.shape[-1])
            target_names = tuple(config.get("target_feature_columns") or config.get("target_feature_names") or ())
            core_names = tuple(config.get("primary_reconstruction_feature_columns", ("VO2", "VCO2", "VE")))
            core_indices = (
                [target_names.index(name) for name in core_names]
                if all(name in target_names for name in core_names)
                else list(range(min(3, delta.shape[-1])))
            )
            core_delta = delta[..., core_indices]
            core_squared_error += float((core_delta.square() * steps_mask).sum().cpu())
            core_count += float(steps_mask.sum().cpu()) * len(core_indices)
            at_target = package_batch["teacher_at_norm"]
            peak_target = package_batch["teacher_vo2peak"]
            at_valid = (
                (package_batch.get("teacher_at_mask", torch.ones_like(at_target)) > 0)
                & torch.isfinite(at_target) & (at_target >= 0) & (at_target <= 1)
            )
            peak_valid = (
                (package_batch.get("teacher_vo2peak_mask", torch.ones_like(peak_target)) > 0)
                & torch.isfinite(peak_target)
            )
            if bool((at_valid & ~torch.isfinite(outputs["at_norm"])).any()
                    or (peak_valid & ~torch.isfinite(outputs["vo2peak"])).any()):
                raise FloatingPointError("Non-finite prediction at an eligible teacher endpoint target")
            at_absolute_error += float((outputs["at_norm"][at_valid] - at_target[at_valid]).abs().sum().cpu())
            peak_absolute_error += float((outputs["vo2peak"][peak_valid] - peak_target[peak_valid]).abs().sum().cpu())
            at_count += int(at_valid.sum())
            peak_count += int(peak_valid.sum())
        return {
            "teacher_validation_loss": float(np.mean(loss_values)),
            "teacher_trajectory_rmse": float(np.sqrt(trajectory_squared_error / max(trajectory_count, 1.0))),
            "teacher_core_trajectory_rmse": float(np.sqrt(core_squared_error / max(core_count, 1.0))),
            "teacher_at_norm_mae": at_absolute_error / at_count if at_count else None,
            "teacher_vo2peak_mae": peak_absolute_error / peak_count if peak_count else None,
            "teacher_at_valid_count": at_count,
            "teacher_peak_valid_count": peak_count,
        }

    history: list[dict[str, object]] = []
    best_model_state = None
    stopped_early = False
    for epoch in range(1, int(config.get("epochs", 1)) + 1):
        model.train()
        train_losses: list[float] = []
        for package_batch in loader:
            package_batch = move(package_batch)
            outputs, loss = forward_loss(package_batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite teacher-only student loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config.get("gradient_clip_norm", 5.0)))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        validation = validate()
        monitor_value = float(validation[monitor_name])
        improved = tracker.update(epoch, monitor_value)
        if improved:
            best_model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if scheduler is not None:
            scheduler.step(monitor_value)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **validation,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "improved": improved,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if bool(early_config.get("enabled", True)) and tracker.should_stop:
            stopped_early = True
            break

    if bool(early_config.get("restore_best_weights", True)) and best_model_state is not None:
        model.load_state_dict(best_model_state)
    metrics = validate()
    package_hash = hashlib.sha256()
    with package_path.open("rb") as package_handle:
        for chunk in iter(lambda: package_handle.read(1024 * 1024), b""):
            package_hash.update(chunk)
    package_digest = package_hash.hexdigest()
    training_control = {
        "max_epochs": int(config.get("epochs", 1)),
        "epochs_trained": len(history),
        "monitor": monitor_name,
        "best_epoch": tracker.best_epoch,
        "best_monitor_value": tracker.best_value,
        "stopped_early": stopped_early,
        "restore_best_weights": bool(early_config.get("restore_best_weights", True)),
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
    }
    checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
        "feature_columns": list(config.get("target_feature_columns", config.get("target_feature_names", ()))),
        "metadata_columns": list(config.get("metadata_columns", ())),
        "wearable_channels": list(config.get("watch_channels", ())),
        "watch_channels": list(config.get("watch_channels", ())),
        "protocol_context_channels": list(config.get("protocol_context_channels", ())),
        "input_boundary": config.get("input_boundary", "enhanced"),
        "joint_gas_exchange_decoder": bool(config.get("joint_gas_exchange_decoder", False)),
        "reconstruction_metric_feature_columns": list(
            config.get("reconstruction_metric_feature_columns", config.get("target_feature_columns", ()))
        ),
        "primary_reconstruction_feature_columns": list(
            config.get("primary_reconstruction_feature_columns", ("VO2", "VCO2", "VE"))
        ),
        "metrics": metrics,
        "training_control": training_control,
        "supervision_mode": "teacher_only",
        "distillation_package_sha256": package_digest,
    }
    torch.save(checkpoint, output_dir / "model.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "model": "cpet-world-wearable-student",
        "architecture": "wearable_student",
        "supervision_mode": "teacher_only",
        "student_training_source": "distillation_package_only",
        "raw_cpet_opened_by_student_trainer": False,
        "measured_training_targets_available": False,
        "distillation_package": str(package_path),
        "distillation_package_sha256": package_digest,
        "distillation_package_schema_version": audit["schema_version"],
        "matched_input_contract": matched_contract,
        "training_semantics_version": TRAINING_SEMANTICS_VERSION,
        "mixed_precision": precision,
        "student_visible_fields": audit["student_visible_fields"],
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "validation_scope": "teacher agreement only; measured clinical evaluation is separate",
        "metrics": str(output_dir / "metrics.json"),
        "checkpoint": str(output_dir / "model.pt"),
        "training_control": training_control,
        "loss_weights": config.get("loss_weights", {}),
        "input_ablation": config.get("input_ablation", "none"),
        "trajectory_distillation_mode": config.get("trajectory_distillation_mode", "gaussian_nll"),
        "evidence_boundary": "raw-CPET-isolated data-minimized training; not a formal privacy guarantee",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    train_ds.close()
    val_ds.close()
    return {
        "output_dir": str(output_dir),
        "metrics": metrics,
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "device": str(device),
    }


def reevaluate_wearable_checkpoint(config_path: Path, checkpoint_path: Path | None = None) -> dict:
    """Recompute wearable-student validation metrics without updating model weights."""
    from cpet_foundation.inference.digital_twin import load_checkpoint

    config = load_yaml(config_path)
    if not is_wearable_architecture(str(config.get("architecture", "")).lower()):
        raise ValueError("reevaluate_wearable_checkpoint requires architecture=wearable_student")
    evaluation_seed = int(config.get("seed", 42))
    set_seed(evaluation_seed)
    root = config_path.resolve().parents[1]
    source_h5 = _resolve_config_path(root, str(config["source_h5"]))
    split_path = _resolve_config_path(root, str(config["split_path"]))
    output_dir = _resolve_config_path(root, str(config["output_dir"]))
    checkpoint_path = checkpoint_path.resolve() if checkpoint_path else output_dir / "model.pt"

    limit_exams = _optional_int(config.get("limit_exams", 256))
    val_limit = None if limit_exams is None else max(1, limit_exams // 5)
    val_ids = ids_from_split(split_path, "validation", val_limit)
    normalization_options = normalization_options_from_config(config, root)
    val_ds = CpetSequenceDataset(
        SequenceConfig(
            source_h5=source_h5,
            feature_columns=tuple(config["feature_columns"]),
            metadata_columns=tuple(config["metadata_columns"]),
            max_len=int(config.get("max_len", 160)),
            limit_exams=val_limit,
            exam_ids=val_ids,
            **normalization_options,
        )
    )
    device = _device()
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config.get("batch_size", 16)),
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    teacher_model, teacher_checkpoint = load_checkpoint(_resolve_config_path(root, str(config["teacher_checkpoint"])))
    teacher_model = teacher_model.to(device).eval()
    config["teacher_metadata_columns"] = list(
        teacher_checkpoint.get("config", {}).get("metadata_columns", config.get("metadata_columns", ()))
    )
    target_feature_columns = tuple(config.get("target_feature_columns", config["feature_columns"]))
    feature_index = {name: idx for idx, name in enumerate(config["feature_columns"])}
    target_indices = tuple(feature_index[name] for name in target_feature_columns if name in feature_index)
    shape_feature_indices = shape_feature_indices_from_config(config, target_feature_columns)
    direct_loss_feature_indices = _target_feature_indices(
        config,
        "direct_loss_feature_columns",
        target_feature_columns,
        ("VO2", "VO2_kg", "VCO2", "VE"),
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
    wearable_config = wearable_simulation_config_from_training_config(config)
    model = build_wearable_student_from_config(
        config,
        n_wearable_features=len(wearable_config.selected_channels),
        n_protocol_features=len(wearable_config.protocol_channels),
        n_metadata=len(config["metadata_columns"]),
        n_target_features=len(target_indices),
        max_len=int(config.get("wearable_max_len", config.get("max_len", 160))),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    task_mode = resolve_wearable_task_mode(config)
    shape_loss_prefix_fraction = float(config.get("shape_loss_prefix_fraction", 0.6))
    metrics = evaluate_wearable_student(
        model,
        teacher_model,
        val_loader,
        tuple(config["feature_columns"]),
        target_indices,
        shape_feature_indices,
        shape_loss_prefix_fraction,
        wearable_config,
        device,
        task_mode=task_mode,
        online_safe_training=task_mode == "prefix_forecast",
        online_prefix_min_fraction=float(config.get("online_prefix_min_fraction", shape_loss_prefix_fraction)),
        online_prefix_max_fraction=float(config.get("online_prefix_max_fraction", shape_loss_prefix_fraction)),
        online_action_key=str(config.get("online_action_key", "continue_ramp")),
        training_config=config,
        prediction_output_path=output_dir / "validation_predictions.npz",
    )
    checkpoint["metrics"] = metrics
    checkpoint["evaluation_seed"] = evaluation_seed
    torch.save(checkpoint, checkpoint_path)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation_seed"] = evaluation_seed
    manifest["primary_at_endpoint_metric"] = "at_time_s_mae"
    manifest["validation_predictions"] = str(output_dir / "validation_predictions.npz")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(output_dir), "metrics": metrics, "validation_samples": len(val_ds), "device": str(device)}


def _validate_teacher_access_contract(config: dict) -> bool:
    """Return whether a teacher is configured and reject unsafe teacher-free losses."""
    configured = config.get("teacher_checkpoint")
    has_teacher = configured is not None and str(configured).strip().lower() not in {"", "none", "null"}
    if has_teacher:
        return True
    weights = config.get("loss_weights", {})
    teacher_terms = (
        "trajectory_distillation",
        "latent_alignment",
        "uncertainty_distillation",
        "teacher_endpoint",
        "teacher_at_endpoint",
        "at_region_distillation",
        "relation_distillation",
        "selective_distillation",
    )
    enabled = [name for name in teacher_terms if abs(float(weights.get(name, 0.0))) > 0.0]
    if enabled:
        raise ValueError(
            "teacher_checkpoint is absent but teacher-dependent losses are enabled: "
            + ", ".join(enabled)
        )
    return False


def train_wearable_student(
    *,
    config: dict,
    root: Path,
    output_dir: Path,
    source_h5: Path,
    split_path: Path | None,
    train_ds: CpetSequenceDataset,
    val_ds: CpetSequenceDataset,
    loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    from cpet_foundation.inference.digital_twin import load_checkpoint

    config = dict(config)
    teacher_accessed = _validate_teacher_access_contract(config)
    teacher_path: Path | None = None
    teacher_model: torch.nn.Module | None = None
    teacher_checkpoint: dict = {}
    if teacher_accessed:
        teacher_path = _resolve_config_path(root, str(config["teacher_checkpoint"]))
        teacher_model, teacher_checkpoint = load_checkpoint(teacher_path)
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        config["teacher_feature_columns"] = list(
            teacher_checkpoint.get("feature_columns", teacher_checkpoint.get("config", {}).get("feature_columns", ()))
        )
        config["teacher_metadata_columns"] = list(
            teacher_checkpoint.get("config", {}).get("metadata_columns", config.get("metadata_columns", ()))
        )
        config["teacher_max_len"] = int(
            teacher_checkpoint.get("config", {}).get("max_len", config.get("max_len", 160))
        )
    else:
        config["teacher_feature_columns"] = list(config["feature_columns"])
        config["teacher_metadata_columns"] = list(config.get("metadata_columns", ()))
        config["teacher_max_len"] = int(config.get("max_len", 160))

    selective_gate_model: torch.nn.Module | None = None
    selective_gate_path: Path | None = None
    selective_weight = float(config.get("loss_weights", {}).get("selective_distillation", 0.0))
    if abs(selective_weight) > 0.0:
        from cpet_foundation.inference.wearable_twin import load_wearable_checkpoint

        raw_gate_path = config.get("selective_gate_checkpoint")
        if raw_gate_path is None or not str(raw_gate_path).strip():
            raise ValueError("selective_distillation requires selective_gate_checkpoint")
        formatted_gate_path = str(raw_gate_path).format(seed=int(config.get("seed", 42)))
        selective_gate_path = _resolve_config_path(root, formatted_gate_path)
        selective_gate_model, _ = load_wearable_checkpoint(selective_gate_path)
        selective_gate_model = selective_gate_model.to(device)
        selective_gate_model.eval()
        for parameter in selective_gate_model.parameters():
            parameter.requires_grad_(False)

    target_feature_columns = tuple(config.get("target_feature_columns", config["feature_columns"]))
    feature_index = {name: idx for idx, name in enumerate(config["feature_columns"])}
    target_indices = tuple(feature_index[name] for name in target_feature_columns if name in feature_index)
    if not target_indices:
        raise RuntimeError("wearable student requires at least one target feature column")
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
    base_feature_columns = _deterministic_base_feature_columns(config) if deterministic_outputs else target_feature_columns
    if deterministic_outputs:
        missing_base = sorted(set(base_feature_columns).difference(target_feature_columns))
        missing_derived = sorted(set(DERIVED_CPET_FEATURES).difference(target_feature_columns))
        if missing_base or missing_derived:
            raise ValueError(
                f"deterministic output schema is incomplete; missing base={missing_base}, derived={missing_derived}"
            )
        output_feature_mean, output_feature_std = _training_output_scaler(
            train_ds,
            tuple(config["feature_columns"]),
            target_feature_columns,
            body_mass_metadata_index,
            config,
        )
    else:
        output_feature_mean = torch.zeros(len(target_feature_columns), dtype=torch.float32)
        output_feature_std = torch.ones(len(target_feature_columns), dtype=torch.float32)
    config["target_feature_names"] = list(target_feature_columns)
    config["base_feature_names"] = list(base_feature_columns)
    config["output_feature_mean"] = output_feature_mean.tolist()
    config["output_feature_std"] = output_feature_std.tolist()
    metadata_feature_mean, metadata_feature_std = _training_metadata_scaler(train_ds, metadata_columns)
    config["metadata_feature_mean"] = metadata_feature_mean.tolist()
    config["metadata_feature_std"] = metadata_feature_std.tolist()
    distillation_feature_indices = tuple(target_feature_columns.index(name) for name in base_feature_columns)
    relation_feature_indices = _target_feature_indices(
        config,
        "distillation_relation_feature_columns",
        target_feature_columns,
        ("VO2", "VCO2", "VE"),
    )
    derived_loss_feature_indices = tuple(target_feature_columns.index(name) for name in DERIVED_CPET_FEATURES)
    gas_exchange_feature_indices = tuple(
        target_feature_columns.index(name) for name in ("VO2", "VCO2") if name in target_feature_columns
    )
    shape_loss_prefix_fraction = float(config.get("shape_loss_prefix_fraction", 0.6))
    task_mode = resolve_wearable_task_mode(config)
    online_safe_training = task_mode == "prefix_forecast"
    online_prefix_min_fraction = float(config.get("online_prefix_min_fraction", shape_loss_prefix_fraction))
    online_prefix_max_fraction = float(config.get("online_prefix_max_fraction", shape_loss_prefix_fraction))
    online_action_key = str(config.get("online_action_key", "continue_ramp"))

    wearable_config = wearable_simulation_config_from_training_config(config)
    model = build_wearable_student_from_config(
        config,
        n_wearable_features=len(wearable_config.selected_channels),
        n_protocol_features=len(wearable_config.protocol_channels),
        n_metadata=len(config["metadata_columns"]),
        n_target_features=len(target_indices),
        max_len=int(config.get("wearable_max_len", config.get("max_len", 160))),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)), weight_decay=1e-4)
    early_config = dict(config.get("early_stopping") or {})
    early_enabled = bool(early_config.get("enabled", False))
    monitor_name = str(early_config.get("monitor", "task_trajectory_rmse"))
    monitor_mode = str(early_config.get("mode", "min"))
    tracker = EarlyStoppingTracker(
        mode=monitor_mode,
        patience=int(early_config.get("patience", 8)),
        min_delta=float(early_config.get("min_delta", 0.001)),
    )
    scheduler_config = dict(config.get("lr_scheduler") or {})
    scheduler_type = str(scheduler_config.get("type", "none")).lower()
    scheduler_enabled = scheduler_type == "reduce_on_plateau"
    if scheduler_type not in {"none", "reduce_on_plateau"}:
        raise ValueError(f"unsupported lr_scheduler type: {scheduler_type}")
    scheduler = None
    if scheduler_enabled:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_config.get("mode", monitor_mode)),
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 3)),
            min_lr=float(scheduler_config.get("min_lr", 1e-6)),
        )
    validation_controls_enabled = early_enabled or scheduler_enabled
    from cpet_foundation.data.matched_inputs import compute_precision, TRAINING_SEMANTICS_VERSION
    mixed_precision = compute_precision(config)
    config["mixed_precision"] = mixed_precision
    autocast_enabled = device.type == "cuda" and mixed_precision == "bfloat16"
    epoch_history = []
    best_model_state = None
    stopped_early = False
    for epoch in range(int(config.get("epochs", 1))):
        model.train()
        epoch_losses = []
        epoch_loss_parts: dict[str, list[float]] = {}
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                student_batch = build_wearable_student_batch(
                    batch,
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
                    distillation_relation_feature_indices=relation_feature_indices,
                    body_mass_metadata_index=body_mass_metadata_index,
                    target_feature_columns=target_feature_columns,
                    output_feature_mean=output_feature_mean,
                    output_feature_std=output_feature_std,
                    deterministic_config=config,
                    shape_loss_prefix_fraction=shape_loss_prefix_fraction,
                    simulation_config=wearable_config,
                    teacher_model=teacher_model,
                    device=device,
                    task_mode=task_mode,
                    online_safe_training=online_safe_training,
                    online_prefix_min_fraction=online_prefix_min_fraction,
                    online_prefix_max_fraction=online_prefix_max_fraction,
                    online_action_key=online_action_key,
                    supervision_mode=str(config.get("supervision_mode", "hybrid")),
                )
                if selective_gate_model is not None:
                    with torch.no_grad():
                        reference_outputs = selective_gate_model(
                            student_batch["wearable_x"],
                            student_batch["metadata"],
                            student_batch["wearable_mask"],
                            student_batch["mask"],
                            student_batch["phase"],
                            protocol_x=student_batch["protocol_x"],
                            protocol_mask=student_batch["protocol_mask"],
                            phase_mask=student_batch.get("phase_mask"),
                            metadata_mask=student_batch.get("metadata_mask"),
                            body_mass_kg=student_batch["body_mass_kg"],
                        )
                    student_batch["reference_trajectory_mu"] = reference_outputs["trajectory_mu"].detach()
                    student_batch["reference_at_norm"] = reference_outputs["at_norm"].detach()
                outputs = model(
                    student_batch["wearable_x"],
                    student_batch["metadata"],
                    student_batch["wearable_mask"],
                    student_batch["mask"],
                    student_batch["phase"],
                    protocol_x=student_batch["protocol_x"],
                    protocol_mask=student_batch["protocol_mask"],
                    phase_mask=student_batch.get("phase_mask"),
                    metadata_mask=student_batch.get("metadata_mask"),
                    body_mass_kg=student_batch["body_mass_kg"],
                )
                loss, parts = wearable_student_loss(
                    outputs,
                    student_batch,
                    config.get("loss_weights"),
                    supervision_mode=str(config.get("supervision_mode", "hybrid")),
                )
            if not bool(torch.isfinite(loss)):
                nonfinite_parts = {
                    name: value for name, value in parts.items() if not np.isfinite(float(value))
                }
                finite_fraction = float(torch.isfinite(outputs["trajectory_mu"]).float().mean().item())
                raise FloatingPointError(
                    f"non-finite wearable student loss: parts={nonfinite_parts}, "
                    f"trajectory_finite_fraction={finite_fraction:.6f}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(parts["loss"])
            for name, value in parts.items():
                epoch_loss_parts.setdefault(name, []).append(float(value))
        epoch_number = epoch + 1
        train_loss = float(np.mean(epoch_losses))
        validation_metrics = None
        improved = False
        if validation_controls_enabled:
            validation_metrics = evaluate_wearable_student(
                model,
                teacher_model,
                val_loader,
                tuple(config["feature_columns"]),
                target_indices,
                shape_feature_indices,
                shape_loss_prefix_fraction,
                wearable_config,
                device,
                task_mode=task_mode,
                online_safe_training=online_safe_training,
                online_prefix_min_fraction=online_prefix_min_fraction,
                online_prefix_max_fraction=online_prefix_max_fraction,
                online_action_key=online_action_key,
                training_config=config,
            )
            if monitor_name not in validation_metrics:
                raise KeyError(f"early stopping monitor {monitor_name!r} is not present in validation metrics")
            monitor_value = float(validation_metrics[monitor_name])
            improved = tracker.update(epoch_number, monitor_value)
            if improved:
                best_model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if scheduler is not None:
                scheduler_monitor = str(scheduler_config.get("monitor", monitor_name))
                if scheduler_monitor not in validation_metrics:
                    raise KeyError(f"lr scheduler monitor {scheduler_monitor!r} is not present in validation metrics")
                scheduler.step(float(validation_metrics[scheduler_monitor]))
        learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_record = {
            "epoch": epoch_number,
            "train_loss": train_loss,
            "train_loss_parts": {name: float(np.mean(values)) for name, values in epoch_loss_parts.items()},
            "learning_rate": learning_rate,
            "improved": improved,
            "bad_epochs": tracker.bad_epochs if validation_controls_enabled else 0,
            "validation": validation_metrics,
        }
        epoch_history.append(epoch_record)
        validation_text = ""
        if validation_metrics is not None:
            validation_text = f" {monitor_name}={float(validation_metrics[monitor_name]):.4f} lr={learning_rate:.7f}"
        print(f"epoch={epoch_number} train_loss={train_loss:.4f}{validation_text}", flush=True)
        if early_enabled and tracker.should_stop:
            stopped_early = True
            print(f"early_stop epoch={epoch_number} best_epoch={tracker.best_epoch} best_{monitor_name}={tracker.best_value:.4f}", flush=True)
            break

    restore_best_weights = bool(early_config.get("restore_best_weights", True))
    if validation_controls_enabled and restore_best_weights and best_model_state is not None:
        model.load_state_dict(best_model_state)
    metrics = evaluate_wearable_student(
        model,
        teacher_model,
        val_loader,
        tuple(config["feature_columns"]),
        target_indices,
        shape_feature_indices,
        shape_loss_prefix_fraction,
        wearable_config,
        device,
        task_mode=task_mode,
        online_safe_training=online_safe_training,
        online_prefix_min_fraction=online_prefix_min_fraction,
        online_prefix_max_fraction=online_prefix_max_fraction,
        online_action_key=online_action_key,
        training_config=config,
        prediction_output_path=output_dir / "validation_predictions.npz",
    )
    training_control = {
        "max_epochs": int(config.get("epochs", 1)),
        "epochs_trained": len(epoch_history),
        "monitor": monitor_name if validation_controls_enabled else None,
        "mode": monitor_mode if validation_controls_enabled else None,
        "best_epoch": tracker.best_epoch if validation_controls_enabled else len(epoch_history),
        "best_monitor_value": tracker.best_value if validation_controls_enabled else None,
        "stopped_early": stopped_early,
        "restore_best_weights": restore_best_weights if validation_controls_enabled else False,
        "early_stopping": early_config,
        "lr_scheduler": scheduler_config,
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
    }
    checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
        "workout_scenario": wearable_config.workout_scenario,
        "wearable_channels": list(wearable_config.selected_channels),
        "watch_channels": list(wearable_config.selected_channels),
        "protocol_context_channels": list(wearable_config.protocol_channels),
        "feature_columns": list(target_feature_columns),
        "metadata_columns": config["metadata_columns"],
        "input_boundary": config.get("input_boundary", "legacy"),
        "joint_gas_exchange_decoder": bool(config.get("joint_gas_exchange_decoder", False)),
        "reconstruction_metric_feature_columns": list(
            config.get("reconstruction_metric_feature_columns", target_feature_columns)
        ),
        "primary_reconstruction_feature_columns": list(
            config.get("primary_reconstruction_feature_columns", ("VO2", "VCO2", "VE"))
        ),
        "metrics": metrics,
        "training_control": training_control,
    }
    torch.save(checkpoint, output_dir / "model.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(epoch_history, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "model": model_name_for_architecture("wearable_student"),
        "architecture": "wearable_student",
        "comparison_role": config.get("comparison_role", "distilled_neural_sde"),
        "experiment_group": config.get("experiment_group"),
        "temporal_backbone_type": config.get("temporal_backbone_type", "transformer"),
        "workout_scenario": wearable_config.workout_scenario,
        "rollout_type": config.get("rollout_type", "neural_sde"),
        "loss_weights": config.get("loss_weights", {}),
        "future_sample_count": int(config.get("future_sample_count", 0)),
        "action_future_sample_count": int(config.get("action_future_sample_count", 0)),
        "online_safe_training": online_safe_training,
        "task_mode": task_mode,
        "training_control": training_control,
        "online_prefix_min_fraction": online_prefix_min_fraction,
        "online_prefix_max_fraction": online_prefix_max_fraction,
        "online_action_key": online_action_key,
        "run_type": output_dir.name,
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "device": str(device),
        "checkpoint": str(output_dir / "model.pt"),
        "metrics": str(output_dir / "metrics.json"),
        "validation_predictions": str(output_dir / "validation_predictions.npz"),
        "evaluation_seed": int(config.get("seed", 42)),
        "primary_at_endpoint_metric": "at_time_s_mae",
        "teacher_checkpoint": str(teacher_path) if teacher_path is not None else None,
        "selective_gate_checkpoint": str(selective_gate_path) if selective_gate_path is not None else None,
        "teacher_accessed_during_training": teacher_accessed,
        "matched_input_provenance": config.get("matched_input_provenance"),
        "training_semantics_version": TRAINING_SEMANTICS_VERSION,
        "mixed_precision": mixed_precision,
        "teacher_architecture": teacher_checkpoint.get("config", {}).get("architecture"),
        "teacher_metrics": teacher_checkpoint.get("metrics", {}),
        "wearable_channels": list(wearable_config.selected_channels),
        "watch_channels": list(wearable_config.selected_channels),
        "protocol_context_channels": list(wearable_config.protocol_channels),
        "target_feature_columns": list(target_feature_columns),
        "metadata_columns": config["metadata_columns"],
        "input_boundary": config.get("input_boundary", "legacy"),
        "joint_gas_exchange_decoder": bool(config.get("joint_gas_exchange_decoder", False)),
        "base_output_feature_columns": list(base_feature_columns),
        "reconstruction_metric_feature_columns": list(
            config.get("reconstruction_metric_feature_columns", target_feature_columns)
        ),
        "primary_reconstruction_feature_columns": list(
            config.get("primary_reconstruction_feature_columns", ("VO2", "VCO2", "VE"))
        ),
        "deterministic_derived_channels": list(DERIVED_CPET_FEATURES) if deterministic_outputs else [],
        "source_h5": str(source_h5),
        "split_path": str(split_path) if split_path is not None else None,
        "input_format": config.get("input_format", "registry"),
        "simulation": wearable_manifest(wearable_config, tuple(config["feature_columns"])),
        "real_paired_validation_available": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "metrics": metrics,
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "device": str(device),
    }


def _resolve_config_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (root / path).resolve()


def wearable_simulation_config_from_training_config(config: dict) -> WearableSimulationConfig:
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
        seed=int(config.get("wearable_simulation_seed", 42)),
    )


WEARABLE_TASK_MODES = ("full_reconstruction", "prefix_forecast")


def resolve_wearable_task_mode(config: dict) -> str:
    """Resolve the explicit task mode while preserving legacy boolean configs."""
    raw_mode = config.get("task_mode")
    if raw_mode is None:
        return "prefix_forecast" if bool(config.get("online_safe_training", False)) else "full_reconstruction"
    task_mode = str(raw_mode).strip().lower()
    if task_mode not in WEARABLE_TASK_MODES:
        supported = ", ".join(WEARABLE_TASK_MODES)
        raise ValueError(f"unsupported wearable task_mode {raw_mode!r}; expected one of: {supported}")
    return task_mode


@torch.no_grad()
def _teacher_outputs_for_student_batch(
    batch: dict[str, torch.Tensor],
    teacher_model: torch.nn.Module | None,
    runtime_config: dict,
    student_feature_columns: tuple[str, ...],
    target_feature_columns: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    if teacher_model is None:
        batch_size, steps, _ = batch["x"].shape
        return {
            "latent_rollout": torch.zeros(
                batch_size,
                steps,
                int(runtime_config.get("latent_dim", 192)),
                device=batch["x"].device,
                dtype=batch["x"].dtype,
            ),
            "trajectory_mu": batch["x"].detach(),
            "trajectory_logvar": torch.zeros_like(batch["x"]),
            "at_norm": batch["at_norm"].detach(),
            "vo2peak": batch["vo2peak"].detach(),
        }
    teacher_metadata = batch["metadata"]
    teacher_feature_columns = tuple(runtime_config.get("teacher_feature_columns", student_feature_columns))
    missing_teacher_features = [name for name in teacher_feature_columns if name not in student_feature_columns]
    if missing_teacher_features:
        raise ValueError(f"teacher feature columns are absent from student data: {missing_teacher_features}")
    teacher_feature_indices = torch.tensor(
        [student_feature_columns.index(name) for name in teacher_feature_columns],
        device=batch["x"].device,
        dtype=torch.long,
    )
    teacher_steps = int(runtime_config.get("teacher_max_len", batch["x"].shape[1]))
    teacher_x = _resize_steps(batch["x"].index_select(-1, teacher_feature_indices), teacher_steps)
    teacher_mask = _resize_steps(batch["mask"], teacher_steps)
    teacher_phase = _resize_phase(batch["phase"], teacher_steps)
    student_metadata_columns = tuple(runtime_config.get("metadata_columns", ()))
    teacher_metadata_columns = tuple(runtime_config.get("teacher_metadata_columns", student_metadata_columns))
    if teacher_metadata_columns and teacher_metadata_columns != student_metadata_columns:
        missing = [name for name in teacher_metadata_columns if name not in student_metadata_columns]
        if missing:
            raise ValueError(f"teacher metadata columns are absent from student data: {missing}")
        teacher_metadata_indices = torch.tensor(
            [student_metadata_columns.index(name) for name in teacher_metadata_columns],
            device=batch["metadata"].device,
            dtype=torch.long,
        )
        teacher_metadata = batch["metadata"].index_select(-1, teacher_metadata_indices)
    teacher_kwargs = {}
    if runtime_config.get("input_format") == "v5_derivative_split":
        teacher_kwargs["feature_mask"] = _resize_steps(batch["feature_mask"].index_select(-1, teacher_feature_indices), teacher_steps)
        teacher_kwargs["phase_mask"] = _resize_phase(batch["phase_mask"].long(), teacher_steps).to(teacher_x.dtype)
    teacher_outputs = teacher_model(teacher_x, teacher_metadata, teacher_mask, teacher_phase, **teacher_kwargs)
    if teacher_feature_columns != student_feature_columns:
        aligned_mu = torch.zeros(
            *teacher_outputs["trajectory_mu"].shape[:-1],
            len(student_feature_columns),
            device=teacher_outputs["trajectory_mu"].device,
            dtype=teacher_outputs["trajectory_mu"].dtype,
        )
        aligned_logvar = torch.zeros_like(aligned_mu)
        student_positions = torch.tensor(
            [student_feature_columns.index(name) for name in teacher_feature_columns],
            device=aligned_mu.device,
            dtype=torch.long,
        )
        aligned_mu.index_copy_(-1, student_positions, teacher_outputs["trajectory_mu"])
        aligned_logvar.index_copy_(-1, student_positions, teacher_outputs["trajectory_logvar"])
        teacher_outputs = {
            **teacher_outputs,
            "trajectory_mu": aligned_mu,
            "trajectory_logvar": aligned_logvar,
        }
        if not bool(runtime_config.get("deterministic_derived_channels", False)):
            missing_targets = [name for name in target_feature_columns if name not in teacher_feature_columns]
            if missing_targets:
                raise ValueError(
                    "teacher is missing non-deterministic student targets: "
                    f"{missing_targets}"
                )
    return teacher_outputs


def build_wearable_student_batch(
    batch: dict[str, torch.Tensor],
    *,
    feature_columns: tuple[str, ...],
    target_indices: tuple[int, ...],
    shape_feature_indices: tuple[int, ...] = (),
    direct_loss_feature_indices: tuple[int, ...] = (),
    core_derived_feature_indices: tuple[int, ...] = (),
    slope_loss_feature_indices: tuple[int, ...] = (),
    peak_loss_feature_indices: tuple[int, ...] = (),
    derived_feature_indices: tuple[int, ...] = (),
    derived_loss_feature_indices: tuple[int, ...] = (),
    distillation_feature_indices: tuple[int, ...] = (),
    distillation_relation_feature_indices: tuple[int, ...] = (),
    body_mass_metadata_index: int | None = None,
    target_feature_columns: tuple[str, ...] = (),
    output_feature_mean: torch.Tensor | None = None,
    output_feature_std: torch.Tensor | None = None,
    deterministic_config: dict | None = None,
    shape_loss_prefix_fraction: float = 0.6,
    simulation_config: WearableSimulationConfig,
    teacher_model: torch.nn.Module | None,
    device: torch.device,
    task_mode: str | None = None,
    online_safe_training: bool = False,
    online_prefix_min_fraction: float = 0.35,
    online_prefix_max_fraction: float = 0.75,
    online_action_key: str = "continue_ramp",
    supervision_mode: str = "hybrid",
) -> dict[str, torch.Tensor]:
    from cpet_foundation.inference.wearable_twin import _phase_from_protocol_plan, _planned_protocol

    resolved_task_mode = resolve_wearable_task_mode(
        {"task_mode": task_mode} if task_mode is not None else {"online_safe_training": online_safe_training}
    )
    is_prefix_forecast = resolved_task_mode == "prefix_forecast"
    wearable_samples = []
    phase_samples = []
    prefix_steps_by_row = []
    plan_config = {"feature_columns": list(feature_columns)}
    for row in range(batch["x"].shape[0]):
        if "fixed_wearable_x" in batch:
            if is_prefix_forecast:
                raise ValueError("fixed input packages currently require full_reconstruction")
            wearable_samples.append(WearableSample(
                x=batch["fixed_wearable_x"][row], mask=batch["fixed_wearable_mask"][row],
                channels=simulation_config.selected_channels, sampling_rate_hz=simulation_config.sampling_rate_hz,
                protocol_x=batch["fixed_protocol_x"][row], protocol_mask=batch["fixed_protocol_mask"][row],
                protocol_channels=simulation_config.protocol_channels,
            ))
            continue
        if "sample_token" in batch:
            token = batch["sample_token"][row].detach().cpu().tolist()
            identity = ":".join(str(int(value)) for value in token)
        elif "sample_index" in batch:
            identity = str(int(batch["sample_index"][row]))
        else:
            # Historical ad-hoc callers have no identity. Do not make their
            # corruption depend on minibatch position either.
            identity = "unidentified"
        seed_bytes = hashlib.sha256(f"{simulation_config.seed}:{identity}".encode()).digest()
        row_seed = int.from_bytes(seed_bytes[:8], "big") % (2**32)
        row_config = replace(simulation_config, seed=row_seed)
        row_sample = _row_sequence_sample(batch, row)
        if is_prefix_forecast:
            prefix_fraction = _deterministic_online_prefix_fraction(row_seed, online_prefix_min_fraction, online_prefix_max_fraction)
            prefix_steps = max(2, min(int(row_config.max_len or row_sample["x"].shape[0]), int(round(row_sample["x"].shape[0] * prefix_fraction))))
            protocol_plan = _planned_protocol(row_sample, plan_config, tuple(row_config.protocol_channels), prefix_steps, online_action_key)
            wearable = simulate_action_conditioned_wearable_sample(
                row_sample,
                feature_columns,
                row_config,
                prefix_steps=prefix_steps,
                protocol_plan=protocol_plan,
            )
            phase_samples.append(_phase_from_protocol_plan(wearable, row_sample, wearable.x.shape[0]))
            prefix_steps_by_row.append(prefix_steps)
        else:
            wearable = simulate_wearable_sample(row_sample, feature_columns, row_config)
        wearable_samples.append(wearable)
    wearable_x = torch.stack([sample.x for sample in wearable_samples]).to(device)
    wearable_mask = torch.stack([sample.mask for sample in wearable_samples]).to(device)
    protocol_x = torch.stack([sample.protocol_x for sample in wearable_samples]).to(device)
    protocol_mask = torch.stack([sample.protocol_mask for sample in wearable_samples]).to(device)
    source_time_mask = batch["mask"]
    if source_time_mask.ndim == 3:
        source_time_mask = source_time_mask.gt(0).any(dim=-1)
    sequence_mask = _resize_phase(source_time_mask, wearable_x.shape[1]).gt(0).to(wearable_x.dtype)
    wearable_mask = wearable_mask * sequence_mask.unsqueeze(-1)
    protocol_mask = protocol_mask * sequence_mask.unsqueeze(-1)
    if is_prefix_forecast:
        future_mask = _future_mask_from_prefix_steps(sequence_mask, prefix_steps_by_row)
        shape_mask = future_mask
        phase = torch.stack(phase_samples).to(device)
    else:
        future_mask = torch.zeros_like(sequence_mask)
        shape_mask = sequence_mask
        phase = _resize_phase(batch["phase"], wearable_x.shape[1]).to(device)
    runtime_config = deterministic_config or {}
    student_feature_columns = tuple(feature_columns)
    teacher_outputs = _teacher_outputs_for_student_batch(
        batch,
        teacher_model,
        runtime_config,
        student_feature_columns,
        target_feature_columns,
    )
    target_index_tensor = torch.tensor(target_indices, device=device, dtype=torch.long)
    body_mass_kg = (
        batch["metadata_raw"][:, body_mass_metadata_index]
        if body_mass_metadata_index is not None and "metadata_raw" in batch
        else torch.zeros(batch["x"].shape[0], device=device, dtype=batch["x"].dtype)
    )
    use_deterministic = bool((deterministic_config or {}).get("deterministic_derived_channels", False))
    if use_deterministic:
        if output_feature_mean is None or output_feature_std is None:
            raise ValueError("deterministic outputs require a train-set output scaler")
        (
            target_trajectory,
            target_channel_mask,
            teacher_trajectory_mu,
            teacher_trajectory_logvar,
            core_derived_confidence,
        ) = _deterministic_student_targets(
            batch,
            teacher_outputs,
            target_indices=target_indices,
            target_feature_columns=target_feature_columns,
            output_mean=output_feature_mean,
            output_std=output_feature_std,
            body_mass_kg=body_mass_kg,
            config=deterministic_config or {},
            steps=wearable_x.shape[1],
        )
        target_feature_mean = output_feature_mean.to(device).reshape(1, -1).expand(batch["x"].shape[0], -1)
        target_feature_std = output_feature_std.to(device).reshape(1, -1).expand(batch["x"].shape[0], -1)
    else:
        target_trajectory = _resize_steps(batch["x"].index_select(-1, target_index_tensor), wearable_x.shape[1])
        source_mask = batch.get("feature_mask")
        target_channel_mask = (
            _resize_steps(source_mask.index_select(-1, target_index_tensor), wearable_x.shape[1]) > 0
            if source_mask is not None else torch.ones_like(target_trajectory, dtype=torch.bool)
        )
        teacher_trajectory_mu = _resize_steps(
            teacher_outputs["trajectory_mu"].index_select(-1, target_index_tensor), wearable_x.shape[1]
        )
        teacher_trajectory_logvar = _resize_steps(
            teacher_outputs["trajectory_logvar"].index_select(-1, target_index_tensor), wearable_x.shape[1]
        )
        target_feature_mean = batch.get(
            "feature_mean",
            torch.zeros(batch["x"].shape[0], batch["x"].shape[-1], device=device, dtype=batch["x"].dtype),
        ).index_select(-1, target_index_tensor)
        target_feature_std = batch.get(
            "feature_std",
            torch.ones(batch["x"].shape[0], batch["x"].shape[-1], device=device, dtype=batch["x"].dtype),
        ).index_select(-1, target_index_tensor)
        core_derived_confidence = torch.ones_like(sequence_mask)
    target_channel_mask = target_channel_mask * sequence_mask.unsqueeze(-1)
    student_batch = {
        "wearable_x": wearable_x,
        "wearable_mask": wearable_mask,
        "protocol_x": protocol_x,
        "protocol_mask": protocol_mask,
        "mask": sequence_mask,
        "future_mask": future_mask,
        "shape_mask": shape_mask,
        "task_evaluation_mask": shape_mask,
        "phase": phase,
        "phase_mask": (protocol_mask[..., tuple(simulation_config.protocol_channels).index("Load_Phase")]
                       if "Load_Phase" in simulation_config.protocol_channels
                       else _resize_phase(batch.get("phase_mask", batch["mask"]).long(), wearable_x.shape[1]).to(sequence_mask.dtype)),
        "metadata": batch.get("metadata_raw", batch["metadata"]),
        "metadata_mask": batch.get("metadata_mask", torch.isfinite(batch.get("metadata_raw", batch["metadata"])).to(batch["x"].dtype)),
        "teacher_latent_rollout": _resize_steps(teacher_outputs["latent_rollout"], wearable_x.shape[1]),
        "teacher_trajectory_mu": teacher_trajectory_mu,
        "teacher_trajectory_logvar": teacher_trajectory_logvar,
        "at_norm": batch["at_norm"],
        "at_supervision_mask": batch.get(
            "at_supervision_mask", torch.ones_like(batch["at_norm"])
        ),
        "at_metric_mask": batch.get("at_metric_mask", torch.ones_like(batch["at_norm"])),
        "vo2peak": batch["vo2peak"],
        "vo2peak_mask": batch.get("vo2peak_mask", torch.ones_like(batch["vo2peak"])),
        "trajectory_supervision_mask": batch.get(
            "trajectory_supervision_mask", torch.ones_like(batch["at_norm"])
        ),
        "target_trajectory": target_trajectory,
        "target_channel_mask": target_channel_mask.to(batch["x"].dtype),
        "shape_feature_indices": torch.tensor(shape_feature_indices, device=device, dtype=torch.long),
        "direct_loss_feature_indices": torch.tensor(direct_loss_feature_indices, device=device, dtype=torch.long),
        "core_derived_feature_indices": torch.tensor(
            core_derived_feature_indices, device=device, dtype=torch.long
        ),
        "core_derived_confidence": core_derived_confidence.to(batch["x"].dtype),
        "slope_loss_feature_indices": torch.tensor(slope_loss_feature_indices, device=device, dtype=torch.long),
        "peak_loss_feature_indices": torch.tensor(peak_loss_feature_indices, device=device, dtype=torch.long),
        "derived_feature_indices": torch.tensor(derived_feature_indices, device=device, dtype=torch.long),
        "derived_loss_feature_indices": torch.tensor(derived_loss_feature_indices, device=device, dtype=torch.long),
        "distillation_feature_indices": torch.tensor(distillation_feature_indices, device=device, dtype=torch.long),
        "distillation_relation_feature_indices": torch.tensor(
            distillation_relation_feature_indices, device=device, dtype=torch.long
        ),
        "gas_exchange_feature_indices": torch.tensor(
            tuple(target_feature_columns.index(name) for name in ("VO2", "VCO2") if name in target_feature_columns),
            device=device,
            dtype=torch.long,
        ),
        "target_feature_mean": target_feature_mean,
        "target_feature_std": target_feature_std,
        "body_mass_kg": body_mass_kg,
        "initial_boundary_steps": torch.tensor(
            int((deterministic_config or {}).get("initial_boundary_steps", 8)), device=device
        ),
        "peak_timing_beta": torch.tensor(
            float((deterministic_config or {}).get("peak_timing_beta", 10.0)), device=device
        ),
        "core_at_bandwidth": torch.tensor(
            float((deterministic_config or {}).get("core_at_bandwidth", 0.12)), device=device
        ),
        "core_at_weight": torch.tensor(
            float((deterministic_config or {}).get("core_at_weight", 1.0)), device=device
        ),
        "core_post_at_weight": torch.tensor(
            float((deterministic_config or {}).get("core_post_at_weight", 0.5)), device=device
        ),
        "gas_exchange_min_vo2": torch.tensor(
            float((deterministic_config or {}).get("derived_min_vo2", 50.0)), device=device
        ),
        "gas_exchange_min_vco2": torch.tensor(
            float((deterministic_config or {}).get("derived_min_vco2", 50.0)), device=device
        ),
        "gas_exchange_at_bandwidth": torch.tensor(
            float((deterministic_config or {}).get("gas_exchange_at_bandwidth", 0.1)), device=device
        ),
        "gas_exchange_at_weight": torch.tensor(
            float((deterministic_config or {}).get("gas_exchange_at_weight", 2.0)), device=device
        ),
        "gas_exchange_post_at_weight": torch.tensor(
            float((deterministic_config or {}).get("gas_exchange_post_at_weight", 2.0)), device=device
        ),
        "distillation_at_bandwidth": torch.tensor(
            float((deterministic_config or {}).get("distillation_at_bandwidth", 0.10)), device=device
        ),
        "selective_distillation_margin": torch.tensor(
            float((deterministic_config or {}).get("selective_distillation_margin", 0.0)), device=device
        ),
        "selective_at_weight": torch.tensor(
            float((deterministic_config or {}).get("selective_at_weight", 1.0)), device=device
        ),
        "selective_at_region_weight": torch.tensor(
            float((deterministic_config or {}).get("selective_at_region_weight", 0.25)), device=device
        ),
        "selective_relation_weight": torch.tensor(
            float((deterministic_config or {}).get("selective_relation_weight", 0.25)), device=device
        ),
    }
    if "time_fraction" in batch:
        student_batch["time_fraction"] = _resize_steps(batch["time_fraction"], wearable_x.shape[1])
    if "sample_token" in batch:
        student_batch["sample_token"] = batch["sample_token"]
    if "fixed_wearable_x" in batch:
        from cpet_foundation.data.matched_inputs import INPUT_FIELDS
        for name in INPUT_FIELDS:
            fixed = batch[f"fixed_{name}"]
            if name not in student_batch or student_batch[name].shape != fixed.shape:
                raise ValueError(f"fixed input shape mismatch: {name}")
            student_batch[name] = fixed
    teacher_endpoint_mask_value = 1.0 if teacher_model is not None else 0.0
    if "at_norm" in teacher_outputs:
        student_batch["teacher_at_norm"] = teacher_outputs["at_norm"].detach()
        student_batch["teacher_at_mask"] = (
            torch.full_like(teacher_outputs["at_norm"], teacher_endpoint_mask_value)
            * batch.get("at_supervision_mask", torch.ones_like(teacher_outputs["at_norm"]))
        )
    if "vo2peak" in teacher_outputs:
        student_batch["teacher_vo2peak"] = teacher_outputs["vo2peak"].detach()
        student_batch["teacher_vo2peak_mask"] = torch.full_like(
            teacher_outputs["vo2peak"], teacher_endpoint_mask_value
        )
    normalized_supervision_mode = supervision_mode.strip().lower()
    if normalized_supervision_mode == "teacher_only":
        if "probe_fold" in batch:
            student_batch["probe_fold"] = batch["probe_fold"].detach()
        student_batch.pop("target_trajectory", None)
        student_batch.pop("target_channel_mask", None)
        student_batch.pop("core_derived_confidence", None)
        student_batch.pop("at_norm", None)
        student_batch.pop("vo2peak", None)
        student_batch.pop("vo2peak_mask", None)
        # These source-label eligibility fields are not used by teacher-only
        # losses; the curated teacher_at_mask already carries the AT policy.
        student_batch.pop("at_supervision_mask", None)
        student_batch.pop("at_metric_mask", None)
        student_batch.pop("trajectory_supervision_mask", None)
    elif normalized_supervision_mode != "hybrid":
        raise ValueError(f"unsupported supervision_mode: {supervision_mode!r}")
    return student_batch


def _resize_phase(phase: torch.Tensor, steps: int) -> torch.Tensor:
    if phase.shape[1] == steps:
        return phase
    positions = torch.linspace(0, phase.shape[1] - 1, steps, device=phase.device).round().long()
    return phase.index_select(1, positions)


def _resize_steps(values: torch.Tensor, steps: int) -> torch.Tensor:
    if values.shape[1] == steps:
        return values
    positions = torch.linspace(0, values.shape[1] - 1, steps, device=values.device).round().long()
    return values.index_select(1, positions)


def _row_sequence_sample(batch: dict[str, torch.Tensor], row: int) -> dict[str, torch.Tensor]:
    time_mask = batch["mask"][row].detach().cpu()
    if time_mask.ndim == 2:
        time_mask = time_mask.gt(0).any(dim=-1)
    source_mask = batch.get("feature_mask", batch["mask"])[row].detach().cpu()
    source_mask = source_mask * (time_mask.unsqueeze(-1) if source_mask.ndim == 2 else time_mask)
    sample = {
        "x": batch["x"][row].detach().cpu(),
        "mask": source_mask,
        "time_mask": time_mask,
    }
    for key in ("feature_mean", "feature_std", "phase", "phase_mask"):
        if key in batch:
            sample[key] = batch[key][row].detach().cpu()
    return sample


def _deterministic_online_prefix_fraction(row: int, min_fraction: float, max_fraction: float) -> float:
    low = max(0.05, min(0.95, min(float(min_fraction), float(max_fraction))))
    high = max(0.05, min(0.95, max(float(min_fraction), float(max_fraction))))
    if abs(high - low) <= 1e-6:
        return low
    return low + (high - low) * ((row % 7) / 6.0)


def _future_mask(mask: torch.Tensor, prefix_fraction: float) -> torch.Tensor:
    steps = mask.shape[1]
    prefix_steps = max(2, min(steps, int(round(steps * prefix_fraction))))
    future = mask.clone()
    future[:, :prefix_steps] = 0.0
    return future


def _future_mask_from_prefix_steps(mask: torch.Tensor, prefix_steps_by_row: list[int]) -> torch.Tensor:
    future = mask.clone()
    for row, prefix_steps in enumerate(prefix_steps_by_row):
        prefix_steps = max(0, min(future.shape[1], int(prefix_steps)))
        future[row, :prefix_steps] = 0.0
    return future


@torch.no_grad()
def _apply_evaluation_input_perturbation(
    student_batch: dict[str, torch.Tensor],
    mode: str,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Apply a fixed-model diagnostic perturbation to student-visible signals."""
    if mode == "none":
        return student_batch
    if mode == "signal_masked":
        return {
            **student_batch,
            "wearable_x": torch.zeros_like(student_batch["wearable_x"]),
            "wearable_mask": torch.zeros_like(student_batch["wearable_mask"]),
        }
    if mode == "signal_shuffled":
        batch_size = int(student_batch["wearable_x"].shape[0])
        if batch_size <= 1:
            return student_batch
        shift = 1 + int(seed) % (batch_size - 1)
        return {
            **student_batch,
            "wearable_x": torch.roll(student_batch["wearable_x"], shifts=shift, dims=0),
            "wearable_mask": torch.roll(student_batch["wearable_mask"], shifts=shift, dims=0),
        }
    raise ValueError(f"Unknown evaluation input perturbation: {mode}")


def evaluate_wearable_student(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module | None,
    loader: DataLoader,
    feature_columns: tuple[str, ...],
    target_indices: tuple[int, ...],
    shape_feature_indices: tuple[int, ...],
    shape_loss_prefix_fraction: float,
    simulation_config: WearableSimulationConfig,
    device: torch.device,
    *,
    task_mode: str | None = None,
    online_safe_training: bool = False,
    online_prefix_min_fraction: float = 0.35,
    online_prefix_max_fraction: float = 0.75,
    online_action_key: str = "continue_ramp",
    training_config: dict | None = None,
    prediction_output_path: Path | None = None,
    input_perturbation: str = "none",
    perturbation_seed: int = 42,
) -> dict[str, object]:
    resolved_task_mode = resolve_wearable_task_mode(
        {"task_mode": task_mode} if task_mode is not None else {"online_safe_training": online_safe_training}
    )
    model.eval()
    if teacher_model is not None:
        teacher_model.eval()
    at_pred = []
    at_target = []
    at_time_target = []
    at_time_start = []
    at_duration = []
    at_metric_mask = []
    vo2_pred = []
    vo2_target = []
    vo2_mask = []
    traj_rmse = []
    task_traj_rmse = []
    teacher_rmse = []
    task_teacher_rmse = []
    trajectory_pred = []
    trajectory_target = []
    trajectory_masks = []
    task_trajectory_masks = []
    trajectory_channel_masks = []
    task_trajectory_channel_masks = []
    sample_indices = []
    target_feature_mean = []
    target_feature_std = []
    runtime_config = dict(training_config or {})
    target_feature_columns = tuple(feature_columns[index] for index in target_indices)
    metadata_columns = tuple(runtime_config.get("metadata_columns", ()))
    body_mass_metadata_index = metadata_columns.index("Weight_kg") if "Weight_kg" in metadata_columns else None
    deterministic_outputs = bool(runtime_config.get("deterministic_derived_channels", False))
    base_feature_columns = _deterministic_base_feature_columns(runtime_config) if deterministic_outputs else target_feature_columns
    distillation_feature_indices = tuple(target_feature_columns.index(name) for name in base_feature_columns)
    derived_loss_feature_indices = tuple(
        target_feature_columns.index(name) for name in DERIVED_CPET_FEATURES if name in target_feature_columns
    )
    output_mean = getattr(model, "output_feature_mean", torch.zeros(len(target_feature_columns), device=device)).detach()
    output_std = getattr(model, "output_feature_std", torch.ones(len(target_feature_columns), device=device)).detach()
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        student_batch = build_wearable_student_batch(
            batch,
            feature_columns=feature_columns,
            target_indices=target_indices,
            shape_feature_indices=shape_feature_indices,
            derived_loss_feature_indices=derived_loss_feature_indices,
            distillation_feature_indices=distillation_feature_indices,
            body_mass_metadata_index=body_mass_metadata_index,
            target_feature_columns=target_feature_columns,
            output_feature_mean=output_mean,
            output_feature_std=output_std,
            deterministic_config=runtime_config,
            shape_loss_prefix_fraction=shape_loss_prefix_fraction,
            simulation_config=simulation_config,
            teacher_model=teacher_model,
            device=device,
            task_mode=resolved_task_mode,
            online_safe_training=online_safe_training,
            online_prefix_min_fraction=online_prefix_min_fraction,
            online_prefix_max_fraction=online_prefix_max_fraction,
            online_action_key=online_action_key,
        )
        student_batch = _apply_evaluation_input_perturbation(
            student_batch,
            input_perturbation,
            seed=perturbation_seed,
        )
        outputs = model(
            student_batch["wearable_x"],
            student_batch["metadata"],
            student_batch["wearable_mask"],
            student_batch["mask"],
            student_batch["phase"],
            protocol_x=student_batch["protocol_x"],
            protocol_mask=student_batch["protocol_mask"],
            phase_mask=student_batch.get("phase_mask"),
            metadata_mask=student_batch.get("metadata_mask"),
            body_mass_kg=student_batch["body_mass_kg"],
        )
        at_pred.append(outputs["at_norm"].detach().cpu().numpy())
        at_target.append(batch["at_norm"].detach().cpu().numpy())
        at_time_target.append(batch["at_time_s"].detach().cpu().numpy())
        at_time_start.append(batch["time_start_s"].detach().cpu().numpy())
        at_duration.append(batch["duration_s"].detach().cpu().numpy())
        at_metric_mask.append(
            batch.get("at_metric_mask", torch.ones_like(batch["at_norm"])).detach().cpu().numpy()
        )
        vo2_pred.append(outputs["vo2peak"].detach().cpu().numpy())
        vo2_target.append(batch["vo2peak"].detach().cpu().numpy())
        vo2_mask.append(batch.get("vo2peak_mask", torch.ones_like(batch["vo2peak"])).detach().cpu().numpy())
        mask_np = student_batch["mask"].detach().cpu().numpy()
        task_mask_np = student_batch["task_evaluation_mask"].detach().cpu().numpy()
        channel_mask_np = student_batch["target_channel_mask"].detach().cpu().numpy() * mask_np[..., None]
        task_channel_mask_np = student_batch["target_channel_mask"].detach().cpu().numpy() * task_mask_np[..., None]
        pred_np = outputs["trajectory_mu"].detach().cpu().numpy()
        target_np = student_batch["target_trajectory"].detach().cpu().numpy()
        trajectory_pred.append(pred_np)
        trajectory_target.append(target_np)
        trajectory_masks.append(mask_np)
        task_trajectory_masks.append(task_mask_np)
        trajectory_channel_masks.append(channel_mask_np)
        task_trajectory_channel_masks.append(task_channel_mask_np)
        if prediction_output_path is not None:
            sample_indices.append(batch["sample_index"].detach().cpu().numpy())
            target_feature_mean.append(student_batch["target_feature_mean"].detach().cpu().numpy())
            target_feature_std.append(student_batch["target_feature_std"].detach().cpu().numpy())
        traj_rmse.append(_masked_channel_rmse(pred_np, target_np, channel_mask_np))
        task_traj_rmse.append(_masked_channel_rmse(pred_np, target_np, task_channel_mask_np))
        teacher_np = student_batch["teacher_trajectory_mu"].detach().cpu().numpy()
        if teacher_model is not None:
            teacher_rmse.append(_masked_channel_rmse(pred_np, teacher_np, channel_mask_np))
            task_teacher_rmse.append(_masked_channel_rmse(pred_np, teacher_np, task_channel_mask_np))
    metrics = {}
    at_pred_np = np.concatenate(at_pred)
    at_metric_mask_np = np.concatenate(at_metric_mask) > 0
    metrics["at_evaluable_count"] = int(at_metric_mask_np.sum())
    metrics.update(
        regression_metrics(
            at_pred_np[at_metric_mask_np],
            np.concatenate(at_target)[at_metric_mask_np],
            "at_norm",
        )
    )
    metrics.update(
        at_time_metrics(
            at_pred_np,
            np.concatenate(at_time_target),
            np.concatenate(at_time_start),
            np.concatenate(at_duration),
            eligibility=at_metric_mask_np,
        )
    )
    vo2_valid = np.concatenate(vo2_mask) > 0
    metrics["vo2peak_evaluable_count"] = int(vo2_valid.sum())
    if vo2_valid.any():
        metrics.update(
            regression_metrics(
                np.concatenate(vo2_pred)[vo2_valid],
                np.concatenate(vo2_target)[vo2_valid],
                "vo2peak",
            )
        )
    else:
        metrics.update({"vo2peak_mae": float("nan"), "vo2peak_bias": float("nan"), "vo2peak_rmse": float("nan")})
    metrics["trajectory_rmse"] = float(np.mean(traj_rmse))
    metrics["task_trajectory_rmse"] = float(np.mean(task_traj_rmse))
    metrics["teacher_reference_available"] = teacher_model is not None
    metrics["teacher_student_trajectory_rmse"] = float(np.mean(teacher_rmse)) if teacher_rmse else float("nan")
    metrics["task_teacher_student_trajectory_rmse"] = (
        float(np.mean(task_teacher_rmse)) if task_teacher_rmse else float("nan")
    )
    metrics["task_mode"] = resolved_task_mode
    metrics["input_perturbation"] = input_perturbation
    metrics["perturbation_seed"] = int(perturbation_seed)
    metrics["task_metric_region"] = "full_valid_sequence" if resolved_task_mode == "full_reconstruction" else "unobserved_suffix"
    target_names = tuple(feature_columns[index] for index in target_indices)
    all_pred = np.concatenate(trajectory_pred)
    all_target = np.concatenate(trajectory_target)
    all_channel_mask = np.concatenate(trajectory_channel_masks)
    all_task_channel_mask = np.concatenate(task_trajectory_channel_masks)
    reconstruction_indices = reconstruction_feature_indices(runtime_config, target_names)
    primary_indices = reconstruction_feature_indices(runtime_config, target_names, primary=True)
    if reconstruction_indices:
        metrics["reconstruction_trajectory_rmse"] = _masked_channel_rmse(
            all_pred[..., reconstruction_indices],
            all_target[..., reconstruction_indices],
            all_channel_mask[..., reconstruction_indices],
        )
        metrics["task_reconstruction_trajectory_rmse"] = _masked_channel_rmse(
            all_pred[..., reconstruction_indices],
            all_target[..., reconstruction_indices],
            all_task_channel_mask[..., reconstruction_indices],
        )
    else:
        metrics["reconstruction_trajectory_rmse"] = float("nan")
        metrics["task_reconstruction_trajectory_rmse"] = float("nan")
    if primary_indices:
        metrics["primary_trajectory_rmse"] = _masked_channel_rmse(
            all_pred[..., primary_indices],
            all_target[..., primary_indices],
            all_channel_mask[..., primary_indices],
        )
        metrics["task_primary_trajectory_rmse"] = _masked_channel_rmse(
            all_pred[..., primary_indices],
            all_target[..., primary_indices],
            all_task_channel_mask[..., primary_indices],
        )
    else:
        metrics["primary_trajectory_rmse"] = float("nan")
        metrics["task_primary_trajectory_rmse"] = float("nan")
    metrics["reconstruction_metric_feature_columns"] = [target_names[index] for index in reconstruction_indices]
    metrics["primary_reconstruction_feature_columns"] = [target_names[index] for index in primary_indices]
    metrics["trajectory_channel_metrics"] = _channel_trajectory_metrics(
        all_pred,
        all_target,
        all_channel_mask,
        target_names,
    )
    metrics["task_trajectory_channel_metrics"] = _channel_trajectory_metrics(
        all_pred,
        all_target,
        all_task_channel_mask,
        target_names,
    )
    metrics["real_paired_validation_available"] = 0.0
    if prediction_output_path is not None:
        indices_np = np.concatenate(sample_indices).astype(int)
        dataset_samples = getattr(loader.dataset, "samples", None)
        if dataset_samples is None:
            raise ValueError("prediction export requires a dataset with sample metadata")
        prediction_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_output_path,
            sample_index=indices_np,
            exam_id=np.asarray([dataset_samples[index]["exam_id"] for index in indices_np]),
            institute=np.asarray([dataset_samples[index]["institute"] for index in indices_np]),
            channel_names=np.asarray(target_names),
            trajectory_pred=np.concatenate(trajectory_pred).astype(np.float32),
            trajectory_target=np.concatenate(trajectory_target).astype(np.float32),
            valid_mask=np.concatenate(trajectory_masks).astype(np.float32),
            task_mask=np.concatenate(task_trajectory_masks).astype(np.float32),
            target_channel_mask=np.concatenate(trajectory_channel_masks).astype(np.float32),
            feature_mean=np.concatenate(target_feature_mean).astype(np.float32),
            feature_std=np.concatenate(target_feature_std).astype(np.float32),
            at_pred_norm=at_pred_np.astype(np.float32),
            at_target_norm=np.concatenate(at_target).astype(np.float32),
            at_target_s=np.concatenate(at_time_target).astype(np.float32),
            at_metric_mask=at_metric_mask_np.astype(np.float32),
            time_start_s=np.concatenate(at_time_start).astype(np.float32),
            duration_s=np.concatenate(at_duration).astype(np.float32),
            vo2peak_pred=np.concatenate(vo2_pred).astype(np.float32),
            vo2peak_target=np.concatenate(vo2_target).astype(np.float32),
            vo2peak_mask=np.concatenate(vo2_mask).astype(np.float32),
        )
    return metrics


def _channel_trajectory_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    channel_names: tuple[str, ...],
) -> dict[str, dict[str, float | int]]:
    values: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(channel_names):
        raw_mask = np.asarray(mask, dtype=float)
        valid_steps = raw_mask[..., index] > 0 if raw_mask.ndim == 3 else raw_mask > 0
        channel_pred = np.asarray(pred[..., index], dtype=float)[valid_steps]
        channel_target = np.asarray(target[..., index], dtype=float)[valid_steps]
        finite = np.isfinite(channel_target)
        if np.any(finite & ~np.isfinite(channel_pred)):
            raise FloatingPointError(f"Non-finite prediction at eligible trajectory target: {name}")
        channel_pred = channel_pred[finite]
        channel_target = channel_target[finite]
        if channel_pred.size == 0:
            values[name] = {"rmse": float("nan"), "mae": float("nan"), "pearson_r": float("nan"), "n": 0}
            continue
        pearson = float("nan")
        if channel_pred.size > 1 and np.std(channel_pred) > 0 and np.std(channel_target) > 0:
            pearson = float(np.corrcoef(channel_pred, channel_target)[0, 1])
        delta = channel_pred - channel_target
        values[name] = {
            "rmse": float(np.sqrt(np.mean(delta**2))),
            "mae": float(np.mean(np.abs(delta))),
            "pearson_r": pearson,
            "n": int(channel_pred.size),
        }
    return values


def _masked_channel_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = (np.asarray(mask) > 0) & np.isfinite(target)
    if np.any(valid & ~np.isfinite(pred)):
        raise FloatingPointError("Non-finite prediction at an eligible trajectory target")
    if not valid.any():
        return float("nan")
    delta = np.asarray(pred, dtype=float) - np.asarray(target, dtype=float)
    return float(np.sqrt(np.mean(delta[valid] ** 2)))


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    feature_index: dict[str, int],
    device: torch.device,
    protocol_indices: tuple[int, ...] = (),
    input_mode: str = "prefix",
) -> dict[str, float]:
    model.eval()
    at_pred = []
    at_target = []
    at_metric_mask = []
    at_time_target = []
    at_time_start = []
    at_duration = []
    vo2_pred = []
    vo2_target = []
    vo2_mask = []
    traj_rmse = []
    core_squared_error = 0.0
    core_count = 0.0
    core_indices = [feature_index[name] for name in ("VO2", "VCO2", "VE") if name in feature_index]
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        if input_mode == "prefix":
            batch = apply_prefix_mask(batch, min_fraction=0.6, max_fraction=0.6, protocol_indices=protocol_indices)
        else:
            batch = prepare_world_model_input(batch, mode=input_mode, protocol_indices=protocol_indices)
        outputs = model(
            batch["x_input"], batch["metadata"], batch["mask_input"], batch["phase"],
            feature_mask=batch["feature_mask_input"],
            phase_mask=batch.get("phase_mask"),
        )
        at_pred.append(outputs["at_norm"].detach().cpu().numpy())
        at_target.append(batch["at_norm"].detach().cpu().numpy())
        at_metric_mask.append(
            batch.get("at_metric_mask", torch.ones_like(batch["at_norm"])).detach().cpu().numpy()
        )
        if all(key in batch for key in ("at_time_s", "time_start_s", "duration_s")):
            at_time_target.append(batch["at_time_s"].detach().cpu().numpy())
            at_time_start.append(batch["time_start_s"].detach().cpu().numpy())
            at_duration.append(batch["duration_s"].detach().cpu().numpy())
        vo2_pred.append(outputs["vo2peak"].detach().cpu().numpy())
        vo2_target.append(batch["vo2peak"].detach().cpu().numpy())
        vo2_mask.append(batch.get("vo2peak_mask", torch.ones_like(batch["vo2peak"])).detach().cpu().numpy())
        trajectory_mask = batch["mask"]
        if "trajectory_supervision_mask" in batch:
            trajectory_mask = trajectory_mask * batch["trajectory_supervision_mask"].reshape(-1, 1)
        if len(core_indices) == 3:
            core_delta = outputs["trajectory_mu"][..., core_indices] - batch["x"][..., core_indices]
            if "feature_mask" in batch:
                core_weights = trajectory_mask.unsqueeze(-1) * batch["feature_mask"][..., core_indices]
            else:
                core_weights = trajectory_mask.unsqueeze(-1).expand_as(core_delta)
            core_squared_error += float((core_delta.square() * core_weights).sum().cpu())
            core_count += float(core_weights.sum().cpu())
        traj_rmse.append(
            trajectory_metrics(
                outputs["trajectory_mu"].detach().cpu().numpy(),
                batch["x"].detach().cpu().numpy(),
                trajectory_mask.detach().cpu().numpy(),
            )["trajectory_rmse"]
        )
    metrics = {}
    at_valid = np.concatenate(at_metric_mask) > 0
    metrics["at_evaluable_count"] = int(at_valid.sum())
    metrics.update(
        regression_metrics(
            np.concatenate(at_pred)[at_valid],
            np.concatenate(at_target)[at_valid],
            "at_norm",
        )
    )
    if at_time_target:
        metrics.update(at_time_metrics(
            np.concatenate(at_pred), np.concatenate(at_time_target),
            np.concatenate(at_time_start), np.concatenate(at_duration),
            eligibility=np.concatenate(at_metric_mask),
        ))
    vo2_valid = np.concatenate(vo2_mask) > 0
    metrics["vo2peak_evaluable_count"] = int(vo2_valid.sum())
    if vo2_valid.any():
        metrics.update(
            regression_metrics(
                np.concatenate(vo2_pred)[vo2_valid],
                np.concatenate(vo2_target)[vo2_valid],
                "vo2peak",
            )
        )
    else:
        metrics.update({"vo2peak_mae": float("nan"), "vo2peak_bias": float("nan"), "vo2peak_rmse": float("nan")})
    metrics["trajectory_rmse"] = float(np.mean(traj_rmse))
    if len(core_indices) == 3:
        metrics["core_trajectory_rmse"] = float(np.sqrt(core_squared_error / max(core_count, 1.0)))
    return metrics
