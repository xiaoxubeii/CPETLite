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
from cpet_foundation.model.world_model import build_model
from cpet_foundation.paths import PROJECT_ROOT


SAFETY_NOTICE = (
    "Research-use decision support only. Outputs require clinician review and must not be used as "
    "autonomous diagnosis, treatment, or CPET termination."
)


def load_checkpoint(checkpoint_path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = build_model(
        config,
        n_features=len(checkpoint["feature_columns"]),
        n_metadata=len(checkpoint["metadata_columns"]),
        max_len=int(config["max_len"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def infer_prefix(model: torch.nn.Module, sample: dict[str, torch.Tensor], prefix_fraction: float = 0.6) -> dict[str, Any]:
    x = sample["x"].unsqueeze(0).clone()
    metadata = sample["metadata"].unsqueeze(0).clone()
    mask = sample["mask"].unsqueeze(0).clone()
    steps = x.shape[1]
    prefix_steps = max(2, min(steps, int(round(steps * prefix_fraction))))
    prefix_mask = mask.clone()
    prefix_mask[:, prefix_steps:] = 0.0
    observed_prefix = x.clone()
    observed_prefix[:, prefix_steps:] = 0.0
    protocol_indices = getattr(model, "protocol_feature_indices", ())
    if protocol_indices:
        observed_prefix[:, prefix_steps:, list(protocol_indices)] = x[:, prefix_steps:, list(protocol_indices)]
    phase = sample["phase"].unsqueeze(0)
    outputs = model(observed_prefix, metadata, prefix_mask, phase)
    mu_standardized = outputs["trajectory_mu"].squeeze(0).cpu().numpy()
    sigma_standardized = (0.5 * outputs["trajectory_logvar"].squeeze(0)).exp().cpu().numpy()
    observed_standardized = x.squeeze(0).cpu().numpy()
    mu = denormalize_sequence(mu_standardized, sample)
    sigma = rescale_sigma(sigma_standardized, sample)
    observed = denormalize_sequence(observed_standardized, sample)
    residual = observed - mu
    uncertainty = sigma_standardized.mean(axis=1)
    future_uncertainty = float(np.mean(uncertainty[prefix_steps:])) if prefix_steps < steps else 0.0
    current_uncertainty = float(np.mean(uncertainty[:prefix_steps]))
    information_gain = max(0.0, future_uncertainty - current_uncertainty)
    if future_uncertainty < 0.75 and information_gain < 0.15:
        recommendation = "consider-stop"
    elif future_uncertainty > 1.25:
        recommendation = "continue"
    else:
        recommendation = "insufficient-confidence"
    return {
        "prefix_fraction": prefix_fraction,
        "prefix_steps": prefix_steps,
        "at_norm": float(outputs["at_norm"].item()),
        "vo2peak": float(outputs["vo2peak"].item()),
        "trajectory_mu": mu.tolist(),
        "trajectory_sigma": sigma.tolist(),
        "observed": observed.tolist(),
        "residual": residual.tolist(),
        "uncertainty": uncertainty.tolist(),
        "future_uncertainty": future_uncertainty,
        "information_gain": information_gain,
        "recommendation": recommendation,
        "safety_notice": SAFETY_NOTICE,
    }


def denormalize_sequence(values: np.ndarray, sample: dict[str, torch.Tensor]) -> np.ndarray:
    mean, std = _feature_stats(sample, values.shape[-1])
    return (values * std) + mean


def rescale_sigma(values: np.ndarray, sample: dict[str, torch.Tensor]) -> np.ndarray:
    _mean, std = _feature_stats(sample, values.shape[-1])
    return np.abs(values * std)


def _feature_stats(sample: dict[str, torch.Tensor], n_features: int) -> tuple[np.ndarray, np.ndarray]:
    if "feature_mean" not in sample or "feature_std" not in sample:
        return np.zeros(n_features, dtype=np.float32), np.ones(n_features, dtype=np.float32)
    mean = sample["feature_mean"].detach().cpu().numpy().astype(np.float32, copy=False)
    std = sample["feature_std"].detach().cpu().numpy().astype(np.float32, copy=False)
    if mean.shape[0] != n_features or std.shape[0] != n_features:
        return np.zeros(n_features, dtype=np.float32), np.ones(n_features, dtype=np.float32)
    return mean, np.where(np.abs(std) > 1e-6, std, 1.0).astype(np.float32)


def load_demo_sample(config: dict[str, Any], source_h5: Path, exam_index: int = 0) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
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
        "metadata_columns": list(config["metadata_columns"]),
        "case": {
            "exam_id": str(raw_sample.get("exam_id", f"demo-{exam_index}")),
            "source_institute": str(raw_sample.get("institute", "unavailable")),
            "split": "demo",
            "identifier_policy": "pseudonymized_exam_id",
        },
    }


def write_demo_payload(checkpoint_path: Path, output_path: Path, prefix_fraction: float = 0.6) -> dict[str, Any]:
    model, checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    root = checkpoint_path.resolve().parents[3]
    source_h5 = Path(config["source_h5"])
    if not source_h5.is_absolute():
        source_h5 = (root / source_h5).resolve()
    sample, meta = load_demo_sample(config, source_h5)
    result = infer_prefix(model, sample, prefix_fraction)
    payload = {
        "model": "CPET-World",
        "architecture": "variable-aware masked encoder + controlled latent state-space decoder + probabilistic trajectory head",
        "case": meta["case"],
        "features": meta["feature_columns"],
        "result": result,
        "metrics": checkpoint.get("metrics", {}),
        "safety_notice": SAFETY_NOTICE,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
