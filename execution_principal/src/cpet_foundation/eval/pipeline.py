from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from cpet_foundation.data.sequences import (
    CpetSequenceDataset,
    SequenceConfig,
    ids_from_split,
    normalization_options_from_config,
)
from cpet_foundation.eval.metrics import regression_metrics, trajectory_metrics
from cpet_foundation.inference.digital_twin import load_checkpoint
from cpet_foundation.paths import PROJECT_ROOT
from cpet_foundation.training.runner import apply_prefix_mask


@dataclass(frozen=True)
class LocoFold:
    name: str
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _limit_ids(ids: Iterable[str], limit: int | None) -> tuple[str, ...]:
    values = tuple(map(str, ids))
    if limit is None:
        return values
    return values[:limit]


def load_loco_folds(split_path: Path) -> tuple[LocoFold, ...]:
    split = _read_json(split_path)
    if split.get("policy") != "leave-one-center-out":
        raise ValueError(f"Expected leave-one-center-out split, got {split.get('policy')!r}")
    folds = split.get("folds")
    if not isinstance(folds, dict) or not folds:
        raise ValueError("LOCO split must contain non-empty folds")

    parsed: list[LocoFold] = []
    for name, fold in sorted(folds.items()):
        train_ids = tuple(map(str, fold.get("train", ())))
        test_ids = tuple(map(str, fold.get("test", ())))
        if not train_ids or not test_ids:
            raise ValueError(f"LOCO fold {name!r} must contain train and test ids")
        overlap = set(train_ids) & set(test_ids)
        if overlap:
            example = sorted(overlap)[0]
            raise ValueError(f"LOCO fold {name!r} has train/test overlap, for example {example!r}")
        parsed.append(LocoFold(name=str(name), train_ids=train_ids, test_ids=test_ids))
    return tuple(parsed)


def _resolve_source_h5(config: dict[str, Any]) -> Path:
    source_h5 = Path(config["source_h5"])
    if source_h5.is_absolute():
        return source_h5
    return (PROJECT_ROOT / source_h5).resolve()


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    feature_index: dict[str, int],
    device: torch.device,
    prefix_fraction: float = 0.6,
) -> dict[str, float]:
    model.eval()
    at_pred: list[np.ndarray] = []
    at_target: list[np.ndarray] = []
    at_metric_mask: list[np.ndarray] = []
    vo2_pred: list[np.ndarray] = []
    vo2_target: list[np.ndarray] = []
    vo2_mask: list[np.ndarray] = []
    trajectory_rmse: list[float] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        batch = apply_prefix_mask(batch, min_fraction=prefix_fraction, max_fraction=prefix_fraction)
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
        vo2_pred.append(outputs["vo2peak"].detach().cpu().numpy())
        vo2_target.append(batch["vo2peak"].detach().cpu().numpy())
        vo2_mask.append(batch.get("vo2peak_mask", torch.ones_like(batch["vo2peak"])).detach().cpu().numpy())
        trajectory_mask = batch["mask"]
        if "trajectory_supervision_mask" in batch:
            trajectory_mask = trajectory_mask * batch["trajectory_supervision_mask"].reshape(-1, 1)
        trajectory_rmse.append(
            trajectory_metrics(
                outputs["trajectory_mu"].detach().cpu().numpy(),
                batch["x"].detach().cpu().numpy(),
                trajectory_mask.detach().cpu().numpy(),
            )["trajectory_rmse"]
        )

    metrics: dict[str, float] = {"sample_count": float(sum(len(chunk) for chunk in at_target))}
    if at_pred:
        at_valid = np.concatenate(at_metric_mask) > 0
        metrics["at_evaluable_count"] = float(at_valid.sum())
        metrics.update(
            regression_metrics(
                np.concatenate(at_pred)[at_valid],
                np.concatenate(at_target)[at_valid],
                "at_norm",
            )
        )
        vo2_valid = np.concatenate(vo2_mask) > 0
        metrics["vo2peak_evaluable_count"] = float(vo2_valid.sum())
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
        metrics["trajectory_rmse"] = float(np.mean(trajectory_rmse))
    return metrics


def evaluate_checkpoint_ids(
    checkpoint_path: Path,
    exam_ids: tuple[str, ...],
    prefix_fraction: float = 0.6,
    batch_size: int = 64,
) -> dict[str, float]:
    model, checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_format = str(config.get("input_format", "registry"))
    if input_format == "v5_derivative_split":
        split_name = str(config.get("evaluation_split_name", "val"))
        selected_ids = exam_ids if exam_ids else None
        normalization_options: dict[str, Any] = {}
    else:
        split_name = None
        selected_ids = exam_ids
        normalization_options = normalization_options_from_config(config, PROJECT_ROOT)
    dataset = CpetSequenceDataset(
        SequenceConfig(
            source_h5=_resolve_source_h5(config),
            feature_columns=tuple(checkpoint["feature_columns"]),
            metadata_columns=tuple(checkpoint["metadata_columns"]),
            max_len=int(config["max_len"]),
            exam_ids=selected_ids,
            input_format=input_format,
            split_name=split_name,
            **normalization_options,
        )
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=device.type == "cuda")
    feature_index = {name: idx for idx, name in enumerate(checkpoint["feature_columns"])}
    metrics = evaluate_loader(model, loader, feature_index, device, prefix_fraction=prefix_fraction)
    metrics["requested_count"] = float(len(exam_ids))
    metrics["loaded_count"] = float(len(dataset))
    return metrics


def evaluate_internal_splits(
    checkpoint_path: Path,
    split_path: Path,
    output_path: Path,
    split_names: tuple[str, ...] = ("validation", "test"),
    limit_per_split: int | None = None,
    prefix_fractions: tuple[float, ...] = (0.6,),
) -> dict[str, Any]:
    results: dict[str, Any] = {"status": "complete", "evaluation_type": "internal", "splits": {}}
    for split_name in split_names:
        ids = ids_from_split(split_path, split_name, limit_per_split)
        if not ids:
            results["splits"][split_name] = {"status": "missing_ids"}
            continue
        split_metrics: dict[str, Any] = {}
        for prefix_fraction in prefix_fractions:
            key = f"prefix_{prefix_fraction:.2f}"
            split_metrics[key] = evaluate_checkpoint_ids(checkpoint_path, ids, prefix_fraction=prefix_fraction)
        if len(prefix_fractions) == 1:
            split_metrics.update(split_metrics[f"prefix_{prefix_fractions[0]:.2f}"])
        results["splits"][split_name] = split_metrics
    _write_json(output_path, results)
    return results


def evaluate_loco_checkpoint(
    checkpoint_path: Path,
    split_path: Path,
    output_path: Path,
    limit_per_fold: int | None = None,
    prefix_fraction: float = 0.6,
) -> dict[str, Any]:
    folds = load_loco_folds(split_path)
    results: dict[str, Any] = {
        "status": "complete",
        "evaluation_type": "leave-one-center-out checkpoint evaluation",
        "generalization_note": (
            "This evaluates the current checkpoint on each held-out center. Final clinical "
            "generalization claims still require full fold-specific training and external validation."
        ),
        "folds": {},
    }
    at_mae: list[float] = []
    vo2_mae: list[float] = []
    trajectory: list[float] = []
    for fold in folds:
        test_ids = _limit_ids(fold.test_ids, limit_per_fold)
        metrics = evaluate_checkpoint_ids(checkpoint_path, test_ids, prefix_fraction=prefix_fraction)
        results["folds"][fold.name] = {
            "train_count": len(fold.train_ids),
            "test_count": len(fold.test_ids),
            "evaluated_count": int(metrics.get("loaded_count", 0.0)),
            "metrics": metrics,
        }
        if np.isfinite(metrics.get("at_norm_mae", np.nan)):
            at_mae.append(float(metrics["at_norm_mae"]))
        if np.isfinite(metrics.get("vo2peak_mae", np.nan)):
            vo2_mae.append(float(metrics["vo2peak_mae"]))
        if np.isfinite(metrics.get("trajectory_rmse", np.nan)):
            trajectory.append(float(metrics["trajectory_rmse"]))
    results["summary"] = {
        "fold_count": len(folds),
        "mean_at_norm_mae": float(np.mean(at_mae)) if at_mae else float("nan"),
        "mean_vo2peak_mae": float(np.mean(vo2_mae)) if vo2_mae else float("nan"),
        "mean_trajectory_rmse": float(np.mean(trajectory)) if trajectory else float("nan"),
    }
    _write_json(output_path, results)
    return results
