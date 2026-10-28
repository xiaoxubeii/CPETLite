#!/usr/bin/env python3
"""Create the locked internal/external CPETLite matched-comparison artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/eval/cpetlite_v1_matched_external/comparison.json"
MODELS = {
    "distilled_neural_sde": (
        "wearable_student_enhanced_cpetlite_v1_matched_seed42",
        "distilled",
    ),
    "matched_no_distillation": (
        "wearable_student_enhanced_cpetlite_v1_no_distillation_seed42",
        "no_distillation",
    ),
    "direct_transformer": (
        "direct_transformer_enhanced_cpetlite_v1_seed42",
        "direct_transformer",
    ),
    "direct_tcn": (
        "direct_tcn_enhanced_cpetlite_v1_seed42",
        "direct_tcn",
    ),
}
V2_MODELS = {
    "distilled_neural_sde": (
        "wearable_student_enhanced_cpetlite_v2_matched_seed42",
        "matched_distilled_seed42",
    ),
    "matched_no_distillation": (
        "wearable_student_enhanced_cpetlite_v2_no_distillation_seed42",
        "matched_measured_only_seed42",
    ),
    "direct_transformer": (
        "direct_transformer_enhanced_cpetlite_v2_seed42",
        "direct_transformer_seed42",
    ),
    "direct_tcn": (
        "direct_tcn_enhanced_cpetlite_v2_seed42",
        "direct_tcn_seed42",
    ),
}
PRIMARY = ("VO2", "VCO2", "VE")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_count(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(sum(value.numel() for value in checkpoint["model_state"].values()))


def _case_components(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = np.load(path)
    names = [str(value) for value in data["channel_names"]]
    indices = [names.index(name) for name in PRIMARY]
    prediction = data["trajectory_pred"][:, :, indices].astype(np.float64)
    target = data["trajectory_target"][:, :, indices].astype(np.float64)
    mask = (
        data["task_mask"][:, :, None]
        * data["target_channel_mask"][:, :, indices]
    ).astype(np.float64)
    squared_error = np.square(prediction - target) * mask
    at_pred_s = data["time_start_s"] + data["at_pred_norm"] * data["duration_s"]
    components = {
        "core_sse": squared_error.sum(axis=(1, 2)),
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
    return data["exam_id"].astype(str), components


def _metric(components: dict[str, np.ndarray], indices: np.ndarray | None = None) -> np.ndarray:
    select = slice(None) if indices is None else indices
    at_mask = components["at_metric_mask"][select]
    return np.array(
        [
            np.sqrt(components["core_sse"][select].sum() / components["core_n"][select].sum()),
            (components["at_abs_error_s"][select] * at_mask).sum() / at_mask.sum(),
            (components["peak_abs_error"][select] * components["peak_mask"][select]).sum()
            / components["peak_mask"][select].sum(),
        ],
        dtype=np.float64,
    )


def _paired_bootstrap(
    components: dict[str, dict[str, np.ndarray]],
    reference: str,
    *,
    draws: int = 2000,
    seed: int = 42,
    difference_field: str = "difference_vs_reference",
) -> dict[str, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    n = len(next(iter(components.values()))["core_sse"])
    names = ("primary_trajectory_rmse", "at_time_s_mae", "vo2peak_mae")
    result: dict[str, dict[str, dict[str, float]]] = {}
    reference_point = _metric(components[reference])
    for role, values in components.items():
        if role == reference:
            continue
        point = _metric(values) - reference_point
        samples = np.empty((draws, 3), dtype=np.float64)
        for draw in range(draws):
            indices = rng.integers(0, n, size=n)
            samples[draw] = _metric(values, indices) - _metric(components[reference], indices)
        result[role] = {
            name: {
                difference_field: float(point[index]),
                "ci95_low": float(np.quantile(samples[:, index], 0.025)),
                "ci95_high": float(np.quantile(samples[:, index], 0.975)),
                "lower_is_better": True,
            }
            for index, name in enumerate(names)
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    is_v2 = args.version == "v2"
    models = V2_MODELS if is_v2 else MODELS
    internal_n = 1088 if is_v2 else 1740
    evaluation_dir = ROOT / ("artifacts/eval/cpetlite_v2" if is_v2 else "artifacts/eval/cpetlite_v1_matched_external")
    output = args.output.resolve() if args.output else (
        evaluation_dir / ("matched_comparison.json" if is_v2 else "comparison.json")
    )
    report: dict[str, object] = {
        "comparison_contract": {
            "seed": 42,
            "internal_validation_n": internal_n,
            "external_liuyuan_n": 2224,
            "primary_channels": list(PRIMARY),
            "bootstrap_draws": 2000,
            "bootstrap_unit": "examination",
            "difference_orientation": "model minus direct_transformer; lower is better",
        },
        "models": {},
    }
    components_by_domain: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "internal": {},
        "external": {},
    }
    ids_by_domain: dict[str, np.ndarray] = {}
    for role, (model_dir_name, external_name) in models.items():
        model_dir = ROOT / "artifacts/models" / model_dir_name
        external_dir = evaluation_dir
        internal_metrics = _read_json(
            external_dir / f"{external_name}_internal_metrics.json" if is_v2 else model_dir / "metrics.json"
        )
        external_metrics = _read_json(
            external_dir / f"{external_name}_liuyuan_metrics.json" if is_v2 else external_dir / f"{external_name}_metrics.json"
        )
        manifest = _read_json(model_dir / "manifest.json")
        external_manifest = _read_json(
            external_dir / f"{external_name}_liuyuan_manifest.json" if is_v2 else external_dir / f"{external_name}_manifest.json"
        )
        checkpoint_path = model_dir / "model.pt"
        report["models"][role] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "parameter_count": _parameter_count(checkpoint_path),
            "training_control": manifest.get("training_control"),
            "temporal_backbone_type": manifest.get("temporal_backbone_type", "transformer"),
            "rollout_type": manifest.get("rollout_type"),
            "distillation_weights": {
                key: manifest.get("loss_weights", {}).get(key)
                for key in ("latent_alignment", "trajectory_distillation", "uncertainty_distillation")
            },
            "internal": {
                key: internal_metrics[key]
                for key in ("primary_trajectory_rmse", "reconstruction_trajectory_rmse", "at_time_s_mae", "vo2peak_mae")
            },
            "external": {
                key: external_metrics[key]
                for key in ("primary_trajectory_rmse", "reconstruction_trajectory_rmse", "at_time_s_mae", "vo2peak_mae")
            },
            "split_sha256": external_manifest["split_sha256"],
            "normalization_stats_sha256": external_manifest["normalization_stats_sha256"],
        }
        prediction_paths = {
            "internal": (
                external_dir / f"{external_name}_internal_predictions.npz"
                if is_v2
                else model_dir / "validation_predictions.npz"
            ),
            "external": (
                external_dir / f"{external_name}_liuyuan_predictions.npz"
                if is_v2
                else external_dir / f"{external_name}_predictions.npz"
            ),
        }
        for domain, prediction_path in prediction_paths.items():
            exam_ids, components = _case_components(prediction_path)
            if domain in ids_by_domain and not np.array_equal(ids_by_domain[domain], exam_ids):
                raise ValueError(f"{domain} examination order differs for {role}")
            ids_by_domain.setdefault(domain, exam_ids)
            components_by_domain[domain][role] = components
    report["paired_bootstrap"] = {
        domain: _paired_bootstrap(
            components,
            "direct_transformer",
            difference_field="difference_vs_direct_transformer",
        )
        for domain, components in components_by_domain.items()
    }
    at_counts = {
        domain: {
            role: int(values["at_metric_mask"].sum())
            for role, values in components.items()
        }
        for domain, components in components_by_domain.items()
    }
    for domain, counts in at_counts.items():
        if len(set(counts.values())) != 1:
            raise ValueError(f"{domain} AT eligibility differs across matched models: {counts}")
    report["comparison_contract"]["at_evaluable_n"] = {
        domain: next(iter(counts.values())) for domain, counts in at_counts.items()
    }
    report["distillation_effect"] = {
        domain: _paired_bootstrap(
            components,
            "matched_no_distillation",
            difference_field="difference_distilled_minus_matched_no_distillation",
        )["distilled_neural_sde"]
        for domain, components in components_by_domain.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
