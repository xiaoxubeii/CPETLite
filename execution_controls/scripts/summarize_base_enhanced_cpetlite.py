#!/usr/bin/env python3
"""Summarize the v2 base versus enhanced input boundary with paired bootstrap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts/eval/cpetlite_v2"
OUTPUT = EVAL / "base_enhanced_comparison.json"
PRIMARY = ("VO2", "VCO2", "VE")
RECONSTRUCTION = (
    "VO2", "VO2_kg", "VCO2", "VCO2_kg", "VE", "RER",
    "VE_VO2", "VE_VCO2", "METS", "VO2_HR", "PetO2", "PetCO2",
)
METRICS = (
    "core_rmse", "reconstruction_rmse", "at_mae_s", "at_bias_s",
    "vo2peak_mae", "rer_rmse", "rer_r",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _components(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = np.load(path)
    names = [str(value) for value in data["channel_names"]]
    primary = [names.index(name) for name in PRIMARY]
    reconstruction = [names.index(name) for name in RECONSTRUCTION]
    rer = names.index("RER")
    prediction = data["trajectory_pred"].astype(np.float64)
    target = data["trajectory_target"].astype(np.float64)
    task = data["task_mask"].astype(np.float64)
    channel_mask = data["target_channel_mask"].astype(np.float64)

    def error_terms(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        mask = task[:, :, None] * channel_mask[:, :, indices]
        error = np.where(mask > 0, prediction[:, :, indices] - target[:, :, indices], 0.0)
        return np.square(error).sum(axis=(1, 2)), mask.sum(axis=(1, 2))

    core_sse, core_n = error_terms(primary)
    reconstruction_sse, reconstruction_n = error_terms(reconstruction)
    rer_mask = task * channel_mask[:, :, rer]
    x = prediction[:, :, rer]
    y = target[:, :, rer]
    x_valid = np.where(rer_mask > 0, x, 0.0)
    y_valid = np.where(rer_mask > 0, y, 0.0)
    # Materialize RER sufficient statistics before arithmetic on temporary arrays.
    rer_x = x_valid.sum(axis=1)
    rer_y = y_valid.sum(axis=1)
    rer_x2 = np.einsum("ij,ij->i", x_valid, x_valid)
    rer_y2 = np.einsum("ij,ij->i", y_valid, y_valid)
    rer_xy = np.einsum("ij,ij->i", x_valid, y_valid)
    rer_error = np.subtract(x_valid, y_valid)
    rer_sse = np.einsum("ij,ij->i", rer_error, rer_error)
    at_pred_s = data["time_start_s"] + data["at_pred_norm"] * data["duration_s"]
    at_error = at_pred_s.astype(np.float64) - data["at_target_s"].astype(np.float64)
    return data["exam_id"].astype(str), {
        "core_sse": core_sse,
        "core_n": core_n,
        "reconstruction_sse": reconstruction_sse,
        "reconstruction_n": reconstruction_n,
        "at_error_s": at_error,
        "at_metric_mask": data["at_metric_mask"].astype(np.float64),
        "peak_abs_error": np.abs(data["vo2peak_pred"] - data["vo2peak_target"]).astype(np.float64),
        "peak_mask": data["vo2peak_mask"].astype(np.float64),
        "rer_n": rer_mask.sum(axis=1),
        "rer_sse": rer_sse,
        "rer_x": rer_x,
        "rer_y": rer_y,
        "rer_x2": rer_x2,
        "rer_y2": rer_y2,
        "rer_xy": rer_xy,
    }


def _metrics(values: dict[str, np.ndarray], indices: np.ndarray | None = None) -> dict[str, float]:
    selected = slice(None) if indices is None else indices
    at_mask = values["at_metric_mask"][selected]
    peak_mask = values["peak_mask"][selected]
    n = values["rer_n"][selected].sum()
    sum_x = values["rer_x"][selected].sum()
    sum_y = values["rer_y"][selected].sum()
    covariance = values["rer_xy"][selected].sum() - sum_x * sum_y / n
    variance_x = values["rer_x2"][selected].sum() - sum_x * sum_x / n
    variance_y = values["rer_y2"][selected].sum() - sum_y * sum_y / n
    return {
        "core_rmse": float(np.sqrt(values["core_sse"][selected].sum() / values["core_n"][selected].sum())),
        "reconstruction_rmse": float(np.sqrt(values["reconstruction_sse"][selected].sum() / values["reconstruction_n"][selected].sum())),
        "at_mae_s": float((np.abs(values["at_error_s"][selected]) * at_mask).sum() / at_mask.sum()),
        "at_bias_s": float((values["at_error_s"][selected] * at_mask).sum() / at_mask.sum()),
        "vo2peak_mae": float((values["peak_abs_error"][selected] * peak_mask).sum() / peak_mask.sum()),
        "rer_rmse": float(np.sqrt(values["rer_sse"][selected].sum() / n)),
        "rer_r": float(covariance / np.sqrt(variance_x * variance_y)),
    }


def _bootstrap(
    base: dict[str, np.ndarray],
    enhanced: dict[str, np.ndarray],
    *,
    draws: int = 2000,
    seed: int = 42,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    rng = np.random.default_rng(seed)
    n = len(base["core_sse"])
    base_point = _metrics(base)
    enhanced_point = _metrics(enhanced)
    base_samples = {metric: np.empty(draws) for metric in METRICS}
    enhanced_samples = {metric: np.empty(draws) for metric in METRICS}
    difference_samples = {metric: np.empty(draws) for metric in METRICS}
    for draw in range(draws):
        indices = rng.integers(0, n, size=n)
        base_draw = _metrics(base, indices)
        enhanced_draw = _metrics(enhanced, indices)
        for metric in METRICS:
            base_samples[metric][draw] = base_draw[metric]
            enhanced_samples[metric][draw] = enhanced_draw[metric]
            difference_samples[metric][draw] = enhanced_draw[metric] - base_draw[metric]

    models = {
        role: {
            metric: {
                "estimate": point[metric],
                "ci95": [float(np.quantile(samples[metric], 0.025)), float(np.quantile(samples[metric], 0.975))],
            }
            for metric in METRICS
        }
        for role, point, samples in (
            ("base", base_point, base_samples),
            ("enhanced", enhanced_point, enhanced_samples),
        )
    }
    differences = {
        metric: {
            "estimate": enhanced_point[metric] - base_point[metric],
            "ci95": [float(np.quantile(difference_samples[metric], 0.025)), float(np.quantile(difference_samples[metric], 0.975))],
        }
        for metric in METRICS
    }
    return models, differences


def main() -> None:
    domains = {
        "internal": (1088, 860),
        "external": (2224, 2224),
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "comparison_contract": {
            "internal_validation_n": 1088,
            "external_liuyuan_n": 2224,
            "internal_at_evaluable_n": 860,
            "external_at_evaluable_n": 2224,
            "bootstrap_draws": 2000,
            "bootstrap_unit": "examination",
            "difference_orientation": "enhanced minus base; lower is better except RER r",
        },
        "domains": {},
        "source_hashes": {},
    }
    for domain, (expected_n, expected_at_n) in domains.items():
        suffix = "internal" if domain == "internal" else "liuyuan"
        paths = {
            "base": EVAL / f"core_base_seed42_{suffix}_predictions.npz",
            "enhanced": EVAL / f"core_enhanced_seed42_{suffix}_predictions.npz",
        }
        base_ids, base = _components(paths["base"])
        enhanced_ids, enhanced = _components(paths["enhanced"])
        if not np.array_equal(base_ids, enhanced_ids) or len(base_ids) != expected_n:
            raise ValueError(f"{domain} base/enhanced examination order or count differs")
        for role, values in (("base", base), ("enhanced", enhanced)):
            if int(values["at_metric_mask"].sum()) != expected_at_n:
                raise ValueError(f"{domain} {role} AT denominator is not {expected_at_n}")
        models, differences = _bootstrap(base, enhanced, seed=42 if domain == "internal" else 142)
        report["domains"][domain] = {"models": models, "paired_enhanced_minus_base": differences}
        report["source_hashes"].update({str(path): _sha256(path) for path in paths.values()})
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
