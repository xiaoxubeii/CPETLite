#!/usr/bin/env python3
"""Summarize repeated-seed CPETLite distillation-component ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from cpet_foundation.training.runner import load_yaml


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs/distillation_component_ablation_cpetlite_v1.yaml"
METRICS = ("primary_trajectory_rmse", "at_time_s_mae", "vo2peak_mae")
PRIMARY_CHANNELS = ("VO2", "VCO2", "VE")
FIXED_KEYS = (
    "architecture",
    "source_h5",
    "split_path",
    "normalization_stats_path",
    "teacher_checkpoint",
    "distillation_package_path",
    "supervision_mode",
    "input_boundary",
    "feature_columns",
    "target_feature_columns",
    "watch_channels",
    "protocol_context_channels",
    "metadata_columns",
    "temporal_backbone_type",
    "rollout_type",
    "batch_size",
    "epochs",
    "learning_rate",
    "early_stopping",
    "lr_scheduler",
)
TEACHER_WEIGHT_KEYS = (
    "trajectory_distillation",
    "latent_alignment",
    "uncertainty_distillation",
    "teacher_endpoint",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _evaluation_artifact(
    evaluation_dir: Path, domain: str, stage: str, seed: int, suffix: str
) -> Path:
    return evaluation_dir / domain / f"{stage}_seed{seed}_{suffix}"


def _mean_sd(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n_seeds": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _case_components(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = np.load(path)
    channel_names = [str(value) for value in data["channel_names"]]
    indices = [channel_names.index(name) for name in PRIMARY_CHANNELS]
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


def _component_metrics(values: dict[str, np.ndarray], indices: np.ndarray | None = None) -> np.ndarray:
    selected = slice(None) if indices is None else indices
    at_mask = values["at_metric_mask"][selected]
    return np.asarray(
        [
            np.sqrt(values["core_sse"][selected].sum() / values["core_n"][selected].sum()),
            (values["at_abs_error_s"][selected] * at_mask).sum() / at_mask.sum(),
            (values["peak_abs_error"][selected] * values["peak_mask"][selected]).sum()
            / values["peak_mask"][selected].sum(),
        ],
        dtype=np.float64,
    )


def _paired_bootstrap_across_seeds(
    current_paths: dict[int, Path],
    previous_paths: dict[int, Path],
    *,
    draws: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    pairs = []
    for student_seed in sorted(set(current_paths).intersection(previous_paths)):
        current_ids, current = _case_components(current_paths[student_seed])
        previous_ids, previous = _case_components(previous_paths[student_seed])
        if not np.array_equal(current_ids, previous_ids):
            raise AssertionError(f"prediction IDs differ for seed {student_seed}")
        pairs.append((student_seed, current, previous, len(current_ids)))
    if not pairs:
        return {}
    seed_points = np.stack([_component_metrics(current) - _component_metrics(previous) for _, current, previous, _ in pairs])
    point = seed_points.mean(axis=0)
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, len(METRICS)), dtype=np.float64)
    for draw in range(draws):
        differences = []
        for _, current, previous, n in pairs:
            indices = rng.integers(0, n, size=n)
            differences.append(_component_metrics(current, indices) - _component_metrics(previous, indices))
        samples[draw] = np.stack(differences).mean(axis=0)
    return {
        metric: {
            "n_seeds": len(pairs),
            "mean_difference": float(point[index]),
            "seed_sd": float(seed_points[:, index].std(ddof=1)) if len(pairs) > 1 else 0.0,
            "examination_bootstrap_ci95_low": float(np.quantile(samples[:, index], 0.025)),
            "examination_bootstrap_ci95_high": float(np.quantile(samples[:, index], 0.975)),
        }
        for index, metric in enumerate(METRICS)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    args = parser.parse_args()
    matrix_path = args.matrix.resolve()
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    evaluation_dir = _resolve(matrix["evaluation_dir"])
    registry_path = evaluation_dir / "run_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    stage_order = [stage["id"] for stage in matrix["stages"]]
    stage_labels = {stage["id"]: stage["label"] for stage in matrix["stages"]}
    records = registry["records"]
    complete = [record for record in records if record.get("status") == "complete"]
    incomplete = [
        {"stage": record["stage"], "seed": record["seed"], "status": record.get("status", "not_started")}
        for record in records
        if record.get("status") != "complete"
    ]
    if not complete:
        raise RuntimeError("no completed distillation-component runs are available")

    resolved_configs = {f"{record['stage']}:{record['seed']}": load_yaml(_resolve(record["config"])) for record in complete}
    reference_stage = matrix.get("reference_stage", stage_order[0])
    reference_label = next(
        label for label in resolved_configs if label.startswith(f"{reference_stage}:")
    )
    reference = resolved_configs[reference_label]
    variable_keys = set(matrix.get("comparison_variable_keys", []))
    variable_loss_weights = set(
        matrix.get("comparison_variable_loss_weights", TEACHER_WEIGHT_KEYS)
    )
    for label, config in resolved_configs.items():
        for key in FIXED_KEYS:
            if key in variable_keys:
                continue
            if config[key] != reference[key]:
                raise AssertionError(f"unmatched ablation contract for {label}: {key}")
        for key, value in reference["loss_weights"].items():
            if key not in variable_loss_weights and config["loss_weights"][key] != value:
                raise AssertionError(f"measured loss changed for {label}: {key}")

    by_stage: dict[str, dict] = {}
    per_seed: dict[tuple[str, int], dict] = {}
    for stage in stage_order:
        stage_records = sorted((record for record in complete if record["stage"] == stage), key=lambda item: item["seed"])
        entries = []
        for record in stage_records:
            output_dir = _resolve(record["output_dir"])
            secure_internal_metrics = _evaluation_artifact(
                evaluation_dir, "internal", stage, int(record["seed"]), "metrics.json"
            )
            metrics_path = (
                secure_internal_metrics
                if secure_internal_metrics.is_file()
                else output_dir / "metrics.json"
            )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            selected = {metric: float(metrics[metric]) for metric in METRICS}
            entry = {
                "seed": int(record["seed"]),
                "metrics": selected,
                "output_dir": record["output_dir"],
                "checkpoint_sha256": _sha256(output_dir / "model.pt"),
                "metrics_sha256": _sha256(metrics_path),
            }
            external_metrics = evaluation_dir / "external" / f"{stage}_seed{record['seed']}_metrics.json"
            if external_metrics.is_file():
                payload = json.loads(external_metrics.read_text(encoding="utf-8"))
                entry["external_metrics"] = {metric: float(payload[metric]) for metric in METRICS}
                entry["external_metrics_sha256"] = _sha256(external_metrics)
            entries.append(entry)
            per_seed[(stage, int(record["seed"]))] = entry
        by_stage[stage] = {
            "label": stage_labels[stage],
            "teacher_terms": next(item["teacher_terms"] for item in matrix["stages"] if item["id"] == stage),
            "runs": entries,
            "internal_summary": {
                metric: _mean_sd([entry["metrics"][metric] for entry in entries]) for metric in METRICS
            },
        }
        if entries and all("external_metrics" in entry for entry in entries):
            by_stage[stage]["external_summary"] = {
                metric: _mean_sd([entry["external_metrics"][metric] for entry in entries]) for metric in METRICS
            }

    increments = []
    adjacent_pairs = (
        zip(stage_order, stage_order[1:])
        if matrix.get("comparison_mode", "cumulative") == "cumulative"
        else ()
    )
    for previous, current in adjacent_pairs:
        common_seeds = sorted(
            set(record["seed"] for record in by_stage[previous]["runs"])
            & set(record["seed"] for record in by_stage[current]["runs"])
        )
        internal = {}
        external = {}
        for metric in METRICS:
            deltas = [
                per_seed[(current, seed)]["metrics"][metric] - per_seed[(previous, seed)]["metrics"][metric]
                for seed in common_seeds
            ]
            if deltas:
                internal[metric] = _mean_sd(deltas)
            external_deltas = [
                per_seed[(current, seed)]["external_metrics"][metric]
                - per_seed[(previous, seed)]["external_metrics"][metric]
                for seed in common_seeds
                if "external_metrics" in per_seed[(current, seed)]
                and "external_metrics" in per_seed[(previous, seed)]
            ]
            if external_deltas:
                external[metric] = _mean_sd(external_deltas)
        internal_prediction_paths = {
            seed: _evaluation_artifact(evaluation_dir, "internal", current, seed, "predictions.npz")
            for seed in common_seeds
        }
        previous_internal_prediction_paths = {
            seed: _evaluation_artifact(evaluation_dir, "internal", previous, seed, "predictions.npz")
            for seed in common_seeds
        }
        internal_paired = _paired_bootstrap_across_seeds(
            internal_prediction_paths,
            previous_internal_prediction_paths,
            seed=42 + stage_order.index(current),
        ) if common_seeds else {}
        external_prediction_paths = {
            seed: evaluation_dir / "external" / f"{current}_seed{seed}_predictions.npz"
            for seed in common_seeds
            if (evaluation_dir / "external" / f"{current}_seed{seed}_predictions.npz").is_file()
            and (evaluation_dir / "external" / f"{previous}_seed{seed}_predictions.npz").is_file()
        }
        previous_external_prediction_paths = {
            seed: evaluation_dir / "external" / f"{previous}_seed{seed}_predictions.npz"
            for seed in external_prediction_paths
        }
        external_paired = _paired_bootstrap_across_seeds(
            external_prediction_paths,
            previous_external_prediction_paths,
            seed=142 + stage_order.index(current),
        ) if external_prediction_paths else {}
        increments.append(
            {
                "added_stage": current,
                "relative_to": previous,
                "common_seeds": common_seeds,
                "difference_orientation": "current minus previous; lower is better",
                "internal": internal,
                "external": external,
                "internal_paired_examination_bootstrap": internal_paired,
                "external_paired_examination_bootstrap": external_paired,
            }
        )

    baseline_stage = reference_stage
    net_effects = []
    for current in (stage for stage in stage_order if stage != baseline_stage):
        common_seeds = sorted(
            set(record["seed"] for record in by_stage[baseline_stage]["runs"])
            & set(record["seed"] for record in by_stage[current]["runs"])
        )
        internal_paths = {
            seed: _evaluation_artifact(evaluation_dir, "internal", current, seed, "predictions.npz")
            for seed in common_seeds
        }
        baseline_internal_paths = {
            seed: _evaluation_artifact(evaluation_dir, "internal", baseline_stage, seed, "predictions.npz")
            for seed in common_seeds
        }
        external_paths = {
            seed: evaluation_dir / "external" / f"{current}_seed{seed}_predictions.npz"
            for seed in common_seeds
            if (evaluation_dir / "external" / f"{current}_seed{seed}_predictions.npz").is_file()
            and (evaluation_dir / "external" / f"{baseline_stage}_seed{seed}_predictions.npz").is_file()
        }
        baseline_external_paths = {
            seed: evaluation_dir / "external" / f"{baseline_stage}_seed{seed}_predictions.npz"
            for seed in external_paths
        }
        net_effects.append(
            {
                "stage": current,
                "relative_to": baseline_stage,
                "common_seeds": common_seeds,
                "difference_orientation": "stage minus reference; lower is better",
                "internal_paired_examination_bootstrap": _paired_bootstrap_across_seeds(
                    internal_paths,
                    baseline_internal_paths,
                    seed=242 + stage_order.index(current),
                ),
                "external_paired_examination_bootstrap": _paired_bootstrap_across_seeds(
                    external_paths,
                    baseline_external_paths,
                    seed=342 + stage_order.index(current),
                ) if external_paths else {},
            }
        )

    named_contrasts = []
    for contrast in matrix.get("named_contrasts", []):
        current = contrast["current"]
        previous = contrast["reference"]
        common_seeds = sorted(
            set(record["seed"] for record in by_stage[current]["runs"])
            & set(record["seed"] for record in by_stage[previous]["runs"])
        )
        internal_paths = {
            seed: _evaluation_artifact(evaluation_dir, "internal", current, seed, "predictions.npz")
            for seed in common_seeds
        }
        previous_internal_paths = {
            seed: _evaluation_artifact(evaluation_dir, "internal", previous, seed, "predictions.npz")
            for seed in common_seeds
        }
        external_paths = {
            seed: evaluation_dir / "external" / f"{current}_seed{seed}_predictions.npz"
            for seed in common_seeds
            if (evaluation_dir / "external" / f"{current}_seed{seed}_predictions.npz").is_file()
            and (evaluation_dir / "external" / f"{previous}_seed{seed}_predictions.npz").is_file()
        }
        previous_external_paths = {
            seed: evaluation_dir / "external" / f"{previous}_seed{seed}_predictions.npz"
            for seed in external_paths
        }
        payload = {
            "id": contrast["id"],
            "current": current,
            "reference": previous,
            "difference_orientation": f"{current} minus {previous}; lower is better",
            "internal": _paired_bootstrap_across_seeds(
                internal_paths,
                previous_internal_paths,
                seed=442,
            ),
            "external": _paired_bootstrap_across_seeds(
                external_paths,
                previous_external_paths,
                seed=542,
            ) if external_paths else {},
        }
        named_contrasts.append(payload)
        (evaluation_dir / f"{contrast['id']}_paired.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    output = {
        "schema_version": 1,
        "experiment_id": matrix["experiment_id"],
        "matrix_sha256": _sha256(matrix_path),
        "registry_sha256": _sha256(registry_path),
        "contract": {
            "architecture": matrix.get("contract_label", "direct Transformer"),
            "seeds": matrix["seeds"],
            "stage_order": stage_order,
            "reference_stage": reference_stage,
            "fixed_keys": sorted(set(FIXED_KEYS) - variable_keys),
            "comparison_variable_keys": sorted(variable_keys),
            "teacher_weight_keys": list(TEACHER_WEIGHT_KEYS),
            "difference_orientation": "stage minus reference; lower is better",
            "teacher_only_is_context_not_component_attribution": True,
            "internal_at_evaluable_n": int(
                _case_components(
                    _evaluation_artifact(
                        evaluation_dir,
                        "internal",
                        complete[0]["stage"],
                        int(complete[0]["seed"]),
                        "predictions.npz",
                    )
                )[1]["at_metric_mask"].sum()
            ),
            "external_at_evaluable_n": int(
                _case_components(
                    evaluation_dir
                    / "external"
                    / f"{complete[0]['stage']}_seed{complete[0]['seed']}_predictions.npz"
                )[1]["at_metric_mask"].sum()
            )
            if (
                evaluation_dir
                / "external"
                / f"{complete[0]['stage']}_seed{complete[0]['seed']}_predictions.npz"
            ).is_file()
            else None,
        },
        "completion": {
            "complete": len(complete),
            "planned": len(records),
            "incomplete": incomplete,
        },
        "stages": by_stage,
        "incremental_effects": increments,
        "effects_vs_reference": net_effects,
        "net_effects_vs_measured_only": net_effects,
        "named_contrasts": named_contrasts,
        "evidence_boundary": matrix.get(
            "evidence_boundary",
            "single cumulative ordering with three student seeds; not a full factorial interaction analysis",
        ),
    }
    output_path = evaluation_dir / "comparison.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "complete": len(complete), "planned": len(records)}, indent=2))


if __name__ == "__main__":
    main()
