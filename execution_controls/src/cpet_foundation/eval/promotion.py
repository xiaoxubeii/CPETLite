from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = {
    "at_norm_mae": 0.08,
    "vo2peak_mae": 3.0,
    "trajectory_rmse": 0.5,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _default_eval_path(model_dir: Path, suffix: str) -> Path:
    artifact_root = model_dir.parents[1] if len(model_dir.parents) >= 2 else model_dir.parent
    return artifact_root / "eval" / f"{model_dir.name}_{suffix}.json"


def _check_thresholds(metrics: dict[str, Any], thresholds: dict[str, float], context: str) -> dict[str, dict[str, float | str]]:
    failures: dict[str, dict[str, float | str]] = {}
    for metric_name, max_value in thresholds.items():
        value = metrics.get(metric_name)
        if value is None:
            continue
        if float(value) > max_value:
            failures[f"{context}.{metric_name}"] = {
                "value": float(value),
                "max": float(max_value),
                "context": context,
            }
    return failures


def _internal_threshold_failures(internal_eval: dict[str, Any], thresholds: dict[str, float]) -> dict[str, dict[str, float | str]]:
    failures: dict[str, dict[str, float | str]] = {}
    for split_name, split_payload in internal_eval.get("splits", {}).items():
        if not isinstance(split_payload, dict):
            continue
        for key, value in split_payload.items():
            if isinstance(value, dict):
                failures.update(_check_thresholds(value, thresholds, f"internal.{split_name}.{key}"))
        failures.update(_check_thresholds(split_payload, thresholds, f"internal.{split_name}"))
    return failures


def _loco_threshold_failures(loco_eval: dict[str, Any], thresholds: dict[str, float]) -> dict[str, dict[str, float | str]]:
    failures: dict[str, dict[str, float | str]] = {}
    for fold_name, fold_payload in loco_eval.get("folds", {}).items():
        if not isinstance(fold_payload, dict):
            continue
        failures.update(_check_thresholds(fold_payload.get("metrics", {}), thresholds, f"loco.{fold_name}"))
    if isinstance(loco_eval.get("summary"), dict):
        summary_thresholds = {f"mean_{key}": value for key, value in thresholds.items()}
        failures.update(_check_thresholds(loco_eval["summary"], summary_thresholds, "loco.summary"))
    return failures


def evaluate_promotion(
    model_dir: Path,
    output_path: Path | None = None,
    internal_eval_path: Path | None = None,
    loco_eval_path: Path | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    manifest_path = model_dir / "manifest.json"
    metrics_path = model_dir / "metrics.json"
    internal_eval_path = internal_eval_path or _default_eval_path(model_dir, "internal")
    loco_eval_path = loco_eval_path or _default_eval_path(model_dir, "loco")
    thresholds = thresholds or DEFAULT_THRESHOLDS

    blocked: list[str] = []
    linked_artifacts: dict[str, str] = {}
    manifest: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    internal_failures: dict[str, dict[str, float | str]] = {}
    loco_failures: dict[str, dict[str, float | str]] = {}

    if not manifest_path.exists():
        blocked.append("missing_model_manifest")
    else:
        manifest = _read_json(manifest_path)
        linked_artifacts["manifest"] = str(manifest_path)

    if not metrics_path.exists():
        blocked.append("missing_model_metrics")
    else:
        metrics = _read_json(metrics_path)
        linked_artifacts["metrics"] = str(metrics_path)

    if manifest:
        checkpoint_path = _resolve(manifest.get("checkpoint", "model.pt"), model_dir)
        split_path = _resolve(manifest.get("split_path", ""), model_dir)
        if not checkpoint_path.exists():
            blocked.append("missing_checkpoint")
        else:
            linked_artifacts["checkpoint"] = str(checkpoint_path)
        if not split_path.exists():
            blocked.append("missing_split_file")
        else:
            linked_artifacts["split"] = str(split_path)

    if not internal_eval_path.exists():
        blocked.append("missing_internal_evaluation")
    else:
        internal_eval = _read_json(internal_eval_path)
        linked_artifacts["internal_evaluation"] = str(internal_eval_path)
        if internal_eval.get("status") != "complete":
            blocked.append("incomplete_internal_evaluation")
        internal_failures = _internal_threshold_failures(internal_eval, thresholds)

    if not loco_eval_path.exists():
        blocked.append("missing_loco_evaluation")
    else:
        loco_eval = _read_json(loco_eval_path)
        linked_artifacts["loco_evaluation"] = str(loco_eval_path)
        if loco_eval.get("status") != "complete":
            blocked.append("incomplete_loco_evaluation")
        if not loco_eval.get("folds"):
            blocked.append("empty_loco_evaluation")
        loco_failures = _loco_threshold_failures(loco_eval, thresholds)

    failed_thresholds: dict[str, dict[str, float]] = {}
    for metric_name, max_value in thresholds.items():
        value = metrics.get(metric_name)
        if value is None:
            blocked.append(f"missing_metric_{metric_name}")
            continue
        if float(value) > max_value:
            failed_thresholds[metric_name] = {"value": float(value), "max": float(max_value)}
    if failed_thresholds:
        blocked.append("failed_metric_thresholds")
    if internal_failures:
        blocked.append("failed_internal_thresholds")
    if loco_failures:
        blocked.append("failed_loco_thresholds")

    decision = {
        "eligible": not blocked,
        "model_dir": str(model_dir),
        "thresholds": thresholds,
        "failed_thresholds": failed_thresholds,
        "failed_internal_thresholds": internal_failures,
        "failed_loco_thresholds": loco_failures,
        "blocked_reasons": blocked,
        "linked_artifacts": linked_artifacts,
        "metrics": metrics,
        "status": "promoted_for_research_inference" if not blocked else "blocked",
    }
    if output_path is not None:
        _write_json(output_path, decision)
    return decision
