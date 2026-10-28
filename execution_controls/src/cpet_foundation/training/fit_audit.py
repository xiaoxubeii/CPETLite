"""Read-only native-fit acceptance, reusing the accepted history replay rules.

The epoch replay is promoted from the hash-verified v552 supplementary audit
(kernel f12490e8851f31da05eafa35a105706ac555d262c6d5e573fd2d845a1b8c42f9).
This module never trains, resumes or rewrites a checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def replay_history(history, config, control):
    """Replay epoch continuity, min-delta, patience and restored selection."""
    early = config["early_stopping"]
    package_format = config.get("supervision_mode") == "teacher_only"
    require(early["enabled"] and early["restore_best_weights"], "Unexpected selection policy")
    require(early["mode"] == "min", "This frozen plan requires minimization")
    require(0 < len(history) <= config["epochs"], "Missing or over-budget history")
    require([x["epoch"] for x in history] == list(range(1, len(history) + 1)), "Epoch gap")
    best, best_epoch, bad = None, 0, 0
    for record in history:
        value = record[early["monitor"]] if package_format else record["validation"][early["monitor"]]
        require(math.isfinite(value) and math.isfinite(record["train_loss"]), "Nonfinite loss/monitor")
        require(math.isfinite(record["learning_rate"]) and record["learning_rate"] > 0, "Invalid learning rate")
        improved = best is None or best - value >= early["min_delta"]
        if improved:
            best, best_epoch, bad = value, record["epoch"], 0
        else:
            bad += 1
        require(record["improved"] == improved, "Improvement replay mismatch")
        if not package_format:
            require(record["bad_epochs"] == bad, "Bad-epoch replay mismatch")
        require(record["epoch"] == len(history) or bad < early["patience"], "Continued past required early stop")
    require(control["epochs_trained"] == len(history), "Actual epoch count mismatch")
    require(control["max_epochs"] == config["epochs"], "Epoch ceiling mismatch")
    require(control["best_epoch"] == best_epoch and control["best_monitor_value"] == best, "Selected epoch/value mismatch")
    require(control["monitor"] == early["monitor"], "Monitor mismatch")
    if not package_format:
        require(control["mode"] == early["mode"], "Monitor mode mismatch")
        require(control["early_stopping"] == early and control["lr_scheduler"] == config["lr_scheduler"], "Control config mismatch")
    require(control["restore_best_weights"], "Best weights not restored")
    require(control["stopped_early"] == (bad >= early["patience"]), "Early-stop status mismatch")
    require(len(history) == config["epochs"] or control["stopped_early"], "Premature incomplete training")
    require(control["final_learning_rate"] == history[-1]["learning_rate"], "Final learning rate mismatch")
    return {"epochs": len(history), "best_epoch": best_epoch, "monitor": early["monitor"],
            "best_monitor_value": best, "stopped_early": control["stopped_early"]}


def audit_fit(directory: Path, config: dict, arm: str, counts: dict) -> dict:
    """Verify actual package-only fit artifacts against a declared recipe."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    history = json.loads((directory / "history.json").read_text())
    checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    require(arm in {"full_package", "mean_endpoint", "protocol_static"}, "Unknown control arm")
    require(config.get("supervision_mode") == "teacher_only", "Control is not package-only")
    require(all(checkpoint["config"].get(k) == v for k, v in config.items()), "Checkpoint recipe differs")
    require(checkpoint["training_control"] == manifest["training_control"], "Checkpoint controls differ")
    evidence = replay_history(history, config, manifest["training_control"])
    require(manifest["train_samples"] == counts["train"] and manifest["validation_samples"] == counts["val"],
            "Complete training/validation cohort differs")
    require(manifest.get("raw_cpet_opened_by_student_trainer") is False
            and manifest.get("measured_training_targets_available") is False,
            "Teacher-only access boundary differs")
    expected_ablation = "protocol_static" if arm == "protocol_static" else "none"
    expected_loss = "mean_mse" if arm == "mean_endpoint" else "gaussian_nll"
    for value in (config, checkpoint["config"], manifest):
        require(value.get("input_ablation") == expected_ablation, "Input ablation differs")
        require(value.get("trajectory_distillation_mode") == expected_loss, "Trajectory loss mode differs")
    if arm == "mean_endpoint":
        weights = config["loss_weights"]
        require(weights["trajectory_distillation"] == 0.75 and weights["endpoint"] == 1.0,
                "Mean/endpoint weights differ")
        require(all(float(v) == 0 for k, v in weights.items()
                    if k not in {"trajectory_distillation", "endpoint"}), "Auxiliary loss enabled")
    require(all(torch.isfinite(value).all().item() for value in checkpoint["model_state"].values()
                if torch.is_floating_point(value)), "Nonfinite checkpoint parameter")
    scalers = {}
    for key in ("output_feature_mean", "output_feature_std", "metadata_feature_mean", "metadata_feature_std"):
        values = checkpoint["config"][key]
        require(all(math.isfinite(value) for value in values), "Nonfinite scaler")
        require(not key.endswith("std") or all(value > 0 for value in values), "Nonpositive scaler")
        scalers[key] = values
    with (directory / "model.pt").open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"arm": arm, "checkpoint_sha256": digest, "scalers": scalers, **evidence}
