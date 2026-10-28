#!/usr/bin/env python3
"""Fit endpoint-only CPETLite baselines with sample-level eligibility masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cpet_foundation.data.sequences import (
    CpetSequenceDataset,
    SequenceConfig,
    ids_from_split,
    normalization_options_from_config,
)
from cpet_foundation.eval.endpoint_baselines import build_endpoint_feature_matrix
from cpet_foundation.training.runner import load_yaml


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _dataset(root: Path, config: dict, split_name: str, center: str | None) -> CpetSequenceDataset:
    split_path = _resolve(root, config["split_path"])
    ids = ids_from_split(split_path, split_name)
    dataset = CpetSequenceDataset(
        SequenceConfig(
            source_h5=_resolve(root, config["source_h5"]),
            feature_columns=tuple(config["feature_columns"]),
            metadata_columns=tuple(config["metadata_columns"]),
            max_len=int(config.get("max_len", 160)),
            exam_ids=ids,
            **normalization_options_from_config(config, root),
        )
    )
    if center is not None:
        expected = center.strip().lower()
        dataset.samples = [
            sample for sample in dataset.samples if str(sample.get("institute", "")).strip().lower() == expected
        ]
    if not dataset.samples:
        raise ValueError(f"No samples for split={split_name!r}, center={center!r}")
    return dataset


def _model(seed: int) -> object:
    return make_pipeline(
        StandardScaler(),
        HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            l2_regularization=0.05,
            random_state=seed,
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--eval-split", default="validation", choices=("validation", "test"))
    parser.add_argument("--center")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    train_dataset = _dataset(root, config, "train", None)
    eval_dataset = _dataset(root, config, args.eval_split, args.center)
    modes: dict[str, dict[str, float | int | str]] = {}
    for offset, mode in enumerate(("metadata", "protocol", "metadata_protocol")):
        train = build_endpoint_feature_matrix(train_dataset.samples, config["feature_columns"], mode)
        evaluate = build_endpoint_feature_matrix(eval_dataset.samples, config["feature_columns"], mode)
        if not train.at_supervision_eligible.any() or not evaluate.at_metric_eligible.any():
            raise ValueError(f"No AT-eligible samples for mode={mode}")
        if not train.peak_eligible.any() or not evaluate.peak_eligible.any():
            raise ValueError(f"No peak-eligible samples for mode={mode}")
        at_model = _model(42 + offset)
        peak_model = _model(142 + offset)
        at_model.fit(
            train.features[train.at_supervision_eligible],
            train.at_norm[train.at_supervision_eligible],
        )
        peak_model.fit(train.features[train.peak_eligible], train.peak_vo2[train.peak_eligible])
        at_norm_pred = np.clip(np.asarray(at_model.predict(evaluate.features)), 0.0, 1.0)
        at_time_pred = evaluate.time_start_s + at_norm_pred * evaluate.duration_s
        peak_pred = np.asarray(peak_model.predict(evaluate.features))
        modes[mode] = {
            "feature_count": int(train.features.shape[1]),
            "train_count": int(train.features.shape[0]),
            "evaluation_count": int(evaluate.features.shape[0]),
            "at_train_count": int(train.at_supervision_eligible.sum()),
            "at_evaluable_count": int(evaluate.at_metric_eligible.sum()),
            "at_time_mae_seconds": float(
                np.mean(
                    np.abs(
                        at_time_pred[evaluate.at_metric_eligible]
                        - evaluate.at_time_s[evaluate.at_metric_eligible]
                    )
                )
            ),
            "peak_train_count": int(train.peak_eligible.sum()),
            "peak_evaluable_count": int(evaluate.peak_eligible.sum()),
            "peak_vo2_mae_ml_kg_min": float(
                np.mean(np.abs(peak_pred[evaluate.peak_eligible] - evaluate.peak_vo2[evaluate.peak_eligible]))
            ),
        }
    payload = {
        "schema_version": 1,
        "experiment": "cpetlite_v2_endpoint_only_baselines",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "evaluation_split": args.eval_split,
        "evaluation_center": args.center,
        "normalization_fit_split": "train",
        "at_policy": "fit on at_supervision_mask; score on at_metric_mask",
        "modes": modes,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
