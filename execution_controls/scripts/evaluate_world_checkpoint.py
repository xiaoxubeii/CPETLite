#!/usr/bin/env python3
"""Evaluate a full-input CPET world-model checkpoint on a selected split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cpet_foundation.data.sequences import (
    CpetSequenceDataset,
    SequenceConfig,
    ids_from_split,
    normalization_options_from_config,
)
from cpet_foundation.inference.digital_twin import load_checkpoint
from cpet_foundation.training.runner import evaluate_model, load_yaml, set_seed


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split-name", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--center")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--input-mode", choices=("prefix", "full"), default="prefix")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    checkpoint_path = args.checkpoint.resolve()
    model, checkpoint = load_checkpoint(checkpoint_path)
    config = dict(checkpoint["config"])
    if args.config is not None:
        config.update(load_yaml(args.config.resolve()))
    set_seed(int(config.get("seed", 42)))
    split_path = _resolve(root, config["split_path"])
    ids = ids_from_split(split_path, args.split_name, args.limit)
    dataset = CpetSequenceDataset(
        SequenceConfig(
            source_h5=_resolve(root, config["source_h5"]),
            feature_columns=tuple(config["feature_columns"]),
            metadata_columns=tuple(config["metadata_columns"]),
            max_len=int(config.get("max_len", 160)),
            limit_exams=args.limit,
            exam_ids=ids,
            **normalization_options_from_config(config, root),
        )
    )
    if args.center:
        expected = args.center.strip().lower()
        dataset.samples = [
            sample for sample in dataset.samples if str(sample.get("institute", "")).strip().lower() == expected
        ]
    if not dataset.samples:
        raise ValueError(f"No samples for split={args.split_name!r}, center={args.center!r}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    metrics = evaluate_model(
        model,
        DataLoader(dataset, batch_size=int(config.get("batch_size", 64)), pin_memory=device.type == "cuda"),
        {name: index for index, name in enumerate(config["feature_columns"])},
        device,
        input_mode=args.input_mode,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "split_path": str(split_path),
        "split_name": args.split_name,
        "center": args.center,
        "evaluated_examinations": len(dataset),
        "input_mode": args.input_mode,
        "metrics_path": str(output),
    }
    manifest_output = args.manifest_output.resolve() if args.manifest_output else output.with_name("manifest.json")
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": str(output), "manifest": str(manifest_output)}, indent=2))


if __name__ == "__main__":
    main()
