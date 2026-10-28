#!/usr/bin/env python3
"""Benchmark active CPETLite student dynamics paths from one comparison matrix."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml

from cpet_foundation.inference.wearable_twin import load_wearable_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _model_dir(matrix: dict, stage: str, seed: int) -> Path:
    existing = matrix.get("existing_outputs", {}).get(stage, {})
    reused = existing.get(seed, existing.get(str(seed)))
    value = reused or matrix["output_dir_template"].format(stage=stage, seed=seed)
    return _resolve(value)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    matrix_path = args.matrix.resolve()
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(42)
    results: dict[str, object] = {
        "schema_version": 1,
        "matrix": str(matrix_path),
        "seed": args.seed,
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "stages": {},
    }
    for stage_record in matrix["stages"]:
        stage = stage_record["id"]
        checkpoint = _model_dir(matrix, stage, args.seed) / "model.pt"
        model, _ = load_wearable_checkpoint(checkpoint)
        model = model.to(device).eval()
        config = model.config
        shape = (args.batch_size, config.max_len)
        wearable_x = torch.randn(*shape, config.n_wearable_features, generator=generator, device=device)
        wearable_mask = torch.ones_like(wearable_x)
        metadata = torch.randn(args.batch_size, config.n_metadata, generator=generator, device=device)
        mask = torch.ones(*shape, device=device)
        phase = torch.zeros(*shape, dtype=torch.long, device=device)
        protocol_x = (
            torch.randn(*shape, config.n_protocol_features, generator=generator, device=device)
            if config.n_protocol_features
            else None
        )
        protocol_mask = torch.ones_like(protocol_x) if protocol_x is not None else None
        body_mass = torch.full((args.batch_size,), 70.0, device=device)

        def forward() -> None:
            model(
                wearable_x,
                metadata,
                wearable_mask,
                mask,
                phase,
                protocol_x=protocol_x,
                protocol_mask=protocol_mask,
                body_mass_kg=body_mass,
            )

        with torch.inference_mode():
            for _ in range(args.warmup):
                forward()
            _synchronize(device)
            elapsed_ms = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                forward()
                _synchronize(device)
                elapsed_ms.append((time.perf_counter() - started) * 1000.0)
        median_ms = statistics.median(elapsed_ms)
        results["stages"][stage] = {
            "checkpoint": str(checkpoint),
            "rollout_type": config.rollout_type,
            "batch_size": args.batch_size,
            "steps": config.max_len,
            "mean_ms": statistics.mean(elapsed_ms),
            "median_ms": median_ms,
            "sd_ms": statistics.stdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0,
            "per_exam_median_ms": median_ms / args.batch_size,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = args.output.resolve() if args.output else _resolve(matrix["evaluation_dir"]) / "inference_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
