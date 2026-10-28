#!/usr/bin/env python3
"""Materialize and execute the matched CPETLite v4 three-arm comparison."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from cpet_foundation.training.runner import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def run(config_path: Path, *, stages: set[str], seeds: set[int], train: bool, workers: int = 1) -> list[dict[str, object]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    matrix = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if matrix.get("schema_version") != 1 or len(matrix.get("seeds", ())) < 3:
        raise ValueError("Expected a prespecified three-seed v4 package matrix")
    matched = None
    if matrix.get("matched_inputs"):
        from cpet_foundation.data.matched_inputs import checked_matched_package
        matched = checked_matched_package(ROOT / matrix["package"])
    records: list[dict[str, object]] = []
    # The training loader resolves dataset paths from config_path.parents[1],
    # so generated configs must live directly under the repository configs/.
    generated = ROOT / "configs"
    generated.mkdir(parents=True, exist_ok=True)
    jobs = []
    for stage, spec in matrix["stages"].items():
        if stages and stage not in stages:
            continue
        base = ROOT / spec["base_config"]
        if not base.is_file():
            raise FileNotFoundError(base)
        for seed in matrix["seeds"]:
            if seeds and seed not in seeds:
                continue
            config = load_yaml(base)
            config["seed"] = int(seed)
            config["output_dir"] = spec["output_template"].format(seed=seed)
            config["experiment_group"] = matrix["experiment_id"]
            config["comparison_role"] = stage
            if matched is not None:
                from cpet_foundation.data.matched_inputs import checked_matched_package
                if stage == "measured":
                    config["matched_input_package"] = matrix["package"]
                    if config.get("teacher_checkpoint") not in (None, ""):
                        raise ValueError("Matched measured reference must not access a teacher")
                else:
                    config["require_matched_input_contract"] = True
                checked_matched_package(ROOT / matrix["package"], config=config,
                    source=(ROOT / config["source_h5"]) if stage == "measured" else None)
            if stage != "measured":
                config["distillation_package_path"] = matrix["package"]
                if any(config.get(name) not in (None, "") for name in ("source_h5", "split_path", "normalization_stats_path", "teacher_checkpoint")):
                    raise ValueError(f"Package-only stage {stage} resolves a raw-data or teacher path")
            prefix = matrix.get("config_prefix", "cpetlite_v4_package")
            path = generated / f"{prefix}_{stage}_seed{seed}.yaml"
            text = yaml.safe_dump(config, sort_keys=False)
            if matched is not None and path.exists() and path.read_text() != text:
                raise ValueError(f"Existing matched config differs: {path}")
            path.write_text(text, encoding="utf-8")
            output = ROOT / config["output_dir"]
            status = "existing" if (output / "model.pt").is_file() else "materialized"
            if matched is not None and status == "existing":
                from cpet_foundation.data.matched_inputs import checked_matched_fit
                checked_matched_fit(output, matched, measured=stage == "measured")
            record = {"stage": stage, "seed": seed, "config": str(path),
                      "output": str(output), "status": status}
            records.append(record)
            if train and status != "existing":
                jobs.append(record)
    def fit(record):
        stage, seed = record["stage"], record["seed"]
        log_directory = matrix.get("log_directory", "artifacts/logs/cpetlite_v4_package")
        log_path = ROOT / log_directory / f"{stage}_seed{seed}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "src")
            completed = subprocess.run(
                [sys.executable, "-u", str(ROOT / "scripts/train_config.py"), "--config", record["config"]],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        record["status"] = "complete" if completed.returncode == 0 else "failed"
        if completed.returncode:
            raise RuntimeError(f"{stage} seed {seed} failed; see {log_path}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fit, jobs))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs/cpetlite_v4_package_comparison.yaml")
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    records = run(args.matrix.resolve(), stages=set(args.stage), seeds=set(args.seed), train=args.train, workers=args.workers)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
