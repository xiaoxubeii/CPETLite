#!/usr/bin/env python3
"""Evaluate completed distillation checkpoints on configured secure domains."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _evaluate(
    record: dict,
    domain: dict,
    evaluation_dir: Path,
    force: bool,
    evaluation_teacher_checkpoint: str | None,
    secure_evaluation_config: str | None,
) -> tuple[dict, dict, int]:
    domain_id = str(domain["id"])
    prefix = evaluation_dir / domain_id / f"{record['stage']}_seed{record['seed']}"
    metrics = prefix.with_name(prefix.name + "_metrics.json")
    predictions = prefix.with_name(prefix.name + "_predictions.npz")
    manifest = prefix.with_name(prefix.name + "_manifest.json")
    if metrics.is_file() and predictions.is_file() and manifest.is_file() and not force:
        return record, domain, 0
    log_dir = ROOT / "artifacts/logs/distillation_component_ablation"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{domain_id}_{record['stage']}_seed{record['seed']}.log"
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    src_path = str(ROOT / "src")
    environment["PYTHONPATH"] = src_path + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_wearable_checkpoint.py"),
        "--checkpoint",
        str(_resolve(record["output_dir"]) / "model.pt"),
        "--split-name",
        str(domain["split_name"]),
        "--evaluation-name",
        f"distillation_component_{record['stage']}_seed{record['seed']}_{domain_id}",
        "--output",
        str(metrics),
        "--predictions-output",
        str(predictions),
        "--manifest-output",
        str(manifest),
    ]
    if domain.get("center"):
        command.extend(["--center", str(domain["center"])])
    if secure_evaluation_config:
        command.extend(["--config", str(_resolve(secure_evaluation_config))])
    if evaluation_teacher_checkpoint:
        command.extend(["--teacher-checkpoint", str(_resolve(evaluation_teacher_checkpoint))])
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    return record, domain, result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "configs/distillation_component_ablation_cpetlite_v1.yaml",
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    matrix = yaml.safe_load(args.matrix.resolve().read_text())
    evaluation_dir = _resolve(matrix["evaluation_dir"])
    domains = matrix.get("evaluation_domains") or [
        {"id": "external", "split_name": "test", "center": "liuyuan"}
    ]
    for domain in domains:
        (evaluation_dir / str(domain["id"])).mkdir(parents=True, exist_ok=True)
    registry = json.loads((evaluation_dir / "run_registry.json").read_text())
    records = [record for record in registry["records"] if record.get("status") == "complete"]
    evaluation_teacher_checkpoint = matrix.get("evaluation_teacher_checkpoint")
    secure_evaluation_config = matrix.get("secure_evaluation_config")
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {
            executor.submit(
                _evaluate,
                record,
                domain,
                evaluation_dir,
                args.force,
                evaluation_teacher_checkpoint,
                secure_evaluation_config,
            ): (record, domain)
            for record in records
            for domain in domains
        }
        for future in as_completed(futures):
            record, domain, returncode = future.result()
            print(json.dumps({"stage": record["stage"], "seed": record["seed"], "domain": domain["id"], "returncode": returncode}))
            if returncode != 0:
                failed.append(record)
    if failed:
        raise RuntimeError(f"secure evaluation failed for {len(failed)} run-domain pair(s)")


if __name__ == "__main__":
    main()
