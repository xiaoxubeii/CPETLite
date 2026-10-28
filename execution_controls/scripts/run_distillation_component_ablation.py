#!/usr/bin/env python3
"""Materialize and run the prespecified CPETLite distillation-component matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

from cpet_foundation.training.runner import load_yaml, train_from_config


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _load_matrix(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("distillation ablation schema_version must be 1")
    stages = payload.get("stages", [])
    if not stages or len({stage["id"] for stage in stages}) != len(stages):
        raise ValueError("distillation ablation stages must be non-empty and unique")
    if len(payload.get("seeds", [])) < 3:
        raise ValueError("distillation ablation requires at least three prespecified seeds")
    return payload


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def materialize_configs(matrix_path: Path) -> list[dict]:
    matrix = _load_matrix(matrix_path)
    base_path = _resolve(matrix["base_config"])
    base = load_yaml(base_path)
    config_dir = _resolve(matrix["generated_config_dir"])
    config_dir.mkdir(parents=True, exist_ok=True)
    records = []
    existing = matrix.get("existing_outputs", {})
    for stage in matrix["stages"]:
        for raw_seed in matrix["seeds"]:
            seed = int(raw_seed)
            config = _deep_merge(base, matrix.get("training_overrides", {}))
            config = _deep_merge(config, stage.get("config_overrides", {}))
            config["seed"] = seed
            config["experiment_group"] = matrix["experiment_id"]
            config["comparison_role"] = f"distillation_ablation_{stage['id']}"
            config["distillation_ablation_stage"] = stage["id"]
            config["distillation_teacher_terms"] = list(stage.get("teacher_terms", []))
            config["loss_weights"].update(stage.get("loss_weights", {}))
            configured_output = matrix["output_dir_template"].format(stage=stage["id"], seed=seed)
            reused_output = existing.get(stage["id"], {}).get(seed, existing.get(stage["id"], {}).get(str(seed)))
            config["output_dir"] = reused_output or configured_output
            config_name = matrix.get(
                "config_name_template", "generated_distill_ablation_{stage}_seed{seed}.yaml"
            ).format(stage=stage["id"], seed=seed)
            config_path = config_dir / config_name
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            records.append(
                {
                    "stage": stage["id"],
                    "label": stage["label"],
                    "teacher_terms": list(stage.get("teacher_terms", [])),
                    "seed": seed,
                    "config": str(config_path.relative_to(ROOT)),
                    "config_sha256": _sha256(config_path),
                    "output_dir": config["output_dir"],
                    "reused": bool(reused_output),
                }
            )
    return records


def _completed(record: dict) -> bool:
    output = _resolve(record["output_dir"])
    return all((output / name).is_file() for name in ("model.pt", "metrics.json", "manifest.json"))


def _write_registry(matrix_path: Path, records: list[dict]) -> Path:
    matrix = _load_matrix(matrix_path)
    evaluation_dir = _resolve(matrix["evaluation_dir"])
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        record["status"] = "complete" if _completed(record) else record.get("status", "not_started")
        if record["status"] == "complete":
            output = _resolve(record["output_dir"])
            record["checkpoint_sha256"] = _sha256(output / "model.pt")
            record["metrics"] = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    registry = {
        "schema_version": 1,
        "experiment_id": matrix["experiment_id"],
        "matrix": str(matrix_path.relative_to(ROOT)),
        "matrix_sha256": _sha256(matrix_path),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    path = evaluation_dir / "run_registry.json"
    path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return path


def _run_training_process(record: dict) -> tuple[int, str]:
    log_dir = ROOT / "artifacts/logs/distillation_component_ablation"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{record['stage']}_seed{record['seed']}.log"
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    src_path = str(ROOT / "src")
    environment["PYTHONPATH"] = src_path + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/train_config.py"), "--config", str(_resolve(record["config"]))],
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode, str(log_path.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "configs/distillation_component_ablation_cpetlite_v1.yaml",
    )
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--stage", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    matrix_path = args.matrix.resolve()
    records = materialize_configs(matrix_path)
    selected_stages = set(args.stage or [])
    selected_seeds = set(args.seed or [])
    chosen = [
        record
        for record in records
        if (not selected_stages or record["stage"] in selected_stages)
        and (not selected_seeds or record["seed"] in selected_seeds)
    ]
    if args.max_runs is not None:
        chosen = chosen[: max(args.max_runs, 0)]
    registry_path = _write_registry(matrix_path, records)
    if args.materialize_only:
        print(json.dumps({"materialized": len(records), "registry": str(registry_path)}, indent=2))
        return
    pending = []
    for record in chosen:
        if _completed(record) and not args.force:
            record["status"] = "complete"
            _write_registry(matrix_path, records)
            continue
        record["status"] = "running"
        pending.append(record)
    _write_registry(matrix_path, records)
    if int(args.jobs) <= 1:
        for record in pending:
            try:
                train_from_config(_resolve(record["config"]))
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                _write_registry(matrix_path, records)
                raise
            record["status"] = "complete"
            _write_registry(matrix_path, records)
    else:
        with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
            futures = {executor.submit(_run_training_process, record): record for record in pending}
            for future in as_completed(futures):
                record = futures[future]
                returncode, log_path = future.result()
                record["training_log"] = log_path
                record["status"] = "complete" if returncode == 0 and _completed(record) else "failed"
                if record["status"] == "failed":
                    record["error"] = f"training process exited with code {returncode}"
                _write_registry(matrix_path, records)
                print(json.dumps({"stage": record["stage"], "seed": record["seed"], "status": record["status"]}))
        failed = [record for record in pending if record["status"] == "failed"]
        if failed:
            raise RuntimeError(f"{len(failed)} distillation ablation training process(es) failed")
    print(registry_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
