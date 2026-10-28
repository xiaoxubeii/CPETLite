#!/usr/bin/env python3
"""Securely score frozen v4 students on measured internal and external labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from summarize_teacher_only_cpetlite import _case_components, _metrics


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("primary_trajectory_rmse", "at_time_s_mae", "vo2peak_mae")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs/cpetlite_v4_package_comparison.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--reuse-existing", action="store_true",
                        help="Reuse complete evaluations tied to the current checkpoint hash.")
    args = parser.parse_args()
    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    evaluation = ROOT / matrix["evaluation_dir"]
    evaluation.mkdir(parents=True, exist_ok=True)
    records = []
    for stage, spec in matrix["stages"].items():
        if args.stage and stage not in args.stage:
            continue
        for seed in matrix["seeds"]:
            if args.seed and seed not in args.seed:
                continue
            checkpoint = ROOT / spec["output_template"].format(seed=seed) / "model.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            for domain, split in (("internal", "validation"), ("external", "test")):
                name = f"{stage}_seed{seed}_{domain}"
                path = evaluation / f"{name}.json"
                predictions = evaluation / f"{name}_predictions.npz"
                manifest_path = evaluation / f"{name}_manifest.json"
                expected_count = matrix.get("evaluation_counts", {}).get(domain, 1086 if domain == "internal" else 2224)
                reusable = False
                if args.reuse_existing and args.limit is None and path.is_file() and predictions.is_file() and manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    reusable = (manifest.get("checkpoint_sha256") == checkpoint_hash
                                and manifest.get("loaded_examinations") == expected_count
                                and manifest.get("requested_examinations") == expected_count
                                and manifest.get("split_name") == split)
                    if reusable:
                        with np.load(predictions) as archive:
                            reusable = len(archive["exam_id"]) == expected_count
                if reusable:
                    print(f"Reusing complete evaluation: {name}", flush=True)
                else:
                    command = [
                        sys.executable, str(ROOT / "scripts/evaluate_wearable_checkpoint.py"),
                        "--checkpoint", str(checkpoint),
                        "--config", str(ROOT / matrix["secure_evaluation_config"]),
                        "--teacher-checkpoint", str(ROOT / matrix["teacher_checkpoint"]),
                        "--split-name", split, "--output", str(path),
                        "--predictions-output", str(predictions),
                    ]
                    if args.limit is not None:
                        command.extend(("--limit", str(args.limit)))
                    log_path = evaluation / f"{name}.log"
                    env = dict(os.environ)
                    env["PYTHONPATH"] = str(ROOT / "src")
                    with log_path.open("w", encoding="utf-8") as log:
                        completed = subprocess.run(command, cwd=ROOT, env=env,
                                                   stdout=log, stderr=subprocess.STDOUT, check=False)
                    if completed.returncode:
                        raise RuntimeError(f"Secure evaluation failed: {log_path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                values = payload.get("metrics", payload)
                records.append({"stage": stage, "seed": seed, "domain": domain,
                                "metrics": {key: values[key] for key in METRICS},
                                "at_count": values.get("at_time_s_evaluable_count"),
                                "peak_count": values.get("vo2peak_evaluable_count")})
    summary = {}
    for stage in matrix["stages"]:
        for domain in ("internal", "external"):
            rows = [r for r in records if r["stage"] == stage and r["domain"] == domain]
            if not rows:
                continue
            summary[f"{stage}:{domain}"] = {
                key: {"mean": float(np.mean([row["metrics"][key] for row in rows])),
                      "sd": float(np.std([row["metrics"][key] for row in rows], ddof=1)) if len(rows) > 1 else None}
                for key in METRICS
            }
    paired = {}
    for domain in ("internal", "external"):
        for target_stage, reference_stage in (
            ("ordinary_package", "measured"),
            ("observable_package", "measured"),
            ("observable_package", "ordinary_package"),
        ):
            per_seed = []
            for seed in matrix["seeds"]:
                paths = [evaluation / f"{stage}_seed{seed}_{domain}_predictions.npz"
                         for stage in (target_stage, reference_stage)]
                if not all(path.is_file() for path in paths):
                    continue
                target_ids, target = _case_components(paths[0])
                reference_ids, reference = _case_components(paths[1])
                if not np.array_equal(target_ids, reference_ids):
                    raise ValueError(f"Paired examination order differs: {domain}, seed {seed}")
                per_seed.append((int(seed), target, reference))
            if not per_seed:
                continue
            point = np.mean([_metrics(target) - _metrics(reference)
                             for _, target, reference in per_seed], axis=0)
            rng = np.random.default_rng(42 if domain == "internal" else 142)
            samples = np.empty((2000, len(METRICS)), dtype=np.float64)
            n = len(per_seed[0][1]["core_sse"])
            for draw in range(len(samples)):
                indices = rng.integers(0, n, size=n)
                samples[draw] = np.mean([
                    _metrics(target, indices) - _metrics(reference, indices)
                    for _, target, reference in per_seed
                ], axis=0)
            paired[f"{target_stage}_minus_{reference_stage}:{domain}"] = {
                "seeds": [seed for seed, _, _ in per_seed],
                "conditional_on_fitted_models": True,
                "bootstrap_unit": "examination",
                "bootstrap_draws": len(samples),
                "metrics": {
                    key: {"mean_paired_difference": float(point[index]),
                          "ci95_low": float(np.quantile(samples[:, index], 0.025)),
                          "ci95_high": float(np.quantile(samples[:, index], 0.975))}
                    for index, key in enumerate(METRICS)
                },
            }
    result = {"schema_version": 1, "experiment_id": matrix["experiment_id"],
              "evaluation_boundary": "secure measured-label evaluation after frozen package-only checkpoint selection",
              "exploratory_limit": args.limit, "records": records, "summary": summary,
              "paired": paired}
    output = evaluation / ("pilot_summary.json" if args.limit else "summary.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
