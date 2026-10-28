#!/usr/bin/env python3
"""Prepare a fresh fixed-input paper comparison from a selected recipe family."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

import yaml

from cpet_foundation.data.matched_inputs import input_contract
from cpet_foundation.training.runner import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def prepare(recipe_root, base_prefix, prefix, destination, dataset, report):
    recipe_root, destination, dataset = map(Path, (recipe_root, destination, dataset))
    if not re.fullmatch(r"[a-z][a-z0-9_]*", prefix) or prefix == base_prefix:
        raise ValueError("Matched config prefix must be a fresh namespace")
    source_configs = recipe_root / "configs"
    artifacts = {}
    package = f"artifacts/dataset/{prefix}_teacher_package/distillation.h5"
    teacher_path = f"artifacts/models/{prefix}_teacher_seed42/model.pt"
    common = {"mixed_precision": "none", "wearable_simulation_seed": 42}
    teacher = load_yaml(source_configs / f"train_{base_prefix}_teacher.yaml")
    teacher.update(source_h5=str(dataset.resolve()), output_dir=str(Path(teacher_path).parent))
    artifacts[f"train_{prefix}_teacher"] = teacher
    export = load_yaml(source_configs / f"export_{base_prefix}.yaml")
    export.update(common, source_h5=str(dataset.resolve()), teacher_checkpoint=teacher_path)
    artifacts[f"export_{prefix}"] = export
    contract = input_contract(export)
    fits = 0
    for arch in ("direct", "gru"):
        matrix = yaml.safe_load((source_configs / f"{base_prefix}_{arch}.yaml").read_text())
        matrix = copy.deepcopy(matrix)
        matrix.update(
            experiment_id=f"{prefix}_{arch}", config_prefix=f"{prefix}_{arch}",
            log_directory=f"artifacts/logs/{prefix}_{arch}",
            evaluation_dir=f"artifacts/eval/{prefix}_{arch}",
            package=package, teacher_checkpoint=teacher_path, matched_inputs=True,
            evaluation_counts={"internal": report["splits"]["val"]["examinations"],
                               "external": report["splits"]["test"]["examinations"]},
        )
        for stage, entry in matrix["stages"].items():
            config = load_yaml(recipe_root / entry["base_config"])
            config.update(common)
            if stage == "measured":
                config.update(source_h5=str(dataset.resolve()), teacher_checkpoint=None,
                              matched_input_package=package)
            else:
                if any(config.get(key) not in (None, "") for key in
                       ("source_h5", "split_path", "normalization_stats_path", "teacher_checkpoint")):
                    raise ValueError(f"Package stage {stage} contains raw source/teacher paths")
                config.update(distillation_package_path=package, require_matched_input_contract=True)
            if input_contract(config) != contract:
                raise ValueError(f"Matched input configuration differs: {arch}/{stage}")
            name = f"train_{prefix}_{arch}_{stage}"
            config["output_dir"] = f"artifacts/models/{prefix}_{arch}_{stage}_seed42"
            artifacts[name] = config
            entry.update(base_config=f"configs/{name}.yaml",
                         output_template=f"artifacts/models/{prefix}_{arch}_{stage}_seed{{seed}}")
            fits += len(matrix["seeds"])
        matrix["secure_evaluation_config"] = matrix["stages"]["measured"]["base_config"]
        artifacts[f"{prefix}_{arch}"] = matrix
    # Check the complete write set before writing any generated configuration.
    encoded = {destination / "configs" / f"{name}.yaml": yaml.safe_dump(value, sort_keys=False)
               for name, value in artifacts.items()}
    for path, text in encoded.items():
        if path.exists() and path.read_text() != text:
            raise ValueError(f"Existing generated config differs: {path}")
    (destination / "configs").mkdir(parents=True, exist_ok=True)
    for path, text in encoded.items():
        if not path.exists():
            path.write_text(text)
    return {"config_prefix": prefix, "package": package, "student_fits": fits,
            "evaluations": 2 * fits, "input_contract": contract,
            "recipes": [str(path) for path in encoded]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-root", type=Path, default=ROOT)
    parser.add_argument("--base-prefix", required=True)
    parser.add_argument("--config-prefix", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    from run_cpetlite_paper_v5 import verified_cohort
    report = verified_cohort(args.dataset_dir.resolve())
    plan = prepare(args.recipe_root.resolve(), args.base_prefix, args.config_prefix, ROOT,
                   args.dataset_dir.resolve() / "dataset.h5", report)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
