"""Freeze review-control recipes around existing native matrix runners.

This is deployment preparation, not a trainer/splitter/metric implementation.
All destinations must be new; original accepted artifacts are read-only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
FAMILY = "cpetlite_review_controls_v552_20261021"
DATA_SHA = "bb55511681c026e6286a155af22948a453924a5474dc0feeea04625a2652c30d"
PACKAGE_SHA = "9934acad2643e9a5121d9e24d01374c1c0897a5dcf6ef4159cf593dfe8a4f4c0"
TEACHER_SHA = "d71700838a4bace772b89f3e09d8f4ab99a18a57ddf61fce93dd8ea4e2bf5945"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (yaml.safe_dump(payload, sort_keys=False) if path.suffix == ".yaml"
            else json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with path.open("x") as handle:
        handle.write(text)


def prepare(accepted, dataset, destination, remote):
    from cpet_foundation.data.matched_inputs import checked_matched_package

    if destination.exists():
        raise ValueError("Refusing existing deployment directory")
    if sha(dataset / "dataset.h5") != DATA_SHA or sha(accepted / "teacher_package/distillation.h5") != PACKAGE_SHA:
        raise ValueError("Frozen source/package checksum differs")
    source_report = json.loads((dataset / "COHORT_VERIFICATION.json").read_text())
    if {k: v["examinations"] for k, v in source_report["splits"].items()} != {"train": 11404, "val": 1270, "test": 2385}:
        raise ValueError("Frozen cohort differs")
    package = accepted / "teacher_package/distillation.h5"
    package_info = checked_matched_package(package, source=dataset / "dataset.h5")
    project = destination / "foundation"
    project.mkdir(parents=True)
    for folder in ("src", "scripts", "tests"):
        shutil.copytree(ROOT / folder, project / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "pyproject.toml", project / "pyproject.toml")
    (project / "configs").mkdir()
    assets = destination / "assets"
    assets.mkdir()
    for name in ("COHORT_VERIFICATION.json", "DERIVATIVE_MANIFEST.json", "model_columns.yaml"):
        shutil.copy2(dataset / name, assets / name)
    member = "foundation/artifacts/models/cpetlite_paper_v552_finite_metadata_20261021_teacher_seed42/model.pt"
    with tarfile.open(accepted / "collected/RESULTS.tar.gz", "r:gz") as archive:
        node = archive.getmember(member)
        if not node.isfile():
            raise ValueError("Teacher archive entry is not a regular file")
        with archive.extractfile(node) as src, (assets / "teacher.pt").open("xb") as dst:
            shutil.copyfileobj(src, dst)
    if sha(assets / "teacher.pt") != TEACHER_SHA:
        raise ValueError("Frozen teacher differs")
    if package_info["metadata"]["teacher_checkpoint_sha256"] != TEACHER_SHA:
        raise ValueError("Package teacher binding differs")
    r = Path(remote)
    tasks = []
    for arch in ("direct", "gru"):
        original = accepted / f"snapshot/foundation/configs/train_cpetlite_paper_v552_finite_metadata_20261021_{arch}_ordinary_package.yaml"
        base = yaml.safe_load(original.read_text())
        if base["epochs"] != 150 or base["early_stopping"]["patience"] != 12:
            raise ValueError("Frozen student budget differs")
        checked_matched_package(package, config=base)
        stages = {}
        for arm in ("full_package", "mean_endpoint", "protocol_static"):
            cfg = copy.deepcopy(base)
            cfg.update(input_ablation="protocol_static" if arm == "protocol_static" else "none",
                       trajectory_distillation_mode="mean_mse" if arm == "mean_endpoint" else "gaussian_nll",
                       require_matched_input_contract=True,
                       distillation_package_path=str(r / "assets/distillation.h5"))
            if arm == "mean_endpoint":
                cfg["loss_weights"] = {name: 0.0 for name in cfg["loss_weights"]}
                cfg["loss_weights"].update(trajectory_distillation=0.75, endpoint=1.0)
            for key in ("source_h5", "split_path", "normalization_stats_path", "teacher_checkpoint"):
                cfg[key] = None
            name = f"configs/train_{FAMILY}_{arch}_{arm}.yaml"
            write_new(project / name, cfg)
            output_template = f"artifacts/models/{FAMILY}_{arch}_{arm}_seed{{seed}}"
            stages[arm] = {"base_config": name, "output_template": output_template}
            for seed in (17, 42, 73):
                tasks.append({"architecture": arch, "arm": arm, "seed": seed,
                              "config": f"configs/{FAMILY}_{arch}_{arm}_seed{seed}.yaml",
                              "output": output_template.format(seed=seed)})
        # Only data paths are overlaid during secure evaluation. In particular,
        # the checkpoint's ablation, architecture, seed and losses survive.
        secure = {"source_h5": str(r / "assets/dataset.h5"),
                  "input_format": "v5_derivative_split", "split_path": None,
                  "normalization_stats_path": None, "require_frozen_normalization": True,
                  "require_supervision_policy": True}
        secure_name = f"configs/evaluate_{FAMILY}_{arch}.yaml"
        write_new(project / secure_name, secure)
        matrix = {"schema_version": 1, "matched_inputs": True,
                  "experiment_id": f"{FAMILY}_{arch}", "config_prefix": f"{FAMILY}_{arch}",
                  "seeds": [17, 42, 73], "stages": stages,
                  "package": str(r / "assets/distillation.h5"),
                  "teacher_checkpoint": str(r / "assets/teacher.pt"),
                  "secure_evaluation_config": secure_name,
                  "evaluation_counts": {"internal": 1270, "external": 2385},
                  "evaluation_dir": f"artifacts/eval/{FAMILY}_{arch}",
                  "log_directory": f"artifacts/logs/{FAMILY}_{arch}",
                  "paired_comparisons": [["mean_endpoint", "full_package"], ["protocol_static", "full_package"]]}
        write_new(project / f"configs/{FAMILY}_{arch}_matrix.yaml", matrix)
    plan = {"family": FAMILY, "remote": remote, "tasks": tasks, "expected_fits": 18,
            "expected_evaluations": 36, "teacher_retrained": False,
            "dataset_sha256": DATA_SHA, "package_sha256": PACKAGE_SHA, "teacher_sha256": TEACHER_SHA,
            "max_epochs": 150, "early_stopping_patience": 12, "seeds": [17, 42, 73],
            "min_delta": 0.001,
            "split_counts": {"train": 11404, "val": 1270, "test": 2385},
            "selection": "teacher_core_trajectory_rmse on internal validation only",
            "internal_scope": "checkpoint-selection validation, not untouched internal test",
            "external_scope": "Liuyuan inference only; no checkpoint/hyperparameter selection",
            "workers": 4, "preserve_accepted_results": True}
    write_new(destination / "PLAN.json", plan)
    # Freeze the native runner's seed expansion before deployment. The runner
    # independently regenerates/compares these bytes against the base recipes.
    for task in tasks:
        cfg = yaml.safe_load((project / f"configs/train_{FAMILY}_{task['architecture']}_{task['arm']}.yaml").read_text())
        cfg.update(seed=task["seed"], output_dir=task["output"],
                   experiment_group=f"{FAMILY}_{task['architecture']}", comparison_role=task["arm"])
        write_new(project / task["config"], cfg)
    files = {str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob("*")) if p.is_file()}
    files.update({"assets/dataset.h5": DATA_SHA, "assets/distillation.h5": PACKAGE_SHA})
    write_new(destination / "DEPLOYMENT_MANIFEST.json", {"files": files, "plan": plan})
    return {"snapshot": str(destination), "files": len(files), "fits": 18, "evaluations": 36,
            "assets_to_upload": {"dataset.h5": str(dataset / "dataset.h5"), "distillation.h5": str(package)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--remote", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.accepted.resolve(), args.dataset.resolve(), args.destination.resolve(), args.remote), indent=2))


if __name__ == "__main__":
    main()
