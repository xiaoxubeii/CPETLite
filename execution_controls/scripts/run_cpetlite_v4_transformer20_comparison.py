#!/usr/bin/env python3
"""Matched 20-field direct-Transformer teacher versus the rerun SDE teacher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from cpet_foundation.data.distillation_package import audit_distillation_package
from cpet_foundation.training.runner import load_yaml
from scripts.run_cpetlite_v4_legacy_retest import ROOT, run_command, write_config
from scripts.summarize_teacher_only_cpetlite import _case_components, _metrics


NAME = "cpetlite_v4_transformer20_comparison"
SDE_EVAL = ROOT / "artifacts/eval/cpetlite_v4_legacy_retest"
METRICS = ("primary_trajectory_rmse", "at_time_s_mae", "vo2peak_mae")


def preflight_contract(matrix_path: Path, teacher_path: Path, export_path: Path) -> dict:
    sde = load_yaml(ROOT / "configs/train_protocol_sde_cpetlite_v4.yaml")
    teacher = load_yaml(teacher_path)
    omitted = {"architecture", "output_dir"}
    if {k: v for k, v in sde.items() if k not in omitted} != {k: v for k, v in teacher.items() if k not in omitted}:
        raise ValueError("Transformer teacher differs from SDE scientific data/training contract")
    if teacher["architecture"] != "protocol_direct_transformer" or len(teacher["feature_columns"]) != 20:
        raise ValueError("Expected direct Transformer teacher with the original 20 features")
    old_export = load_yaml(ROOT / "configs/export_teacher_package_cpetlite_v4.yaml")
    export = load_yaml(export_path)
    if {k: v for k, v in old_export.items() if k != "teacher_checkpoint"} != {k: v for k, v in export.items() if k != "teacher_checkpoint"}:
        raise ValueError("Teacher export contract differs from historical SDE export")
    matrix = yaml.safe_load(matrix_path.read_text())
    prior = yaml.safe_load((ROOT / "configs/cpetlite_v4_legacy_retest.yaml").read_text())
    if matrix["seeds"] != prior["seeds"] or list(matrix["stages"]) != list(prior["stages"]):
        raise ValueError("Student seeds/stages differ from historical SDE rerun")
    if matrix["stages"]["measured"]["output_template"] != prior["stages"]["measured"]["output_template"]:
        raise ValueError("Measured-label references must be reused unchanged")
    if matrix["stages"]["ordinary_package"]["base_config"] != prior["stages"]["ordinary_package"]["base_config"]:
        raise ValueError("Package student configuration differs from historical SDE rerun")
    if (ROOT / matrix["teacher_checkpoint"]).resolve() != (ROOT / teacher["output_dir"] / "model.pt").resolve():
        raise ValueError("Matrix does not point to the new Transformer teacher")
    if (ROOT / export["teacher_checkpoint"]).resolve() != (ROOT / teacher["output_dir"] / "model.pt").resolve():
        raise ValueError("Export does not point to the new Transformer teacher")
    if (ROOT / matrix["package"]).resolve() == (ROOT / prior["package"]).resolve():
        raise ValueError("New package would overwrite the SDE package")
    if matrix["stages"]["ordinary_package"]["output_template"] == prior["stages"]["ordinary_package"]["output_template"]:
        raise ValueError("New students would overwrite SDE-package students")
    return {"teacher_features": teacher["feature_columns"], "teacher_metadata": teacher["metadata_columns"],
            "epochs": teacher["epochs"], "seeds": matrix["seeds"]}


def compare_sde(new_eval: Path, destination: Path) -> None:
    old = json.loads((SDE_EVAL / "summary.json").read_text())
    new = json.loads((new_eval / "summary.json").read_text())
    means = {}
    paired = {}
    for domain in ("internal", "external"):
        key = f"ordinary_package:{domain}"
        means[domain] = {metric: {
            "sde": old["summary"][key][metric]["mean"],
            "transformer": new["summary"][key][metric]["mean"],
            "transformer_minus_sde": new["summary"][key][metric]["mean"] - old["summary"][key][metric]["mean"],
        } for metric in METRICS}
        pairs = []
        for seed in (17, 42, 73):
            ids_new, cases_new = _case_components(new_eval / f"ordinary_package_seed{seed}_{domain}_predictions.npz")
            ids_old, cases_old = _case_components(SDE_EVAL / f"ordinary_package_seed{seed}_{domain}_predictions.npz")
            if not np.array_equal(ids_new, ids_old):
                raise ValueError(f"Case order differs for {domain}, seed {seed}")
            pairs.append((cases_new, cases_old))
        point = np.mean([_metrics(a) - _metrics(b) for a, b in pairs], axis=0)
        n = len(pairs[0][0]["core_sse"])
        rng = np.random.default_rng(20260930 if domain == "internal" else 20261001)
        draws = np.empty((2000, len(METRICS)))
        for draw in range(len(draws)):
            indices = rng.integers(0, n, size=n)
            draws[draw] = np.mean([_metrics(a, indices) - _metrics(b, indices) for a, b in pairs], axis=0)
        paired[domain] = {metric: {"mean": float(point[i]),
                                  "ci95": [float(np.quantile(draws[:, i], 0.025)),
                                           float(np.quantile(draws[:, i], 0.975))]}
                          for i, metric in enumerate(METRICS)}
    destination.write_text(json.dumps({"comparison": "Transformer20 minus fresh SDE20",
                                       "teacher_fits_per_architecture": 1,
                                       "student_seeds": [17, 42, 73],
                                       "bootstrap_unit": "examination, conditional on fitted models",
                                       "means": means, "paired": paired}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    matrix_path = ROOT / f"configs/{NAME}.yaml"
    teacher_path = ROOT / "configs/train_protocol_direct_transformer_cpetlite_v4_20.yaml"
    export_path = ROOT / "configs/export_teacher_package_cpetlite_v4_transformer20.yaml"
    print(json.dumps({"preflight": preflight_contract(matrix_path, teacher_path, export_path)}), flush=True)
    matrix = yaml.safe_load(matrix_path.read_text())
    teacher, export = load_yaml(teacher_path), load_yaml(export_path)
    run_name = NAME + "_smoke" if args.smoke else NAME
    log_dir = ROOT / "artifacts/logs" / run_name
    if args.smoke:
        teacher.update(output_dir=f"artifacts/models/{run_name}_teacher_seed42", limit_exams=32, epochs=2)
        export["teacher_checkpoint"] = teacher["output_dir"] + "/model.pt"
        matrix.update(experiment_id=run_name, config_prefix=run_name,
                      teacher_checkpoint=export["teacher_checkpoint"],
                      package=f"artifacts/dataset/{run_name}/distillation.h5",
                      evaluation_dir=f"artifacts/eval/{run_name}")
        matrix["stages"]["measured"]["output_template"] = "artifacts/models/cpetlite_v4_legacy_retest_smoke_measured_seed{seed}"
        matrix["stages"]["ordinary_package"]["output_template"] = f"artifacts/models/{run_name}_package_seed{{seed}}"
        teacher_path = write_config(ROOT / f"configs/{run_name}_teacher.yaml", teacher)
        export_path = write_config(ROOT / f"configs/{run_name}_export.yaml", export)
        matrix_path = write_config(ROOT / f"configs/{run_name}_matrix.yaml", matrix)
    checkpoint = ROOT / teacher["output_dir"] / "model.pt"
    if not all((checkpoint.parent / file).is_file() for file in ("model.pt", "metrics.json", "manifest.json")):
        print(json.dumps({"stage": "teacher", "epochs": teacher["epochs"]}), flush=True)
        run_command([sys.executable, "-u", str(ROOT / "scripts/train_config.py"), "--config", str(teacher_path)], log_dir / "teacher.log")
    package = ROOT / matrix["package"]
    if not package.is_file():
        print(json.dumps({"stage": "package_export"}), flush=True)
        command = [sys.executable, "-u", str(ROOT / "scripts/export_teacher_only_package.py"),
                   "--config", str(export_path), "--output", str(package)]
        if args.smoke:
            command += ["--limit-exams", "32"]
        run_command(command, log_dir / "export.log")
    audit = audit_distillation_package(package)
    if audit["splits"] != ({"train": 32, "validation": 6} if args.smoke else {"train": 9779, "validation": 1086}):
        raise ValueError(f"Unexpected package sample counts: {audit['splits']}")
    if audit["supervision_mode"] != "teacher_only":
        raise ValueError("Package is not teacher-only")
    if not audit["metadata"].get("observability"):
        run_command([sys.executable, str(ROOT / "scripts/add_teacher_package_observability.py"),
                     "--package", str(package)], log_dir / "observability.log")
    student_paths = []
    for seed in ([42] if args.smoke else matrix["seeds"]):
        config = load_yaml(ROOT / matrix["stages"]["ordinary_package"]["base_config"])
        config.update(seed=seed, output_dir=matrix["stages"]["ordinary_package"]["output_template"].format(seed=seed),
                      distillation_package_path=matrix["package"], experiment_group=matrix["experiment_id"],
                      comparison_role="ordinary_package")
        if any(config.get(k) not in (None, "") for k in ("source_h5", "split_path", "normalization_stats_path", "teacher_checkpoint")):
            raise ValueError("Package student resolves raw source or teacher checkpoint")
        if args.smoke:
            config.update(epochs=2, limit_exams=32)
        path = write_config(ROOT / f"configs/{matrix['config_prefix']}_ordinary_package_seed{seed}.yaml", config)
        student_paths.append((seed, path, ROOT / config["output_dir"] / "model.pt"))
    audit_cmd = [sys.executable, str(ROOT / "scripts/audit_cpetlite_v4_package.py"), "--package", str(package)]
    for _, path, _ in student_paths:
        audit_cmd += ["--student-config", str(path)]
    run_command(audit_cmd, log_dir / "package_audit.json")
    for seed, path, model in student_paths:
        if not model.is_file():
            print(json.dumps({"stage": "package_student", "seed": seed}), flush=True)
            run_command([sys.executable, "-u", str(ROOT / "scripts/train_config.py"), "--config", str(path)],
                        log_dir / f"student_seed{seed}.log")
    command = [sys.executable, "-u", str(ROOT / "scripts/evaluate_cpetlite_v4_package_comparison.py"),
               "--matrix", str(matrix_path)]
    if args.smoke:
        command += ["--seed", "42", "--limit", "32"]
    print(json.dumps({"stage": "evaluation"}), flush=True)
    run_command(command, log_dir / "evaluation.log")
    evaluation = ROOT / matrix["evaluation_dir"]
    if not args.smoke:
        compare_sde(evaluation, evaluation / "transformer_vs_sde.json")
    print(json.dumps({"status": "complete", "summary": str(evaluation / ("pilot_summary.json" if args.smoke else "summary.json"))}), flush=True)


if __name__ == "__main__":
    main()
