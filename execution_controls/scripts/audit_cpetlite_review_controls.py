"""Acceptance adapter for the frozen 18-fit review controls, not a trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from cpet_foundation.data.matched_inputs import checked_matched_fit, checked_matched_package
from cpet_foundation.training.fit_audit import audit_fit, require
from audit_cpetlite_paper_rerun import checked_evaluation
from run_cpetlite_paper_v5 import verified_cohort


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_plan(plan):
    expected = {(a, b, s) for a in ("direct", "gru")
                for b in ("full_package", "mean_endpoint", "protocol_static") for s in (17, 42, 73)}
    actual = [(t["architecture"], t["arm"], t["seed"]) for t in plan["tasks"]]
    require(len(actual) == 18 and set(actual) == expected, "Incomplete or duplicate fit coverage")
    for key in ("config", "output"):
        require(len({t[key] for t in plan["tasks"]}) == 18, f"Repeated {key} path")
    require(plan["expected_fits"] == 18 and plan["expected_evaluations"] == 36, "Study scope differs")
    require(plan["seeds"] == [17, 42, 73] and plan["max_epochs"] == 150
            and plan["early_stopping_patience"] == 12 and plan["min_delta"] == 0.001, "Budget differs")
    require(plan["teacher_retrained"] is False, "Teacher must be reused")
    require(plan["split_counts"] == {"train": 11404, "val": 1270, "test": 2385}, "Cohort differs")
    return plan


def verify_inputs(snapshot):
    snapshot = Path(snapshot).resolve()
    deployment = json.loads((snapshot / "DEPLOYMENT_MANIFEST.json").read_text())
    plan = validate_plan(json.loads((snapshot / "PLAN.json").read_text()))
    require(deployment["plan"] == plan, "Deployment/plan disagree")
    for name, expected in deployment["files"].items():
        path = (snapshot / name).resolve()
        require(path.is_relative_to(snapshot), "Unsafe frozen input path")
        require(digest(path) == expected, f"Frozen input differs: {name}")
    return plan


def audit(root):
    root = Path(root).resolve()
    plan = verify_inputs(root.parent)
    assets = root.parent / "assets"
    require(digest(assets / "dataset.h5") == plan["dataset_sha256"], "Source differs")
    require(digest(assets / "distillation.h5") == plan["package_sha256"], "Package differs")
    require(digest(assets / "teacher.pt") == plan["teacher_sha256"], "Teacher differs")
    cohort = verified_cohort(assets)
    package = checked_matched_package(assets / "distillation.h5", source=assets / "dataset.h5",
                                      teacher=assets / "teacher.pt")
    fits, evaluations = [], []
    torch.set_num_threads(1)
    for arch in ("direct", "gru"):
        matrix = yaml.safe_load((root / f"configs/{plan['family']}_{arch}_matrix.yaml").read_text())
        require(matrix["seeds"] == plan["seeds"], "Matrix seeds differ")
        summary = json.loads((root / matrix["evaluation_dir"] / "summary.json").read_text())
        expected_records = {(a, s, d) for a in ("full_package", "mean_endpoint", "protocol_static")
                            for s in plan["seeds"] for d in ("internal", "external")}
        records = summary["records"]
        require(len(records) == 18 and {(r["stage"], r["seed"], r["domain"]) for r in records} == expected_records,
                "Missing/duplicate evaluation coverage")
        require(summary["exploratory_limit"] is None, "Limited evaluation is not full acceptance")
        for task in [t for t in plan["tasks"] if t["architecture"] == arch]:
            arm, seed = task["arm"], task["seed"]
            spec = matrix["stages"][arm]
            require(task["output"] == spec["output_template"].format(seed=seed), "Matrix output differs")
            config = yaml.safe_load((root / task["config"]).read_text())
            base = yaml.safe_load((root / spec["base_config"]).read_text())
            expected = dict(base, seed=seed, output_dir=task["output"],
                            experiment_group=matrix["experiment_id"], comparison_role=arm)
            expected["require_matched_input_contract"] = True
            require(config == expected, "Seed recipe differs from native matrix expansion")
            require(config["epochs"] == 150 and config["early_stopping"]["patience"] == 12
                    and config["early_stopping"]["min_delta"] == 0.001, "Fit budget differs")
            require(config["early_stopping"]["monitor"] == "teacher_core_trajectory_rmse", "External/other selection")
            require(all(config.get(k) is None for k in ("source_h5", "split_path", "normalization_stats_path", "teacher_checkpoint")),
                    "Package-only recipe resolves raw data/teacher")
            directory = root / task["output"]
            manifest = checked_matched_fit(directory, package, measured=False)
            evidence = audit_fit(directory, config, arm, plan["split_counts"])
            require(manifest["distillation_package_sha256"] == plan["package_sha256"], "Fit package differs")
            with h5py.File(assets / "distillation.h5", "r") as h:
                for part in ("train", "validation"):
                    group = h[f"splits/{part}/samples"]
                    for field, key in (("target_feature_mean", "output_feature_mean"),
                                       ("target_feature_std", "output_feature_std")):
                        actual = group[field][:]
                        require(np.array_equal(actual, np.broadcast_to(evidence["scalers"][key], actual.shape)), "Scaler differs")
            fits.append({**task, **evidence})
            for domain, split in (("internal", "val"), ("external", "test")):
                stem = root / matrix["evaluation_dir"] / f"{arm}_seed{seed}_{domain}"
                metrics = checked_evaluation(stem, directory / "model.pt", plan["dataset_sha256"], split, cohort)
                record = next(r for r in records if (r["stage"], r["seed"], r["domain"]) == (arm, seed, domain))
                require(all(math.isclose(v, metrics[k], rel_tol=1e-10, abs_tol=1e-10)
                            for k, v in record["metrics"].items()), "Summary metric differs")
                evaluations.append({"architecture": arch, "arm": arm, "seed": seed, "domain": domain})
        for key, values in summary["summary"].items():
            arm, domain = key.split(":")
            rows = [r for r in records if (r["stage"], r["domain"]) == (arm, domain)]
            require(len(rows) == 3, "Summary seed coverage differs")
            for metric, value in values.items():
                actual = [r["metrics"][metric] for r in rows]
                require(np.isclose(value["mean"], np.mean(actual), rtol=1e-10, atol=1e-10)
                        and np.isclose(value["sd"], np.std(actual, ddof=1), rtol=1e-10, atol=1e-10), "Mean/SD differs")
        for arm in ("mean_endpoint", "protocol_static"):
            for domain in ("internal", "external"):
                pair = summary["paired"][f"{arm}_minus_full_package:{domain}"]
                require(pair["seeds"] == [17, 42, 73] and pair["bootstrap_draws"] == 2000, "Paired scope differs")
                require(all(math.isfinite(v) for metric in pair["metrics"].values() for v in metric.values()),
                        "Nonfinite paired result")
    require(len(fits) == 18 and len(evaluations) == 36, "Study incomplete")
    result = {"status": "verified", "fits": fits, "evaluations": evaluations,
              "student_fits": 18, "evaluation_count": 36, "teacher_retrained": False,
              "internal_scope": plan["internal_scope"], "external_scope": plan["external_scope"]}
    output = root / f"artifacts/eval/{plan['family']}/FIT_AND_COVERAGE_AUDIT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    result = verify_inputs(root.parent) if args.check_inputs else audit(root)
    print(json.dumps({k: v for k, v in result.items() if k not in ("tasks", "fits", "evaluations")}))
