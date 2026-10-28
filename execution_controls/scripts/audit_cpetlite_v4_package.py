#!/usr/bin/env python3
"""Fail closed on CPETLite v4 teacher-package data-access boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from cpet_foundation.data.distillation_package import audit_distillation_package
from cpet_foundation.model.wearable_student import TEACHER_ONLY_FORBIDDEN_BATCH_KEYS
from cpet_foundation.training.runner import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--student-config", action="append", required=True, type=Path)
    args = parser.parse_args()
    audit = audit_distillation_package(args.package)
    exposed = set(audit["student_visible_fields"]) & set(TEACHER_ONLY_FORBIDDEN_BATCH_KEYS)
    exposed |= set(audit["student_visible_fields"]) & {
        "at_supervision_mask", "at_metric_mask", "trajectory_supervision_mask"
    }
    if exposed:
        raise SystemExit(f"Forbidden measured targets in package: {sorted(exposed)}")
    provenance = audit["metadata"].get("observability", {})
    if provenance.get("raw_measured_labels_used") is not False:
        raise SystemExit("Observability provenance does not exclude measured labels")
    with h5py.File(args.package, "r") as handle:
        for split in ("train", "validation"):
            group = handle[f"splits/{split}/samples"]
            if "teacher_observability_weight" not in group:
                raise SystemExit(f"Missing observability weights in {split}")
            if group["teacher_observability_weight"].shape != group["teacher_trajectory_mu"].shape:
                raise SystemExit(f"Weight/teacher shape mismatch in {split}")
            n = len(group["wearable_x"])
            for start in range(0, n, 128):
                values = group["teacher_observability_weight"][start:start + 128]
                if not np.isfinite(values).all() or (values < 0.1).any() or (values > 2.01).any():
                    raise SystemExit(f"Invalid observability weights in {split}")
    for path in args.student_config:
        config = load_yaml(path.resolve())
        if str(config.get("supervision_mode")) != "teacher_only":
            raise SystemExit(f"Student does not use teacher-only supervision: {path}")
        for key in ("source_h5", "split_path", "normalization_stats_path", "teacher_checkpoint"):
            if config.get(key) not in (None, ""):
                raise SystemExit(f"Student resolves privileged {key}: {path}")
        configured_package = Path(config["distillation_package_path"])
        if not configured_package.is_absolute():
            configured_package = Path(__file__).resolve().parents[1] / configured_package
        if configured_package.resolve() != args.package.resolve():
            raise SystemExit(f"Student package path differs: {path}")
    print(json.dumps({"status": "pass", "package": str(args.package),
                      "splits": audit["splits"], "forbidden_fields": [],
                      "student_configs": [str(path) for path in args.student_config]}))


if __name__ == "__main__":
    main()
