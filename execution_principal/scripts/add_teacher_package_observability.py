#!/usr/bin/env python3
"""Attach label-free, out-of-fold restricted-view teacher predictability weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


CORE = ("VO2", "VCO2", "VE")
FOLDS = 3
MIN_TAU = 0.05


def design_matrix(group: h5py.Group, start: int, stop: int) -> np.ndarray:
    watch = group["wearable_x"][start:stop].astype(np.float64)
    watch_mask = group["wearable_mask"][start:stop].astype(np.float64)
    protocol = group["protocol_x"][start:stop].astype(np.float64)
    protocol_mask = group["protocol_mask"][start:stop].astype(np.float64)
    metadata = group["metadata"][start:stop].astype(np.float64)
    phase = group["phase"][start:stop].astype(np.float64)[..., None] / 2.0
    if "phase_mask" in group:
        phase *= group["phase_mask"][start:stop].astype(np.float64)[..., None]
    n, steps, _ = watch.shape
    if metadata.shape[-1] != 4:
        raise ValueError("CPETLite v4 probe expects Age, Gender, Height_cm and Weight_kg metadata")
    static = np.broadcast_to(metadata[:, None, :], (n, steps, metadata.shape[-1]))
    position = np.broadcast_to(np.linspace(0, 1, steps)[None, :, None], (n, steps, 1))
    intercept = np.ones((n, steps, 1), dtype=np.float64)
    parts = (watch * watch_mask, watch_mask, protocol * protocol_mask,
             protocol_mask, static / np.array([100.0, 1.0, 200.0, 200.0]),
             phase, position, intercept)
    return np.clip(np.concatenate(parts, axis=-1), -10.0, 10.0)


def fit_probes(group: h5py.Group, core_indices: list[int], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = group["wearable_x"].shape[0]
    if "probe_fold" not in group:
        raise ValueError("Package lacks centre-scoped subject-group probe folds")
    width = design_matrix(group, 0, 1).shape[-1]
    xtx = np.zeros((FOLDS, width, width), dtype=np.float64)
    xty = np.zeros((FOLDS, width, len(core_indices)), dtype=np.float64)
    counts = np.zeros(FOLDS, dtype=np.float64)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        x = design_matrix(group, start, stop)[:, ::4, :]
        y = group["teacher_trajectory_mu"][start:stop, ::4, :][:, :, core_indices].astype(np.float64)
        valid = np.isfinite(y).all(axis=-1) & (group["mask"][start:stop, ::4] > 0)
        folds = group["probe_fold"][start:stop]
        for fold in range(FOLDS):
            rows = folds == fold
            xf = x[rows][valid[rows]]
            yf = y[rows][valid[rows]]
            if len(xf):
                xtx[fold] += xf.T @ xf
                xty[fold] += xf.T @ yf
                counts[fold] += len(xf)
    if (counts == 0).any():
        raise ValueError("An observability fold has no valid teacher targets")
    all_xx, all_xy = xtx.sum(axis=0), xty.sum(axis=0)

    def solve(matrix: np.ndarray, rhs: np.ndarray, count: float) -> np.ndarray:
        penalty = np.eye(width) * max(count, 1.0) * 1e-3
        penalty[-1, -1] = 0.0  # do not penalize the intercept
        return np.linalg.solve(matrix + penalty, rhs)

    heldout = np.stack([solve(all_xx - xtx[fold], all_xy - xty[fold], counts.sum() - counts[fold])
                        for fold in range(FOLDS)])
    return heldout, solve(all_xx, all_xy, counts.sum())


def estimate_tau(group: h5py.Group, core_indices: list[int], probes: np.ndarray,
                 batch_size: int) -> np.ndarray:
    residuals = []
    n = group["wearable_x"].shape[0]
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        x = design_matrix(group, start, stop)[:, ::4, :]
        y = group["teacher_trajectory_mu"][start:stop, ::4, :][:, :, core_indices].astype(np.float64)
        folds = group["probe_fold"][start:stop]
        predictions = np.stack([x[j] @ probes[int(folds[j])]
                                for j in range(stop - start)])
        valid = group["mask"][start:stop, ::4] > 0
        residuals.append(np.abs(y - predictions)[valid])
    values = np.concatenate(residuals, axis=0)
    return np.maximum(np.median(values, axis=0), MIN_TAU)


def add_weights(group: h5py.Group, core_indices: list[int], probes: np.ndarray,
                *, validation: bool, batch_size: int, tau: np.ndarray,
                scale: np.ndarray | None = None) -> np.ndarray:
    n, steps, channels = group["teacher_trajectory_mu"].shape
    if "teacher_observability_weight" in group:
        raise FileExistsError(f"{group.name}/teacher_observability_weight already exists")
    output = group.create_dataset("teacher_observability_weight", (n, steps, channels),
                                  dtype="float16", chunks=(min(batch_size, n), steps, channels),
                                  compression="lzf")
    sums = np.zeros(len(core_indices), dtype=np.float64)
    counts = np.zeros(len(core_indices), dtype=np.float64)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        x = design_matrix(group, start, stop)
        teacher = group["teacher_trajectory_mu"][start:stop, :, :][:, :, core_indices].astype(np.float64)
        fold_ids = group["probe_fold"][start:stop]
        prediction = x @ probes if validation else np.stack([x[j] @ probes[fold_ids[j]]
                                                               for j in range(stop - start)])
        residual = np.abs(teacher - prediction)
        raw = 0.2 + 0.8 / (1.0 + (residual / tau) ** 2)
        raw = np.where(np.isfinite(raw), raw, 0.2)
        valid = (group["mask"][start:stop] > 0)[..., None]
        sums += (raw * valid).sum(axis=(0, 1))
        counts += valid.sum(axis=(0, 1))
        weights = np.ones((stop - start, steps, channels), dtype=np.float16)
        weights[:, :, core_indices] = raw.astype(np.float16)
        output[start:stop] = weights
    if scale is None:
        scale = counts / np.maximum(sums, 1e-8)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        weights = output[start:stop].astype(np.float32)
        weights[:, :, core_indices] = np.clip(weights[:, :, core_indices] * scale, 0.1, 2.0)
        output[start:stop] = weights.astype(np.float16)
    return scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    with h5py.File(args.package, "r+") as handle:
        metadata = json.loads(handle.attrs["metadata_json"])
        if metadata["model_contract"]["metadata_columns"] != [
            "Age", "Gender", "Height_cm", "Weight_kg"
        ]:
            raise ValueError("Unexpected CPETLite v4 package metadata order")
        columns = metadata["model_contract"]["target_feature_columns"]
        core_indices = [columns.index(name) for name in CORE]
        train = handle["splits/train/samples"]
        validation = handle["splits/validation/samples"]
        heldout, full = fit_probes(train, core_indices, args.batch_size)
        tau = estimate_tau(train, core_indices, heldout, args.batch_size)
        scale = add_weights(train, core_indices, heldout, validation=False,
                            batch_size=args.batch_size, tau=tau)
        add_weights(validation, core_indices, full, validation=True,
                    batch_size=args.batch_size, scale=scale, tau=tau)
        metadata["observability"] = {
            "method": "three-fold centre-scoped subject-group out-of-fold restricted-view ridge probe",
            "probe_fit": "training package inputs and teacher outputs only",
            "validation_probe": "all training package samples",
            "core_channels": list(CORE), "tau_train_oof_median": tau.tolist(),
            "weight_range": [0.1, 2.0],
            "raw_measured_labels_used": False,
        }
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        visible = set(json.loads(handle.attrs["student_visible_fields_json"]))
        visible.add("teacher_observability_weight")
        handle.attrs["student_visible_fields_json"] = json.dumps(sorted(visible))
        print(json.dumps({"package": str(args.package), "train": len(train["wearable_x"]),
                          "validation": len(validation["wearable_x"]), "scale": scale.tolist(),
                          "tau": tau.tolist()}))


if __name__ == "__main__":
    main()
