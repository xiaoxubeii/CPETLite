"""Fixed, input-only package view for measured-supervision comparisons."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from cpet_foundation.data.sequences import examination_token


INPUT_FIELDS = (
    "wearable_x", "wearable_mask", "protocol_x", "protocol_mask", "mask",
    "phase", "phase_mask", "metadata", "metadata_mask", "body_mass_kg", "time_fraction",
    "future_mask", "shape_mask", "task_evaluation_mask",
)
TRAINING_SEMANTICS_VERSION = "physical-time-matched-v3-finite-metadata"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_precision(config: dict) -> str:
    value = str(config.get("mixed_precision", "none")).strip().lower()
    if value not in {"none", "bf16", "bfloat16"}:
        raise ValueError("mixed_precision must be none or bfloat16")
    return "bfloat16" if value == "bf16" else value


def input_contract(config: dict) -> dict:
    from cpet_foundation.training.runner import (
        resolve_wearable_task_mode, wearable_simulation_config_from_training_config,
    )
    return json.loads(json.dumps({
        "version": 2, "storage_dtype": "float32", "compute_precision": compute_precision(config),
        "feature_columns": list(config["feature_columns"]),
        "metadata_columns": list(config["metadata_columns"]),
        "max_len": int(config.get("max_len", 160)),
        "task_mode": resolve_wearable_task_mode(config),
        "simulation": asdict(wearable_simulation_config_from_training_config(config)),
    }, sort_keys=True))


def checked_matched_package(path: Path, *, config: dict | None = None,
                            source: Path | None = None, teacher: Path | None = None) -> dict:
    """Reject legacy/stale packages at the matched-paper execution boundary."""
    with h5py.File(path, "r") as handle:
        metadata = json.loads(str(handle.attrs.get("metadata_json", "{}")))
        contract = metadata.get("matched_input_contract")
        if (int(handle.attrs.get("schema_version", -1)) != 1
                or str(handle.attrs.get("supervision_mode", "")) != "teacher_only"
                or str(handle.attrs.get("floating_storage_dtype", "")) != "float32"
                or not isinstance(contract, dict) or contract.get("version") != 2
                or contract.get("storage_dtype") != "float32"):
            raise ValueError("Matched paper inputs require a new float32 package with a matching contract")
        if config is not None and contract != input_contract(config):
            raise ValueError("Matched package configuration/precision contract differs")
        for label, reference in (("source_h5", source), ("teacher_checkpoint", teacher)):
            if reference is not None and metadata.get(label + "_sha256") != sha256(reference):
                raise ValueError(f"Matched package {label} checksum differs")
        token_sets = []
        for split in ("train", "validation"):
            samples = handle.get(f"splits/{split}/samples")
            if samples is None or set((*INPUT_FIELDS, "sample_token")) - set(samples):
                raise ValueError(f"Matched package lacks fixed inputs/time/tokens: {split}")
            token_node = samples["sample_token"]
            tokens = token_node[:]
            if tokens.ndim != 2 or tokens.shape[1] != 2 or tokens.dtype.kind not in "iu":
                raise ValueError("Invalid matched package examination tokens")
            token_set = {tuple(row) for row in tokens.tolist()}
            if len(token_set) != len(tokens) or len(tokens) == 0:
                raise ValueError("Empty or duplicate matched package examination tokens")
            for name in INPUT_FIELDS:
                node = samples[name]
                if len(node) != len(tokens) or (node.dtype.kind == "f" and node.dtype != np.dtype("float32")):
                    raise ValueError(f"Matched package input length/dtype differs: {split}/{name}")
            token_sets.append(token_set)
        if token_sets[0] & token_sets[1]:
            raise ValueError("Matched package examinations overlap training/validation")
    return {"sha256": sha256(path), "metadata": metadata, "contract": contract}


def checked_matched_fit(directory: Path, package: dict, *, measured: bool) -> dict:
    """Existing checkpoints need actual matching provenance before resume."""
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        raise ValueError("Existing matched fit requires a fitting manifest")
    record = json.loads(path.read_text())
    if record.get("training_semantics_version") != TRAINING_SEMANTICS_VERSION:
        raise ValueError("Existing matched fit predates the current timing/validation corrections")
    if record.get("mixed_precision") != package["contract"]["compute_precision"]:
        raise ValueError("Existing matched fit precision differs")
    if measured:
        provenance = record.get("matched_input_provenance") or {}
        for split in ("train", "validation"):
            evidence = provenance.get(split, {})
            if (evidence.get("sha256") != package["sha256"]
                    or evidence.get("source_h5_sha256") != package["metadata"].get("source_h5_sha256")
                    or evidence.get("contract") != package["contract"]
                    or evidence.get("teacher_targets_read") is not False):
                raise ValueError("Existing measured fit lacks matched input provenance")
        if record.get("teacher_accessed_during_training") is not False:
            raise ValueError("Matched measured reference accessed a teacher")
    elif (record.get("distillation_package_sha256") != package["sha256"]
          or record.get("matched_input_contract") != package["contract"]
          or record.get("raw_cpet_opened_by_student_trainer") is not False
          or record.get("measured_training_targets_available") is not False):
        raise ValueError("Existing package fit lacks matched isolated-input provenance")
    return record


class MatchedInputDataset(torch.utils.data.Dataset):
    """Join allowlisted inputs only; never attach teacher targets to raw data.

    Hash tokens link examinations, not patients. Arrays are loaded once so a
    DataLoader worker does not inherit an open HDF5 handle.
    """
    def __init__(self, source, path: Path, split: str, *, config: dict):
        self.source = source
        # Preserve source-order metadata for evaluation/prediction exports. The
        # package join only changes model inputs, never sample identities.
        self.samples = source.samples
        self.config = source.config
        self.path = Path(path)
        with h5py.File(path, "r") as handle:
            if int(handle.attrs.get("schema_version", -1)) != 1 or str(handle.attrs.get("supervision_mode", "")) != "teacher_only":
                raise ValueError("invalid matched input package schema or supervision mode")
            metadata = json.loads(str(handle.attrs.get("metadata_json", "{}")))
            if metadata.get("matched_input_contract") != input_contract(config):
                raise ValueError("matched input package contract mismatch; re-export with the current configuration")
            if metadata["matched_input_contract"]["task_mode"] != "full_reconstruction":
                raise ValueError("matched inputs currently require full_reconstruction")
            if metadata.get("source_h5_sha256") != sha256(source.config.source_h5):
                raise ValueError("matched input source checksum mismatch")
            if str(handle.attrs.get("floating_storage_dtype", "")) != "float32":
                raise ValueError("matched input package requires a new float32 export")
            samples = handle[f"splits/{split}/samples"]
            absent = set((*INPUT_FIELDS, "sample_token")) - set(samples)
            if absent:
                raise ValueError(f"matched input package missing fields: {sorted(absent)}")
            tokens = np.asarray(samples["sample_token"])
            if tokens.ndim != 2 or tokens.shape[1] != 2 or tokens.dtype.kind not in "iu":
                raise ValueError("invalid matched input examination tokens")
            lookup = {tuple(row): i for i, row in enumerate(tokens.tolist())}
            if len(lookup) != len(tokens):
                raise ValueError("duplicate matched input examination tokens")
            expected = [examination_token(sample["exam_id"]) for sample in source.samples]
            if len(set(expected)) != len(expected) or set(expected) != set(lookup):
                raise ValueError("matched input examination membership mismatch")
            self.indices = [lookup[key] for key in expected]
            self.inputs = {}
            for name in INPUT_FIELDS:
                node = samples[name]
                if len(node) != len(tokens):
                    raise ValueError(f"matched input field length mismatch: {name}")
                if node.dtype.kind == "f" and node.dtype != np.dtype("float32"):
                    raise ValueError(f"matched input field is not float32: {name}")
                values = node[:]
                if not np.isfinite(values).all():
                    raise ValueError(f"non-finite matched input: {name}")
                tensor = torch.from_numpy(values)
                self.inputs[name] = tensor.long() if name == "phase" else tensor.float()
        self.provenance = {
            "path": str(self.path.resolve()), "sha256": sha256(self.path), "split": split,
            "source_h5_sha256": metadata["source_h5_sha256"],
            "contract": metadata["matched_input_contract"], "examinations": len(self.indices),
            "teacher_targets_read": False,
        }

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        raw = self.source[index]
        row = self.indices[index]
        return {**raw, **{f"fixed_{key}": value[row].clone() for key, value in self.inputs.items()}}
