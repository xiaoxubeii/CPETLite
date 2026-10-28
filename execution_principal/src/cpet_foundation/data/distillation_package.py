from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from cpet_foundation.model.wearable_student import TEACHER_ONLY_FORBIDDEN_BATCH_KEYS


DISTILLATION_PACKAGE_SCHEMA_VERSION = 1
DISTILLATION_CONSTANT_KEYS = frozenset(
    {
        "shape_feature_indices",
        "direct_loss_feature_indices",
        "core_derived_feature_indices",
        "slope_loss_feature_indices",
        "peak_loss_feature_indices",
        "derived_feature_indices",
        "derived_loss_feature_indices",
        "distillation_feature_indices",
        "distillation_relation_feature_indices",
        "gas_exchange_feature_indices",
        "initial_boundary_steps",
        "peak_timing_beta",
        "core_at_bandwidth",
        "core_at_weight",
        "core_post_at_weight",
        "gas_exchange_min_vo2",
        "gas_exchange_min_vco2",
        "gas_exchange_at_bandwidth",
        "gas_exchange_at_weight",
        "gas_exchange_post_at_weight",
        "distillation_at_bandwidth",
        "selective_distillation_margin",
        "selective_at_weight",
        "selective_at_region_weight",
        "selective_relation_weight",
    }
)
DISTILLATION_REQUIRED_SAMPLE_KEYS = frozenset(
    {
        "wearable_x",
        "wearable_mask",
        "protocol_x",
        "protocol_mask",
        "mask",
        "phase",
        "metadata",
        "teacher_latent_rollout",
        "teacher_trajectory_mu",
        "teacher_trajectory_logvar",
        "teacher_at_norm",
        "teacher_vo2peak",
    }
)
_MASK_FIELDS = frozenset(
    {
        "metadata_mask",
        "phase_mask",
        "wearable_mask",
        "protocol_mask",
        "mask",
        "future_mask",
        "shape_mask",
        "task_evaluation_mask",
        "target_channel_mask",
        "teacher_vo2peak_mask",
    }
)


def _numpy_for_storage(name: str, value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if name == "phase":
        return array.astype(np.int16, copy=False)
    if name in _MASK_FIELDS:
        return (array > 0).astype(np.uint8, copy=False)
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32, copy=False)
    return array


def _torch_from_storage(name: str, array: np.ndarray) -> torch.Tensor:
    value = torch.from_numpy(np.asarray(array))
    if name == "phase" or name.endswith("_indices"):
        return value.long()
    if name in _MASK_FIELDS or value.is_floating_point():
        return value.float()
    return value


def _write_split(group: h5py.Group, batches: Iterable[dict[str, torch.Tensor]]) -> tuple[int, set[str]]:
    sample_group = group.create_group("samples")
    constant_group = group.create_group("constants")
    sample_datasets: dict[str, h5py.Dataset] = {}
    constant_values: dict[str, np.ndarray] = {}
    visible_fields: set[str] = set()
    total = 0

    for batch_index, batch in enumerate(batches):
        exposed = sorted(TEACHER_ONLY_FORBIDDEN_BATCH_KEYS.intersection(batch))
        if exposed:
            raise ValueError(f"distillation package contains forbidden measured target(s): {exposed}")
        missing = sorted(DISTILLATION_REQUIRED_SAMPLE_KEYS.difference(batch))
        if missing:
            raise ValueError(f"distillation package batch is missing required field(s): {missing}")
        if not all(isinstance(value, torch.Tensor) for value in batch.values()):
            raise TypeError("distillation package batches must contain tensors only")

        batch_size = int(batch["wearable_x"].shape[0])
        if batch_size < 1:
            continue
        for name, value in batch.items():
            visible_fields.add(name)
            array = _numpy_for_storage(name, value)
            if name in DISTILLATION_CONSTANT_KEYS:
                if batch_index == 0:
                    constant_values[name] = array
                    constant_group.create_dataset(name, data=array)
                elif name not in constant_values or not np.array_equal(constant_values[name], array):
                    raise ValueError(f"distillation constant changed across batches: {name}")
                continue
            if array.ndim == 0 or int(array.shape[0]) != batch_size:
                raise ValueError(f"sample field {name!r} does not have batch-leading shape")
            if name not in sample_datasets:
                chunks = (min(64, batch_size), *array.shape[1:])
                sample_datasets[name] = sample_group.create_dataset(
                    name,
                    shape=(0, *array.shape[1:]),
                    maxshape=(None, *array.shape[1:]),
                    chunks=chunks,
                    dtype=array.dtype,
                    compression="lzf",
                )
            dataset = sample_datasets[name]
            if tuple(dataset.shape[1:]) != tuple(array.shape[1:]):
                raise ValueError(f"sample field shape changed across batches: {name}")
            dataset.resize(total + batch_size, axis=0)
            dataset[total : total + batch_size] = array
        total += batch_size

    if total == 0:
        raise ValueError(f"distillation split {group.name!r} contains no samples")
    group.attrs["sample_count"] = total
    return total, visible_fields


def write_distillation_package(
    path: Path,
    split_batches: Mapping[str, Iterable[dict[str, torch.Tensor]]],
    *,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {}
    visible_fields: set[str] = set()
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = DISTILLATION_PACKAGE_SCHEMA_VERSION
        handle.attrs["supervision_mode"] = "teacher_only"
        handle.attrs["floating_storage_dtype"] = "float32"
        handle.attrs["metadata_json"] = json.dumps(dict(metadata or {}), sort_keys=True)
        splits_group = handle.create_group("splits")
        for split_name, batches in split_batches.items():
            count, fields = _write_split(splits_group.create_group(split_name), batches)
            split_counts[split_name] = count
            visible_fields.update(fields)
        handle.attrs["student_visible_fields_json"] = json.dumps(sorted(visible_fields))
    return {
        "path": str(path),
        "schema_version": DISTILLATION_PACKAGE_SCHEMA_VERSION,
        "splits": split_counts,
        "student_visible_fields": sorted(visible_fields),
        "metadata": dict(metadata or {}),
    }


def audit_distillation_package(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        split_counts = {
            name: int(group.attrs["sample_count"])
            for name, group in handle["splits"].items()
        }
        return {
            "path": str(Path(path)),
            "schema_version": int(handle.attrs["schema_version"]),
            "supervision_mode": str(handle.attrs["supervision_mode"]),
            "splits": split_counts,
            "student_visible_fields": json.loads(str(handle.attrs["student_visible_fields_json"])),
            "metadata": json.loads(str(handle.attrs["metadata_json"])),
        }


class DistillationPackageDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, split: str, *, preload: bool = False) -> None:
        self.path = Path(path)
        self.split = split
        self._handle: h5py.File | None = None
        self._preloaded: dict[str, np.ndarray] | None = None
        with h5py.File(self.path, "r") as handle:
            if int(handle.attrs.get("schema_version", -1)) != DISTILLATION_PACKAGE_SCHEMA_VERSION:
                raise ValueError("unsupported distillation package schema")
            if str(handle.attrs.get("supervision_mode", "")) != "teacher_only":
                raise ValueError("distillation package is not teacher_only")
            if split not in handle["splits"]:
                raise KeyError(f"distillation package has no split {split!r}")
            group = handle["splits"][split]
            self._length = int(group.attrs["sample_count"])
            self.sample_keys = tuple(sorted(group["samples"].keys()))
            self.constants = {
                name: _torch_from_storage(name, dataset[...])
                for name, dataset in group["constants"].items()
            }
            if preload:
                self._preloaded = {
                    name: group["samples"][name][...]
                    for name in self.sample_keys
                }
        exposed = TEACHER_ONLY_FORBIDDEN_BATCH_KEYS.intersection(self.sample_keys)
        if exposed:
            raise ValueError(f"package exposes forbidden measured target(s): {sorted(exposed)}")

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int | list[int] | np.ndarray) -> dict[str, torch.Tensor]:
        if isinstance(index, (list, tuple, np.ndarray)):
            return self.batch(index)
        if self._preloaded is not None:
            return {
                name: _torch_from_storage(name, values[index])
                for name, values in self._preloaded.items()
            }
        samples = self._file()["splits"][self.split]["samples"]
        return {name: _torch_from_storage(name, samples[name][index]) for name in self.sample_keys}

    def batch(self, indices: list[int] | tuple[int, ...] | np.ndarray) -> dict[str, torch.Tensor]:
        """Fetch a complete batch with one array conversion per package field."""
        index = np.asarray(indices, dtype=np.int64)
        if self._preloaded is not None:
            batch = {
                name: _torch_from_storage(name, values[index])
                for name, values in self._preloaded.items()
            }
        else:
            samples = self._file()["splits"][self.split]["samples"]
            # h5py requires monotonically increasing fancy indices. Preserve
            # sampler order after one sorted read per field.
            order = np.argsort(index)
            inverse = np.argsort(order)
            sorted_index = index[order]
            batch = {
                name: _torch_from_storage(name, samples[name][sorted_index][inverse])
                for name in self.sample_keys
            }
        batch.update({name: value.clone() for name, value in self.constants.items()})
        return batch

    def collate(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch = default_collate(samples)
        batch.update({name: value.clone() for name, value in self.constants.items()})
        return batch

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()
