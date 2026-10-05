"""Deterministic subject-wise loaders for the APAVA and ADFTD Ours experiments."""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetBundle:
    dataset: str
    x: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    window_ids: np.ndarray
    splits: Dict[str, List[int]]
    num_classes: int
    class_names: Dict[int, str]


class EEGWindowDataset(Dataset):
    def __init__(self, bundle: DatasetBundle, subject_ids: List[int]):
        keep = np.isin(bundle.subject_ids, np.asarray(subject_ids, dtype=np.int64))
        self.x = torch.from_numpy(bundle.x[keep])
        self.y = torch.from_numpy(bundle.y[keep].astype(np.int64, copy=False))
        self.subject_ids = torch.from_numpy(bundle.subject_ids[keep].astype(np.int64, copy=False))
        self.window_ids = torch.from_numpy(bundle.window_ids[keep].astype(np.int64, copy=False))

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.subject_ids[idx], self.window_ids[idx]


def _feature_index(feature_dir: str) -> Dict[int, str]:
    result = {}
    for name in os.listdir(feature_dir):
        match = re.fullmatch(r"feature_(\d+)\.npy", name, flags=re.IGNORECASE)
        if match:
            sid = int(match.group(1))
            if sid in result:
                raise ValueError(f"Duplicate feature file for subject {sid}: {result[sid]}, {name}")
            result[sid] = os.path.join(feature_dir, name)
    if not result:
        raise FileNotFoundError(f"No feature_*.npy files in {feature_dir}")
    return result


def _subject_splits(dataset: str, labels: np.ndarray) -> Dict[str, List[int]]:
    rows = [(int(row[1]), int(row[0])) for row in labels]
    if dataset == "APAVA":
        val = [15, 16, 19, 20]
        test = [1, 2, 17, 18]
        known = {sid for sid, _ in rows}
        if not set(val + test).issubset(known):
            raise ValueError("APAVA fixed TeCh validation/test IDs are missing from label.npy")
        train = [sid for sid, _ in rows if sid not in set(val + test)]
        return {"train": train, "val": val, "test": test}

    if dataset == "ADFTD":
        train, val, test = [], [], []
        for cls in sorted({label for _, label in rows}):
            ids = [sid for sid, label in rows if label == cls]
            n = len(ids)
            a, b = int(0.6 * n), int(0.8 * n)
            train.extend(ids[:a])
            val.extend(ids[a:b])
            test.extend(ids[b:])
        return {"train": train, "val": val, "test": test}

    raise ValueError(f"Unsupported dataset: {dataset}")


def load_dataset(dataset: str, root_path: str) -> DatasetBundle:
    dataset = dataset.upper()
    data_dir = os.path.join(root_path, "APAVA" if dataset == "APAVA" else "ADFD")
    feature_dir = os.path.join(data_dir, "Feature")
    label_path = os.path.join(data_dir, "Label", "label.npy")
    labels = np.load(label_path, allow_pickle=False)
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError(f"Expected label.npy with [subject, (class,id)] rows; got {labels.shape}")

    files = _feature_index(feature_dir)
    subject_labels = {int(row[1]): int(row[0]) for row in labels}
    if len(subject_labels) != labels.shape[0]:
        raise ValueError("Duplicate subject IDs found in label.npy")
    missing = sorted(set(subject_labels) - set(files))
    extra = sorted(set(files) - set(subject_labels))
    if missing or extra:
        raise ValueError(f"Feature/label subject mismatch; missing={missing}, extra={extra}")

    shapes = {}
    total = 0
    channels = None
    sequence_length = None
    for sid in subject_labels:
        arr = np.load(files[sid], mmap_mode="r", allow_pickle=False)
        if arr.ndim != 3:
            raise ValueError(f"{files[sid]} must have [windows,T,C], got {arr.shape}")
        if sequence_length is None:
            _, sequence_length, channels = arr.shape
        if arr.shape[1:] != (sequence_length, channels):
            raise ValueError(f"Inconsistent feature shape for subject {sid}: {arr.shape}")
        shapes[sid] = arr.shape[0]
        total += arr.shape[0]

    # Preallocate once to avoid a second dataset-sized copy during concatenation.
    x = np.empty((total, sequence_length, channels), dtype=np.float32)
    y = np.empty(total, dtype=np.int64)
    subject_ids = np.empty(total, dtype=np.int64)
    window_ids = np.empty(total, dtype=np.int64)
    cursor = 0
    for sid in subject_labels:
        raw = np.load(files[sid], allow_pickle=False)
        raw64 = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(raw64).all():
            raise ValueError(f"Non-finite values found in subject {sid}")
        mean = raw64.mean(axis=1, keepdims=True)
        std = raw64.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        normalized = ((raw64 - mean) / std).astype(np.float32, copy=False)
        count = raw.shape[0]
        x[cursor : cursor + count] = normalized
        y[cursor : cursor + count] = subject_labels[sid]
        subject_ids[cursor : cursor + count] = sid
        window_ids[cursor : cursor + count] = np.arange(count, dtype=np.int64)
        cursor += count
        del raw, raw64, normalized

    if not np.isfinite(x).all():
        raise ValueError("Non-finite values remain after normalization")
    splits = _subject_splits(dataset, labels)
    split_sets = {name: set(ids) for name, ids in splits.items()}
    if any(split_sets[a] & split_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise AssertionError("Subject leakage across splits")
    if set.union(*split_sets.values()) != set(subject_labels):
        raise AssertionError("Splits do not cover every labeled subject")

    class_ids = sorted(set(subject_labels.values()))
    if dataset == "APAVA":
        class_names = {0: "HC", 1: "AD"}
    else:
        class_names = {0: "HC", 1: "FTD", 2: "AD"}
    if set(class_ids) != set(class_names):
        raise ValueError(f"Unexpected {dataset} labels: {class_ids}")

    return DatasetBundle(
        dataset=dataset,
        x=x,
        y=y,
        subject_ids=subject_ids,
        window_ids=window_ids,
        splits=splits,
        num_classes=len(class_names),
        class_names=class_names,
    )


def split_indices(bundle: DatasetBundle) -> Dict[str, np.ndarray]:
    return {
        split: np.flatnonzero(np.isin(bundle.subject_ids, np.asarray(ids, dtype=np.int64)))
        for split, ids in bundle.splits.items()
    }

