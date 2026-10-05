"""Reproducible subject-wise training, evaluation, and diagnostics for Ours."""

import csv
import json
import os
import random
import time
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from experiments.ours_data import DatasetBundle, EEGWindowDataset, load_dataset
from models.Ours import Ours


DATASET_SETTINGS = {
    "APAVA": {
        "d_model": 256,
        "learning_rate": 1e-4,
        "max_epochs": 40,
        "patience": 20,
        "augmentations": ["flip0.2", "frequency0.2", "jitter0.", "mask0.", "channel0.", "drop0.4"],
    },
    "ADFTD": {
        "d_model": 128,
        "learning_rate": 3e-5,
        "max_epochs": 6,
        "patience": 6,
        "augmentations": ["flip0.", "frequency0.", "jitter0.", "mask0.25", "channel0.", "drop0."],
    },
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _metrics(labels: np.ndarray, probabilities: np.ndarray, n_classes: int):
    predictions = probabilities.argmax(axis=1)
    labels = labels.astype(np.int64, copy=False)
    one_hot = np.eye(n_classes, dtype=np.float64)[labels]
    if n_classes == 2:
        auroc = roc_auc_score(labels, probabilities[:, 1])
        auprc = average_precision_score(labels, probabilities[:, 1])
    else:
        auroc = roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
        auprc = average_precision_score(one_hot, probabilities, average="macro")
    return {
        "Accuracy": float(accuracy_score(labels, predictions)),
        "Precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "Recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "F1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
    }, predictions


def _make_loader(bundle, split, batch_size, shuffle, seed, num_workers=0):
    ds = EEGWindowDataset(bundle, bundle.splits[split])
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def calibrate_shared_basis(model: Ours, train_loader, device):
    """Estimate the shared basis from the training fold only, without augmentation."""
    previous_training = model.training
    model.eval()
    laplacian_sum = torch.zeros(model.n_nodes, model.n_nodes, dtype=torch.float64, device="cpu")
    sample_count = 0
    for batch_x, _, _, _ in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        nodes, _ = model.encode_nodes(batch_x, return_attention=False)
        _, _, laplacian = model.build_graph(nodes)
        laplacian_sum += laplacian.double().sum(dim=0).cpu()
        sample_count += int(batch_x.shape[0])
    if sample_count == 0:
        raise RuntimeError("Cannot calibrate the shared basis: empty training fold")
    mean_laplacian = laplacian_sum / sample_count
    mean_laplacian = 0.5 * (mean_laplacian + mean_laplacian.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(mean_laplacian)
    model.set_shared_basis(eigenvectors.float(), eigenvalues.float())
    if previous_training:
        model.train()
    return eigenvalues.float(), eigenvectors.float(), sample_count


@torch.no_grad()
def evaluate(model, loader, device, n_classes, sample_weighted=False):
    model.eval()
    losses = []
    all_labels, all_probs, all_subjects, all_windows = [], [], [], []
    criterion = nn.CrossEntropyLoss()
    for batch_x, labels, subjects, windows in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        logits = model(batch_x)
        losses.append(float(criterion(logits, labels_device).item()) * (len(labels) if sample_weighted else 1))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_labels.append(labels.numpy())
        all_probs.append(probs)
        all_subjects.append(subjects.numpy())
        all_windows.append(windows.numpy())
    labels = np.concatenate(all_labels).astype(np.int64, copy=False)
    probs = np.concatenate(all_probs)
    subjects = np.concatenate(all_subjects).astype(np.int64, copy=False)
    windows = np.concatenate(all_windows).astype(np.int64, copy=False)
    metrics, predictions = _metrics(labels, probs, n_classes)
    return {
        "loss": float(np.sum(losses) / len(labels) if sample_weighted else np.mean(losses)),
        "metrics": metrics,
        "labels": labels,
        "probabilities": probs,
        "predictions": predictions,
        "subject_ids": subjects,
        "window_ids": windows,
    }


def _diagnostic_examples(bundle: DatasetBundle):
    selected = {}
    val_ids = set(bundle.splits["val"])
    for idx, (label, subject) in enumerate(zip(bundle.y, bundle.subject_ids)):
        if int(subject) not in val_ids or int(label) in selected:
            continue
        selected[int(label)] = idx
        if len(selected) == bundle.num_classes:
            break
    if len(selected) != bundle.num_classes:
        raise RuntimeError("Could not select at least one validation example per class")
    return selected


@torch.no_grad()
def save_epoch_diagnostics(model, bundle, selected, epoch_dir, device):
    model.eval()
    os.makedirs(epoch_dir, exist_ok=True)
    for label, idx in selected.items():
        x = torch.from_numpy(bundle.x[idx : idx + 1]).to(device)
        logits, details = model(x, return_details=True)
        adjacency = details["adjacency"][0].cpu().numpy()
        edge_i, edge_j = np.triu(adjacency, k=1).nonzero()
        np.savez_compressed(
            os.path.join(epoch_dir, f"class_{label}_{bundle.class_names[label]}_subject_{int(bundle.subject_ids[idx])}_window_{int(bundle.window_ids[idx])}.npz"),
            attention=details["attention"][:, 0].cpu().numpy(),
            adjacency=adjacency,
            edge_index=np.stack([edge_i, edge_j], axis=0).astype(np.int32),
            edge_weight=adjacency[edge_i, edge_j].astype(np.float32),
            alpha=details["alpha"].cpu().numpy(),
            gate=details["gate"][0].cpu().numpy(),
            node_band_energy=details["node_band_energy"][0].cpu().numpy(),
            logits=logits[0].cpu().numpy(),
            label=np.int64(label),
            subject_id=np.int64(bundle.subject_ids[idx]),
            window_id=np.int64(bundle.window_ids[idx]),
            tau1=np.float32(details["tau1"].cpu().item()),
            tau2=np.float32(details["tau2"].cpu().item()),
            eigenvalues=details["eigenvalues"].cpu().numpy(),
        )


def _write_split_files(bundle: DatasetBundle, output_dir: str):
    count_by_subject = {
        int(sid): int(np.sum(bundle.subject_ids == sid)) for sid in bundle.splits["train"] + bundle.splits["val"] + bundle.splits["test"]
    }
    rows = []
    for split, ids in bundle.splits.items():
        for sid in ids:
            label = int(bundle.y[np.flatnonzero(bundle.subject_ids == sid)[0]])
            rows.append({
                "subject_id": int(sid),
                "label": label,
                "class_name": bundle.class_names[label],
                "split": split,
                "window_count": count_by_subject[int(sid)],
            })
    pd.DataFrame(rows).sort_values(["split", "subject_id"]).to_csv(
        os.path.join(output_dir, "subject_split.csv"), index=False
    )


def _save_prediction_file(result, bundle, output_path):
    table = {
        "subject_id": result["subject_ids"],
        "window_id": result["window_ids"],
        "label": result["labels"],
        "prediction": result["predictions"],
    }
    for cls in range(bundle.num_classes):
        table[f"prob_{cls}_{bundle.class_names[cls]}"] = result["probabilities"][:, cls]
    pd.DataFrame(table).to_csv(output_path, index=False)


def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=lambda value: float(value))


def run_seed(bundle, dataset_root, output_dir, seed, args, device):
    seed_everything(seed)
    seed_dir = os.path.join(output_dir, bundle.dataset, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    train_loader = _make_loader(bundle, "train", args.batch_size, True, seed)
    train_basis_loader = _make_loader(bundle, "train", args.batch_size, False, seed)
    val_loader = _make_loader(bundle, "val", args.batch_size, False, seed)
    test_loader = _make_loader(bundle, "test", args.batch_size, False, seed)

    settings = DATASET_SETTINGS[bundle.dataset]
    model_config = {
        "seq_len": int(bundle.x.shape[1]),
        "enc_in": int(bundle.x.shape[2]),
        "num_class": int(bundle.num_classes),
        "patch_len": args.patch_len,
        "stride": args.patch_stride,
        "d_model": args.d_model if args.d_model is not None else settings["d_model"],
        "n_heads": args.n_heads,
        "e_layers": args.e_layers,
        "dropout": args.dropout,
        "top_k": args.top_k,
        "self_loop": args.self_loop,
        "soft_temperature": args.soft_temperature,
        "augmentations": settings["augmentations"] if not args.disable_augmentation else ["none"],
    }
    model = Ours(**model_config).to(device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate if args.learning_rate else settings["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    max_epochs = args.epochs if args.epochs is not None else settings["max_epochs"]
    patience = args.patience if args.patience is not None else settings["patience"]
    selected_examples = _diagnostic_examples(bundle)
    diagnostics_root = os.path.join(seed_dir, "visualization_data")
    os.makedirs(diagnostics_root, exist_ok=True)

    # Node order: channel-major, then patch index; saved once for downstream plots.
    node_rows = [
        {"node_index": i, "channel_index": i // model.n_patches, "patch_index": i % model.n_patches}
        for i in range(model.n_nodes)
    ]
    pd.DataFrame(node_rows).to_csv(os.path.join(seed_dir, "node_mapping.csv"), index=False)

    env = {
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_version": torch.version.cuda,
    }
    _save_json(os.path.join(seed_dir, "environment.json"), env)
    _save_json(os.path.join(seed_dir, "config.json"), {
        "model_name": "ours",
        "dataset": bundle.dataset,
        "seed": seed,
        "model": model_config,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate if args.learning_rate else settings["learning_rate"],
        "max_epochs": max_epochs,
        "patience": patience,
        "selection_metric": "validation_macro_F1",
        "num_parameters": total_params,
        "num_trainable_parameters": trainable_params,
        "basis_policy": "train-fold mean normalized Laplacian, refreshed after each epoch and fixed within epoch",
        "dataset_root": os.path.abspath(dataset_root),
    })

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    basis_start = time.perf_counter()
    eigenvalues, eigenvectors, basis_samples = calibrate_shared_basis(model, train_basis_loader, device)
    _sync(device)
    basis_init_time = time.perf_counter() - basis_start
    basis_history = {"epoch_0_eigenvalues": eigenvalues.cpu().numpy(), "epoch_0_eigenvectors": eigenvectors.cpu().numpy()}

    best_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        epoch_start = time.perf_counter()
        model.train()
        train_losses = []
        train_samples = 0
        train_start = time.perf_counter()
        for batch_x, labels, _, _ in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_samples += int(batch_x.shape[0])
        _sync(device)
        train_sec = time.perf_counter() - train_start

        val_start = time.perf_counter()
        val_result = evaluate(model, val_loader, device, bundle.num_classes)
        save_epoch_diagnostics(
            model,
            bundle,
            selected_examples,
            os.path.join(diagnostics_root, f"epoch_{epoch:03d}"),
            device,
        )
        _sync(device)
        val_sec = time.perf_counter() - val_start

        val_f1 = val_result["metrics"]["F1"]
        improved = (val_f1 > best_f1 + 1e-12) or (
            abs(val_f1 - best_f1) <= 1e-12 and val_result["loss"] < best_val_loss
        )
        if improved:
            best_f1 = val_f1
            best_val_loss = val_result["loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                    "val_f1": best_f1,
                    "val_loss": best_val_loss,
                    "model_config": model_config,
                },
                os.path.join(seed_dir, "best.pt"),
            )
        else:
            stale_epochs += 1

        # Refresh the reference basis only after this epoch's validation/selection.
        # The current epoch uses the same U for train, validation, and diagnostics.
        basis_refresh_sec = 0.0
        if epoch < max_epochs and stale_epochs < patience:
            _sync(device)
            refresh_start = time.perf_counter()
            eigenvalues, eigenvectors, basis_samples = calibrate_shared_basis(model, train_basis_loader, device)
            _sync(device)
            basis_refresh_sec = time.perf_counter() - refresh_start
            basis_history[f"epoch_{epoch}_eigenvalues"] = eigenvalues.cpu().numpy()
            basis_history[f"epoch_{epoch}_eigenvectors"] = eigenvectors.cpu().numpy()

        _sync(device)
        epoch_sec = time.perf_counter() - epoch_start
        peak_mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
            if device.type == "cuda"
            else 0.0
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_result["loss"],
            **{f"val_{key.lower()}": value for key, value in val_result["metrics"].items()},
            "train_seconds": train_sec,
            "validation_seconds": val_sec,
            "basis_refresh_seconds": basis_refresh_sec,
            "epoch_seconds": epoch_sec,
            "train_samples_per_second": train_samples / max(train_sec, 1e-12),
            "peak_gpu_memory_mb": peak_mem_mb,
            "basis_calibration_samples": basis_samples,
            "best_epoch_so_far": best_epoch,
            "is_best": int(improved),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(seed_dir, "epochs.csv"), index=False)
        print(
            f"[{bundle.dataset} seed={seed}] epoch={epoch}/{max_epochs} "
            f"train_loss={row['train_loss']:.4f} val_F1={val_f1:.4f} "
            f"val_AUROC={val_result['metrics']['AUROC']:.4f} "
            f"train={train_sec:.2f}s basis={basis_refresh_sec:.2f}s total={epoch_sec:.2f}s "
            f"peak={peak_mem_mb:.0f}MB"
        )
        if stale_epochs >= patience:
            print(f"[{bundle.dataset} seed={seed}] early stopping at epoch {epoch}; best epoch {best_epoch}")
            break

    np.savez_compressed(os.path.join(seed_dir, "basis_history.npz"), **basis_history)
    checkpoint = torch.load(os.path.join(seed_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_result = evaluate(model, test_loader, device, bundle.num_classes)
    _save_prediction_file(test_result, bundle, os.path.join(seed_dir, "test_predictions.csv"))
    _save_json(os.path.join(seed_dir, "test_metrics.json"), {
        "dataset": bundle.dataset,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "test_loss": test_result["loss"],
        **test_result["metrics"],
    })

    summary = {
        "dataset": bundle.dataset,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "test_loss": test_result["loss"],
        **test_result["metrics"],
        "num_parameters": total_params,
        "num_trainable_parameters": trainable_params,
        "basis_initialization_seconds": basis_init_time,
        "last_epoch_seconds": history[-1]["epoch_seconds"],
        "mean_epoch_seconds": float(np.mean([item["epoch_seconds"] for item in history])),
        "mean_train_samples_per_second": float(np.mean([item["train_samples_per_second"] for item in history])),
        "max_peak_gpu_memory_mb": float(np.max([item["peak_gpu_memory_mb"] for item in history])),
        "test_sample_count": int(len(test_result["labels"])),
    }
    _save_json(os.path.join(seed_dir, "run_summary.json"), summary)
    return summary


def run_dataset(dataset: str, dataset_root: str, output_dir: str, args, device):
    print(f"Loading {dataset} from {dataset_root}")
    bundle = load_dataset(dataset, dataset_root)
    os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)
    _write_split_files(bundle, os.path.join(output_dir, dataset))
    manifest = {
        "dataset": dataset,
        "root_path": os.path.abspath(dataset_root),
        "shape": {"samples": int(bundle.x.shape[0]), "time": int(bundle.x.shape[1]), "channels": int(bundle.x.shape[2])},
        "class_names": {str(key): value for key, value in bundle.class_names.items()},
        "split_subject_counts": {key: len(value) for key, value in bundle.splits.items()},
        "split_sample_counts": {
            split: int(np.isin(bundle.subject_ids, ids).sum()) for split, ids in bundle.splits.items()
        },
        "normalization": "per-window, per-channel z-score over the 256 time points; zero std replaced by one",
        "augmentation": "TeCh dataset-specific augmentation menu during training only",
        "test_usage": "test evaluated once after validation-macro-F1 checkpoint selection",
        "device": str(device),
    }
    _save_json(os.path.join(output_dir, dataset, "manifest.json"), manifest)
    print(
        f"{dataset}: shape={bundle.x.shape}, splits="
        f"{manifest['split_sample_counts']}, device={device}"
    )

    summaries = []
    for seed in args.seeds:
        summaries.append(run_seed(bundle, dataset_root, output_dir, seed, args, device))
    pd.DataFrame(summaries).to_csv(
        os.path.join(output_dir, dataset, "seed_summary.csv"), index=False
    )
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC"]
    aggregate = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in summaries], dtype=np.float64)
        aggregate[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1) if len(values) > 1 else 0.0)}
    aggregate["num_parameters"] = summaries[0]["num_parameters"]
    aggregate["num_trainable_parameters"] = summaries[0]["num_trainable_parameters"]
    _save_json(os.path.join(output_dir, dataset, "aggregate_metrics.json"), aggregate)
    return summaries


def create_run_dir(output_root: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir
