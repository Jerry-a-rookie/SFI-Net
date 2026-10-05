"""Train and evaluate SFI-Net on APAVA and/or ADFTD."""

import argparse
import json
import os
import sys

import torch

from experiments.ours_trainer import create_run_dir, run_dataset


DEFAULT_DATA_ROOTS = {
    "APAVA": "./datasets/APAVA",
    "ADFTD": "./datasets/ADFTD",
}
DEFAULT_OUTPUT_ROOT = "./outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate SFI-Net")
    parser.add_argument("--dataset", choices=["APAVA", "ADFTD", "all"], default="APAVA")
    parser.add_argument("--apava-root", default=DEFAULT_DATA_ROOTS["APAVA"])
    parser.add_argument("--adftd-root", default=DEFAULT_DATA_ROOTS["ADFTD"])
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-dir", default=None, help="Reuse an existing timestamped run directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=None, help="Override dataset default max epochs")
    parser.add_argument("--patience", type=int, default=None, help="Override dataset default early-stopping patience")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patch-len", type=int, default=32)
    parser.add_argument("--patch-stride", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--self-loop", type=float, default=1.0)
    parser.add_argument("--soft-temperature", type=float, default=4.0)
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU only if CUDA is unavailable")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        if not args.allow_cpu:
            raise RuntimeError("CUDA is unavailable. Run with the DL environment and RTX 4060, or pass --allow-cpu explicitly.")
        device = torch.device("cpu")
    else:
        if args.gpu >= torch.cuda.device_count():
            raise ValueError(f"GPU index {args.gpu} unavailable; visible devices={torch.cuda.device_count()}")
        device = torch.device(f"cuda:{args.gpu}")
    if device.type == "cuda":
        print(f"Using {torch.cuda.get_device_name(device)} via {device}; torch={torch.__version__}, CUDA={torch.version.cuda}")

    run_dir = args.run_dir or create_run_dir(args.output_root)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    datasets = ["APAVA", "ADFTD"] if args.dataset == "all" else [args.dataset]
    roots = {"APAVA": args.apava_root, "ADFTD": args.adftd_root}
    for dataset in datasets:
        run_dataset(dataset, roots[dataset], run_dir, args, device)
    print(f"Completed. Results: {run_dir}")


if __name__ == "__main__":
    main()
