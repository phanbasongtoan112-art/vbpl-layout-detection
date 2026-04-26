#!/usr/bin/env python
"""
Train a YOLOv8 layout detector on the synthetic VBPL dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from ultralytics import YOLO
import yaml

CLASS_NAMES_DEFAULT = ["title", "section", "paragraph", "table", "list"]


def configure_logging(log_file: Path, verbose: bool) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_yolo")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def read_classes(dataset_dir: Path) -> List[str]:
    classes_path = dataset_dir / "classes.txt"
    if classes_path.exists():
        return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return list(CLASS_NAMES_DEFAULT)


def collect_dataset_images(dataset_dir: Path) -> List[Path]:
    images_dir = dataset_dir / "images"
    image_paths = sorted(
        path for path in images_dir.glob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    return image_paths


def ensure_splits(dataset_dir: Path, train_ratio: float, seed: int, force: bool) -> None:
    train_path = dataset_dir / "train.txt"
    val_path = dataset_dir / "val.txt"

    if train_path.exists() and val_path.exists() and not force:
        return

    import random

    image_paths = collect_dataset_images(dataset_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images were found in {dataset_dir / 'images'}")

    shuffled = list(image_paths)
    random.Random(seed).shuffle(shuffled)
    split_index = max(1, int(len(shuffled) * train_ratio)) if shuffled else 0
    if len(shuffled) >= 2:
        split_index = min(split_index, len(shuffled) - 1)

    train_images = shuffled[:split_index]
    val_images = shuffled[split_index:] if len(shuffled) > 1 else shuffled

    train_path.write_text("\n".join(path.resolve().as_posix() for path in train_images) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(path.resolve().as_posix() for path in val_images) + "\n", encoding="utf-8")


def ensure_dataset_yaml(dataset_dir: Path, class_names: List[str]) -> Path:
    dataset_yaml_path = dataset_dir / "dataset.yaml"
    dataset_yaml = {
        "path": dataset_dir.resolve().as_posix(),
        "train": "train.txt",
        "val": "val.txt",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    dataset_yaml_path.write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return dataset_yaml_path


def resolve_device(requested: Optional[str], logger: logging.Logger) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "0"
    logger.warning("CUDA is not available. Falling back to CPU. Training will be slower.")
    return "cpu"


def ensure_pretrained_model(models_dir: Path, requested_model: str, logger: logging.Logger) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    candidate = Path(requested_model)
    if candidate.exists():
        return candidate.resolve()

    if candidate.name == requested_model and (models_dir / candidate.name).exists():
        return (models_dir / candidate.name).resolve()

    logger.info("Loading pre-trained weights: %s", requested_model)
    model = YOLO(requested_model)
    checkpoint_path = getattr(model, "ckpt_path", None)
    if checkpoint_path:
        checkpoint = Path(checkpoint_path)
        if checkpoint.exists():
            target = models_dir / checkpoint.name
            if checkpoint.resolve() != target.resolve():
                shutil.copy2(checkpoint, target)
            return target.resolve()

    fallback = models_dir / Path(requested_model).name
    if fallback.exists():
        return fallback.resolve()

    return candidate


def read_last_metrics(results_csv_path: Path) -> dict:
    if not results_csv_path.exists():
        return {}
    with results_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def export_tensorboard(results_csv_path: Path, tensorboard_dir: Path, logger: logging.Logger) -> None:
    if not results_csv_path.exists():
        return
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        logger.warning("TensorBoard export skipped: %s", exc)
        return

    with results_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    for step, row in enumerate(rows):
        for key, value in row.items():
            if not value:
                continue
            try:
                writer.add_scalar(key.strip().replace(" ", "_"), float(value), step)
            except ValueError:
                continue
    writer.close()
    logger.info("Exported TensorBoard scalars to %s", tensorboard_dir)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on the synthetic VBPL layout dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=project_root / "data" / "synthetic_dataset",
        help="Directory containing images/, labels/, classes.txt, train.txt, val.txt",
    )
    parser.add_argument("--models-dir", type=Path, default=project_root / "models", help="Model output directory")
    parser.add_argument("--model", default="yolov8n.pt", help="Pre-trained YOLO checkpoint or model alias")
    parser.add_argument("--epochs", type=int, default=75, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--workers", type=int, default=4, help="Data loader worker count")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--device", default="", help="Torch device override, e.g. 0, 0,1 or cpu")
    parser.add_argument("--project", type=Path, default=project_root / "runs" / "detect", help="Ultralytics project directory")
    parser.add_argument("--run-name", default="vbpl_layout", help="Ultralytics run name")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio when regenerating splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split generation")
    parser.add_argument("--force-split", action="store_true", help="Rebuild train.txt and val.txt before training")
    parser.add_argument("--cache", default="disk", help="Ultralytics cache mode: False, ram, or disk")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose console logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = configure_logging(Path(__file__).resolve().parents[1] / "scripts" / "train_yolo.log", args.verbose)

    dataset_dir = args.dataset_dir
    models_dir = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    class_names = read_classes(dataset_dir)
    ensure_splits(dataset_dir, train_ratio=args.train_ratio, seed=args.seed, force=args.force_split)
    dataset_yaml_path = ensure_dataset_yaml(dataset_dir, class_names)
    device = resolve_device(args.device or None, logger)
    pretrained_model_path = ensure_pretrained_model(models_dir, args.model, logger)

    logger.info("Dataset YAML: %s", dataset_yaml_path)
    logger.info("Device: %s", device)
    logger.info("Pre-trained model: %s", pretrained_model_path)

    model = YOLO(str(pretrained_model_path))
    results = model.train(
        data=str(dataset_yaml_path),
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        workers=args.workers,
        device=device,
        patience=args.patience,
        project=str(args.project),
        name=args.run_name,
        exist_ok=True,
        plots=True,
        cache=args.cache,
        seed=args.seed,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
    )

    save_dir = Path(getattr(results, "save_dir", None) or getattr(model.trainer, "save_dir", ""))
    weights_dir = save_dir / "weights"
    best_source = weights_dir / "best.pt"
    best_target = models_dir / "best.pt"
    if best_source.exists():
        shutil.copy2(best_source, best_target)
        logger.info("Copied best weights to %s", best_target)

    results_csv_path = save_dir / "results.csv"
    metrics_target = models_dir / "training_metrics.csv"
    if results_csv_path.exists():
        shutil.copy2(results_csv_path, metrics_target)

    tensorboard_dir = models_dir / "tensorboard"
    export_tensorboard(results_csv_path, tensorboard_dir, logger)

    summary = {
        "dataset_yaml": dataset_yaml_path.resolve().as_posix(),
        "save_dir": save_dir.resolve().as_posix() if save_dir else "",
        "best_model": best_target.resolve().as_posix() if best_target.exists() else "",
        "device": device,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "metrics": read_last_metrics(results_csv_path),
    }
    summary_path = models_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Training summary written to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
