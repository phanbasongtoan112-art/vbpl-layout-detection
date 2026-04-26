#!/usr/bin/env python
"""
Run layout inference with the trained YOLO model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def configure_logging(log_file: Path, verbose: bool) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("predict_layout")
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


def resolve_device(requested: Optional[str], logger: logging.Logger) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "0"
    logger.warning("CUDA is not available. Falling back to CPU for inference.")
    return "cpu"


def collect_images(source: Path) -> List[Path]:
    if source.is_file():
        return [source]
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run YOLO layout inference on one image or a folder of images.")
    parser.add_argument("source", type=Path, help="Image file or directory of images")
    parser.add_argument("--model", type=Path, default=project_root / "models" / "best.pt", help="Trained model path")
    parser.add_argument("--output-dir", type=Path, default=project_root / "models" / "predictions", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="", help="Torch device override, e.g. 0 or cpu")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose console logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = configure_logging(Path(__file__).resolve().parents[1] / "scripts" / "predict_layout.log", args.verbose)

    if not args.model.exists():
        logger.error("Model file does not exist: %s", args.model)
        return 1

    source_images = collect_images(args.source)
    if not source_images:
        logger.error("No images found at %s", args.source)
        return 1

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device or None, logger)

    logger.info("Loading model from %s", args.model)
    model = YOLO(str(args.model))
    results = model.predict(
        source=[str(path) for path in source_images],
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        verbose=False,
        save=False,
        stream=False,
    )

    prediction_manifest: List[dict] = []

    for image_path, result in zip(source_images, results):
        annotated_array = result.plot()
        if annotated_array is None:
            annotated_image = Image.open(image_path).convert("RGB")
        else:
            rgb_array = annotated_array[..., ::-1] if annotated_array.shape[-1] == 3 else annotated_array
            annotated_image = Image.fromarray(np.asarray(rgb_array, dtype=np.uint8))

        annotated_path = output_dir / f"{image_path.stem}_pred{image_path.suffix}"
        annotated_image.save(annotated_path)

        boxes_payload: List[dict] = []
        txt_lines: List[str] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = [float(value) for value in box.xyxy.squeeze().tolist()]
                label = result.names.get(class_id, str(class_id))
                payload = {
                    "class_id": class_id,
                    "label": label,
                    "confidence": round(confidence, 6),
                    "xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                }
                boxes_payload.append(payload)
                txt_lines.append(
                    f"{label} conf={confidence:.4f} x1={x1:.2f} y1={y1:.2f} x2={x2:.2f} y2={y2:.2f}"
                )

        json_path = output_dir / f"{image_path.stem}_pred.json"
        txt_path = output_dir / f"{image_path.stem}_pred.txt"
        json_path.write_text(json.dumps(boxes_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        txt_path.write_text("\n".join(txt_lines) + ("\n" if txt_lines else ""), encoding="utf-8")

        prediction_manifest.append(
            {
                "source": image_path.resolve().as_posix(),
                "annotated_image": annotated_path.resolve().as_posix(),
                "boxes_json": json_path.resolve().as_posix(),
                "boxes_txt": txt_path.resolve().as_posix(),
                "detections": len(boxes_payload),
            }
        )

        logger.info("%s -> %d detection(s)", image_path.name, len(boxes_payload))

    manifest_path = output_dir / "predictions.json"
    manifest_path.write_text(json.dumps(prediction_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved annotated predictions to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
