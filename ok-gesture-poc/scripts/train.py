#!/usr/bin/env python3
"""
YOLO Training Script for OK Sign Detection
Trains YOLOv8 to detect persons and OK signs.
"""

import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime


def train_yolo(
    config_path="config/config.yaml",
    data_path="config/data.yaml",
    model_size="n",
    epochs=150,
    batch=16,
    imgsz=640,
):
    """
    Train YOLO model for OK sign detection.

    Args:
        config_path: Path to training config
        data_path: Path to data.yaml
        model_size: YOLO size ('n'=nano, 's'=small, 'm'=medium)
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Input image size
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed!")
        print("Install with: pip install ultralytics")
        return None

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load data config
    with open(data_path, "r") as f:
        data_config = yaml.safe_load(f)

    print("=" * 60)
    print("YOLO Training - OK Sign Detection")
    print("=" * 60)
    print(f"Model: YOLOv8{model_size}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}")
    print(f"Data: {data_path}")
    print("=" * 60)

    # Load pre-trained model
    model_name = f"yolov8{model_size}.pt"
    print(f"\nLoading pre-trained model: {model_name}")
    model = YOLO(model_name)

    # Training arguments
    train_args = {
        "data": data_path,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "patience": config["training"].get("patience", 20),
        "save": True,
        "device": config["training"].get("device", 0),
        "workers": config["training"].get("workers", 8),
        "project": "models/trained",
        "name": f"ok_sign_yolov8{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "exist_ok": False,
        "pretrained": True,
        "optimizer": config["training"].get("optimizer", "SGD"),
        "lr0": config["training"].get("lr0", 0.01),
        "lrf": config["training"].get("lrf", 0.01),
        "momentum": config["training"].get("momentum", 0.937),
        "weight_decay": config["training"].get("weight_decay", 0.0005),
        "augment": config["training"].get("augment", True),
        "mosaic": config["training"].get("mosaic", 1.0),
        "mixup": config["training"].get("mixup", 0.1),
        "copy_paste": config["training"].get("copy_paste", 0.1),
        "hsv_h": config["training"].get("hsv_h", 0.015),
        "hsv_s": config["training"].get("hsv_s", 0.7),
        "hsv_v": config["training"].get("hsv_v", 0.4),
        "degrees": config["training"].get("degrees", 0.0),
        "translate": config["training"].get("translate", 0.1),
        "scale": config["training"].get("scale", 0.5),
        "shear": config["training"].get("shear", 0.0),
        "perspective": config["training"].get("perspective", 0.0),
        "flipud": config["training"].get("flipud", 0.0),
        "fliplr": config["training"].get("fliplr", 0.5),
    }

    print("\nStarting training...")
    print("This may take 2-4 hours depending on your hardware.\n")

    # Train
    results = model.train(**train_args)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best model saved to: {results.best}")
    print(f"Results directory: {results.save_dir}")

    return results


def export_model(model_path, format="onnx", half=True):
    """
    Export trained model to deployment format.

    Args:
        model_path: Path to trained .pt model
        format: Export format ('onnx', 'openvino', 'engine')
        half: Use half precision (FP16)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed!")
        return None

    print(f"\nExporting model to {format.upper()}...")
    model = YOLO(model_path)

    export_path = model.export(format=format, half=half, imgsz=640)

    print(f"Exported model: {export_path}")
    return export_path


def evaluate_model(model_path, data_path="config/data.yaml"):
    """
    Evaluate trained model on test set.

    Args:
        model_path: Path to trained model
        data_path: Path to data.yaml
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed!")
        return None

    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    model = YOLO(model_path)

    # Validate on test set
    results = model.val(data=data_path, split="test")

    print("\nResults:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")

    # Per-class metrics
    if hasattr(results.box, "ap_class_index"):
        print("\nPer-class mAP50:")
        for i, class_name in enumerate(["person", "ok_sign"]):
            if i < len(results.box.ap50):
                print(f"  {class_name}: {results.box.ap50[i]:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO for OK sign detection")
    parser.add_argument(
        "--model",
        choices=["n", "s", "m"],
        default="n",
        help="YOLO model size (n=nano, s=small, m=medium)",
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of training epochs"
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument("--data", default="config/data.yaml", help="Path to data.yaml")
    parser.add_argument(
        "--export", action="store_true", help="Export best model after training"
    )
    parser.add_argument(
        "--export-format", default="onnx", help="Export format (onnx, openvino, engine)"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate model after training"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume training from checkpoint"
    )

    args = parser.parse_args()

    # Check if data exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data config not found: {data_path}")
        print("Collect and prepare data first!")
        return

    # Train
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        from ultralytics import YOLO

        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        results = train_yolo(
            config_path=args.config,
            data_path=args.data,
            model_size=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
        )

    if results is None:
        return

    best_model_path = results.best

    # Export
    if args.export:
        export_model(best_model_path, format=args.export_format)

    # Evaluate
    if args.evaluate:
        evaluate_model(best_model_path, data_path=args.data)

    print("\n" + "=" * 60)
    print("Training pipeline complete!")
    print(f"Best model: {best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
