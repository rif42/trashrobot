#!/usr/bin/env python3
"""
Dataset Preparation Script
Organizes collected data into train/val/test splits for YOLO training.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict


def organize_dataset(
    raw_dir="data/raw",
    output_dir="data/processed",
    split_ratio=(0.8, 0.1, 0.1),
    seed=42,
):
    """
    Organize collected images into train/val/test splits.

    Args:
        raw_dir: Directory with collected images
        output_dir: Output directory for organized dataset
        split_ratio: (train, val, test) proportions
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    raw_path = Path(raw_dir)
    output_path = Path(output_dir)

    if not raw_path.exists():
        print(f"Error: Raw directory not found: {raw_path}")
        return

    # Create output structure
    splits = ["train", "val", "test"]
    for split in splits:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "labels").mkdir(parents=True, exist_ok=True)

    # Collect all images
    all_images = []
    for img_path in raw_path.rglob("*.jpg"):
        all_images.append(img_path)
    for img_path in raw_path.rglob("*.png"):
        all_images.append(img_path)

    if len(all_images) == 0:
        print(f"Warning: No images found in {raw_path}")
        print("Collect data first using: python scripts/collect_data.py")
        return

    print(f"Found {len(all_images)} images")

    # Shuffle and split
    random.shuffle(all_images)
    n_total = len(all_images)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])

    train_images = all_images[:n_train]
    val_images = all_images[n_train : n_train + n_val]
    test_images = all_images[n_train + n_val :]

    splits_dict = {"train": train_images, "val": val_images, "test": test_images}

    # Copy images
    for split_name, images in splits_dict.items():
        print(f"\nProcessing {split_name}: {len(images)} images")
        for img_path in images:
            # Copy image
            dest_img = output_path / split_name / "images" / img_path.name
            shutil.copy2(img_path, dest_img)

            # Create empty label file (will be filled during annotation)
            label_name = img_path.stem + ".txt"
            dest_label = output_path / split_name / "labels" / label_name
            dest_label.touch()

    print(f"\n{'=' * 50}")
    print("Dataset organization complete!")
    print(f"Train: {len(train_images)} images")
    print(f"Val: {len(val_images)} images")
    print(f"Test: {len(test_images)} images")
    print(f"Total: {n_total} images")
    print(f"\nNext steps:")
    print("1. Annotate images using label-studio or labelImg")
    print("2. Place .txt label files in data/processed/{split}/labels/")
    print("3. Update config/data.yaml if needed")
    print(f"{'=' * 50}\n")


def create_data_yaml(output_dir="data/processed", config_path="config/data.yaml"):
    """Create or update data.yaml configuration."""
    yaml_content = f"""# YOLO Dataset Configuration
path: {output_dir}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')
test: test/images  # test images (optional)

# Classes
names:
  0: person
  1: ok_sign

# Number of classes
nc: 2
"""

    with open(config_path, "w") as f:
        f.write(yaml_content)

    print(f"Created {config_path}")


def verify_dataset(data_dir="data/processed"):
    """Verify dataset structure and report statistics."""
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Dataset directory not found: {data_path}")
        return

    print(f"\n{'=' * 50}")
    print("Dataset Verification")
    print(f"{'=' * 50}")

    for split in ["train", "val", "test"]:
        split_path = data_path / split
        if not split_path.exists():
            print(f"{split}: Not found")
            continue

        images = list((split_path / "images").glob("*.jpg"))
        images += list((split_path / "images").glob("*.png"))
        labels = list((split_path / "labels").glob("*.txt"))

        # Check for matched pairs
        img_names = {i.stem for i in images}
        label_names = {l.stem for l in labels}

        matched = len(img_names & label_names)
        img_only = len(img_names - label_names)
        label_only = len(label_names - img_names)

        print(f"\n{split.upper()}:")
        print(f"  Images: {len(images)}")
        print(f"  Labels: {len(labels)}")
        print(f"  Matched pairs: {matched}")
        if img_only > 0:
            print(f"  Images without labels: {img_only}")
        if label_only > 0:
            print(f"  Labels without images: {label_only}")

    print(f"\n{'=' * 50}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument(
        "--organize",
        action="store_true",
        help="Organize raw data into train/val/test splits",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify dataset structure"
    )
    parser.add_argument(
        "--create-config", action="store_true", help="Create data.yaml configuration"
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    parser.add_argument(
        "--output-dir", default="data/processed", help="Output directory"
    )

    args = parser.parse_args()

    if args.organize:
        organize_dataset(args.raw_dir, args.output_dir)

    if args.verify:
        verify_dataset(args.output_dir)

    if args.create_config:
        create_data_yaml(args.output_dir)

    # If no flags, run all
    if not any([args.organize, args.verify, args.create_config]):
        print("Running all preparation steps...")
        organize_dataset(args.raw_dir, args.output_dir)
        create_data_yaml(args.output_dir)
        verify_dataset(args.output_dir)
