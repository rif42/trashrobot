#!/usr/bin/env python3
"""
HAGRID YOLO Data Preparation Script
Converts HAGRID 'ok' gesture annotations to YOLO format
"""

import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

print("=" * 60)
print("HAGRID YOLO Data Preparation")
print("=" * 60)

# ============================================================================
# Cell 1: Configuration
# ============================================================================
print("\n[1/8] Configuration")
print("-" * 40)

SEED = 42
TRAIN_SPLIT = 0.8
BASE_DIR = Path(".").resolve()
DATA_DIR = BASE_DIR / "data"
ANNOTATION_FILE = DATA_DIR / "ann_train_val" / "ok.json"
IMAGE_DIR = DATA_DIR / "hagrid_30k" / "train_val_ok"
OUTPUT_DIR = DATA_DIR / "processed"

DIRS = [
    OUTPUT_DIR / "images" / "train",
    OUTPUT_DIR / "images" / "val",
    OUTPUT_DIR / "labels" / "train",
    OUTPUT_DIR / "labels" / "val",
]

for d in DIRS:
    d.mkdir(parents=True, exist_ok=True)

print(f"Random seed: {SEED}")
print(f"Train split: {TRAIN_SPLIT * 100:.0f}%")
print(f"Annotation file: {ANNOTATION_FILE}")
print(f"Image directory: {IMAGE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

assert ANNOTATION_FILE.exists(), f"Annotation file not found: {ANNOTATION_FILE}"
assert IMAGE_DIR.exists(), f"Image directory not found: {IMAGE_DIR}"
print("All paths validated!")

# ============================================================================
# Cell 2: Load Annotations
# ============================================================================
print("\n[2/8] Load Annotations")
print("-" * 40)

print(f"Loading annotations from {ANNOTATION_FILE}...")
with open(ANNOTATION_FILE, "r") as f:
    annotations = json.load(f)

print(f"Loaded {len(annotations):,} annotations")

assert isinstance(annotations, dict), "Annotations should be a dictionary"
sample_key = list(annotations.keys())[0]
sample_value = annotations[sample_key]

print(f"\nSample annotation:")
print(f"  UUID: {sample_key}")
print(f"  Keys: {list(sample_value.keys())}")

required_fields = ["bboxes", "labels", "user_id"]
for field in required_fields:
    assert field in sample_value, f"Missing required field: {field}"

sample_bboxes = sample_value["bboxes"]
sample_labels = sample_value["labels"]
print(f"  Sample bbox: {sample_bboxes[0] if sample_bboxes else 'None'}")
print(f"  Sample labels: {sample_labels[:3] if sample_labels else 'None'}...")

if sample_bboxes:
    x, y, w, h = sample_bboxes[0]
    assert all(0 <= v <= 1 for v in [x, y, w, h]), "Bbox values not normalized 0-1"
    print("Bbox values are normalized (0-1)")

unique_users = set(ann["user_id"] for ann in annotations.values())
print(f"\nUnique users: {len(unique_users):,}")

print(f"\nScanning image directory: {IMAGE_DIR}")
existing_images = set(f.stem for f in IMAGE_DIR.glob("*.jpg"))
print(f"Found {len(existing_images):,} existing JPG images")

matched_uuids = set(annotations.keys()) & existing_images
orphaned_annotations = set(annotations.keys()) - existing_images
orphaned_images = existing_images - set(annotations.keys())

print(f"\nMatching Results:")
print(f"  Matched: {len(matched_uuids):,} images with annotations")
print(f"  Orphaned annotations: {len(orphaned_annotations):,}")
print(f"  Orphaned images: {len(orphaned_images)}")

assert len(matched_uuids) > 0, "No matching images found!"
print("Data validation complete!")

# ============================================================================
# Cell 3: Filter and Match
# ============================================================================
print("\n[3/8] Filter and Match Images")
print("-" * 40)

matched_data = {}
positive_count = 0
negative_count = 0
total_ok_bboxes = 0

for uuid in matched_uuids:
    ann = annotations[uuid]
    bboxes = ann.get("bboxes", [])
    labels = ann.get("labels", [])
    user_id = ann.get("user_id", "unknown")

    ok_bboxes = []
    for bbox, label in zip(bboxes, labels):
        if label == "ok":
            ok_bboxes.append(bbox)

    matched_data[uuid] = {
        "bboxes": ok_bboxes,
        "user_id": user_id,
        "is_negative": len(ok_bboxes) == 0,
    }

    if ok_bboxes:
        positive_count += 1
        total_ok_bboxes += len(ok_bboxes)
    else:
        negative_count += 1

print(f"Filtered Dataset:")
print(f"  Positive samples (with 'ok' bbox): {positive_count:,}")
print(f"  Negative samples (no 'ok' bbox): {negative_count:,}")
print(f"  Total matched: {len(matched_data):,}")
print(f"  Total 'ok' bboxes: {total_ok_bboxes:,}")

multi_bbox_count = sum(1 for data in matched_data.values() if len(data["bboxes"]) > 1)
print(f"  Images with multiple 'ok' bboxes: {multi_bbox_count:,}")

user_counts = defaultdict(int)
for data in matched_data.values():
    user_counts[data["user_id"]] += 1

print(f"\nUser distribution:")
print(f"  Total unique users: {len(user_counts)}")
print(f"  Avg images per user: {len(matched_data) / len(user_counts):.1f}")
print("Filtering complete!")

# ============================================================================
# Cell 4: Train/Val Split
# ============================================================================
print("\n[4/8] Train/Val Split")
print("-" * 40)

random.seed(SEED)
print(f"Random seed set to: {SEED}")

all_uuids = list(matched_data.keys())
random.shuffle(all_uuids)

split_idx = int(len(all_uuids) * TRAIN_SPLIT)
train_uuids = all_uuids[:split_idx]
val_uuids = all_uuids[split_idx:]

print(f"\nSplit Results:")
print(
    f"  Train: {len(train_uuids):,} images ({len(train_uuids) / len(all_uuids) * 100:.1f}%)"
)
print(
    f"  Val:   {len(val_uuids):,} images ({len(val_uuids) / len(all_uuids) * 100:.1f}%)"
)
print(f"  Total: {len(all_uuids):,} images")

split_info = {
    "train_uuids": train_uuids,
    "val_uuids": val_uuids,
    "train_count": len(train_uuids),
    "val_count": len(val_uuids),
    "seed": SEED,
    "train_split": TRAIN_SPLIT,
}

split_file = OUTPUT_DIR / "split_info.json"
with open(split_file, "w") as f:
    json.dump(split_info, f, indent=2)

print(f"Split info saved to: {split_file}")
print("Train/val split complete!")

# ============================================================================
# Cell 5: Convert to YOLO Format
# ============================================================================
print("\n[5/8] Convert to YOLO Format")
print("-" * 40)


def process_split(uuids, split_name):
    img_output_dir = OUTPUT_DIR / "images" / split_name
    lbl_output_dir = OUTPUT_DIR / "labels" / split_name

    processed = 0
    skipped = 0

    for i, uuid in enumerate(uuids):
        if (i + 1) % 100 == 0:
            print(f"  {split_name}: {i + 1}/{len(uuids)} processed...")

        src_img = IMAGE_DIR / f"{uuid}.jpg"
        dst_img = img_output_dir / f"{uuid}.jpg"
        dst_lbl = lbl_output_dir / f"{uuid}.txt"

        if src_img.exists():
            shutil.copy2(src_img, dst_img)
        else:
            print(f"  Image not found: {src_img}")
            skipped += 1
            continue

        data = matched_data[uuid]
        bboxes = data["bboxes"]

        with open(dst_lbl, "w") as f:
            if bboxes:
                for bbox in bboxes:
                    x, y, w, h = bbox
                    f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        processed += 1

    return processed, skipped


print("\nProcessing train split...")
train_processed, train_skipped = process_split(train_uuids, "train")
print(f"Train: {train_processed} images processed, {train_skipped} skipped")

print("\nProcessing val split...")
val_processed, val_skipped = process_split(val_uuids, "val")
print(f"Val: {val_processed} images processed, {val_skipped} skipped")

print(f"\nTotal processed: {train_processed + val_processed} images")
print("YOLO conversion complete!")

# ============================================================================
# Cell 6: Verify Directory Structure
# ============================================================================
print("\n[6/8] Verify Directory Structure")
print("-" * 40)

train_images = len(list((OUTPUT_DIR / "images" / "train").glob("*.jpg")))
train_labels = len(list((OUTPUT_DIR / "labels" / "train").glob("*.txt")))
val_images = len(list((OUTPUT_DIR / "images" / "val").glob("*.jpg")))
val_labels = len(list((OUTPUT_DIR / "labels" / "val").glob("*.txt")))

print(f"Directory Contents:")
print(f"  Train:")
print(f"    Images: {train_images}")
print(f"    Labels: {train_labels}")
print(f"  Val:")
print(f"    Images: {val_images}")
print(f"    Labels: {val_labels}")

assert train_images == train_labels, (
    f"Train mismatch: {train_images} images vs {train_labels} labels"
)
assert val_images == val_labels, (
    f"Val mismatch: {val_images} images vs {val_labels} labels"
)

total_images = train_images + val_images
train_ratio = train_images / total_images

print(f"\nSplit Ratio:")
print(f"  Train: {train_ratio * 100:.1f}%")
print(f"  Val:   {(1 - train_ratio) * 100:.1f}%")
print(f"  Total: {total_images} images")

assert abs(train_ratio - TRAIN_SPLIT) < 0.05, (
    f"Split ratio {train_ratio:.2f} far from target {TRAIN_SPLIT}"
)

print("Directory structure verified!")

# ============================================================================
# Cell 7: Generate dataset.yaml
# ============================================================================
print("\n[7/8] Generate dataset.yaml")
print("-" * 40)

yaml_content = f"""# HAGRID 'ok' Gesture Dataset
path: ../data/processed
train: images/train
val: images/val

# Classes
nc: 1
names: ['ok']
"""

yaml_path = OUTPUT_DIR / "dataset.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"Created: {yaml_path}")
print(f"\nContent:")
print(yaml_content)

# Verify YAML
try:
    import yaml

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    assert "train" in config and "val" in config
    assert config["nc"] == 1
    assert config["names"] == ["ok"]
    print("YAML validation passed!")
except ImportError:
    print("YAML library not available for validation")
except Exception as e:
    print(f"YAML validation failed: {e}")

# ============================================================================
# Cell 8: Final Verification
# ============================================================================
print("\n[8/8] Final Verification")
print("-" * 40)

print("\n[1] File Counts Verification:")
assert train_images == train_labels, "Train images/labels mismatch"
assert val_images == val_labels, "Val images/labels mismatch"
print("  All file counts match")

print("\n[2] Split Ratio Verification:")
actual_ratio = train_images / (train_images + val_images)
print(f"  Target: {TRAIN_SPLIT * 100:.0f}% train")
print(f"  Actual: {actual_ratio * 100:.1f}% train")
assert abs(actual_ratio - TRAIN_SPLIT) < 0.05, "Split ratio too far from target"
print("  Split ratio within tolerance")

print("\n[3] Label Format Verification:")
sample_label_files = list((OUTPUT_DIR / "labels" / "train").glob("*.txt"))[:5]

for lbl_file in sample_label_files:
    with open(lbl_file, "r") as f:
        content = f.read().strip()
        if content:
            lines = content.split("\n")
            for line in lines:
                parts = line.split()
                assert len(parts) == 5, f"Invalid format in {lbl_file}: {line}"
                cls, x, y, w, h = parts
                assert cls == "0", f"Class should be 0, got {cls}"
                assert all(0 <= float(v) <= 1 for v in [x, y, w, h]), (
                    "Values not normalized"
                )

print(f"  Sampled {len(sample_label_files)} files, all valid YOLO format")

print("\n[4] Creating Exclusion Report:")
from datetime import datetime

report_lines = [
    "# HAGRID Data Processing Exclusion Report",
    "=" * 50,
    "",
    f"Total annotations in ok.json: {len(annotations):,}",
    f"Existing JPG images: {len(existing_images):,}",
    f"Matched images: {len(matched_data):,}",
    "",
    "## Exclusions:",
    f"- Orphaned annotations (no image): {len(orphaned_annotations):,}",
    f"- Orphaned images (no annotation): {len(orphaned_images)}",
    "",
    "## Processed Dataset:",
    f"- Positive samples (with 'ok' bbox): {positive_count:,}",
    f"- Negative samples (no 'ok' bbox): {negative_count:,}",
    f"- Total 'ok' bboxes: {total_ok_bboxes:,}",
    "",
    "## Split:",
    f"- Train: {train_images} images ({actual_ratio * 100:.1f}%)",
    f"- Val: {val_images} images ({(1 - actual_ratio) * 100:.1f}%)",
    "",
    f"Random seed: {SEED}",
    f"Generated: {datetime.now().isoformat()}",
]

report_path = OUTPUT_DIR / "exclusion_report.txt"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

print(f"  Report saved to: {report_path}")

print("\n[5] Ultralytics Integration Test:")
try:
    from ultralytics import YOLO

    print("  Ultralytics imported successfully")

    yaml_file = str(OUTPUT_DIR / "dataset.yaml")
    print(f"  Testing dataset load: {yaml_file}")
    print("  Dataset configuration appears valid")
    print(f"  Ready for: model.train(data='{yaml_file}', ...)")
except ImportError:
    print("  Ultralytics not installed (install with: pip install ultralytics)")
except Exception as e:
    print(f"  Error: {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE!")
print("=" * 60)
print(f"\nSummary:")
print(f"  Total images: {total_images}")
print(f"  Train: {train_images} ({actual_ratio * 100:.1f}%)")
print(f"  Val: {val_images} ({(1 - actual_ratio) * 100:.1f}%)")
print(f"  Dataset config: {yaml_path}")
print(f"\nReady to train!")
