# Update train_yolo_local.ipynb for Raw Dataset

## Summary
Update the `train_yolo_local.ipynb` notebook to use the new dataset located in `data/raw/` instead of the old `data/processed/` structure.

## Background
- **Dataset Location**: `data/raw/hagrid_30k/`
  - `ok/` - 1,750 OK gesture images (.jpg)
  - `no_gesture/` - 27,823 negative samples (.jpeg)
- **Annotations**: `data/raw/ann_train_val/ok.json` - 27,999 entries
- **Format**: JSON contains bboxes in YOLO format `[x_center, y_center, width, height]`

## Current Issues
1. Cell 2 references `data/processed/` which doesn't exist with raw data structure
2. No JSON parsing - annotations are ignored
3. Labels need to be generated from JSON, not copied from existing files

## Changes Required

### Cell 2: Complete Rewrite
**File**: `train_yolo_local.ipynb` (cell index 1)

Replace the entire Cell 2 source code with:

```python
# Cell 2: Prepare Dataset from Raw Data
import os
import shutil
import random
import json
from pathlib import Path

print('\n📊 Preparing dataset from raw data...')
print('='*50)

# Set random seed for reproducibility
random.seed(42)

# Paths
raw_no_gesture_dir = Path('data/raw/hagrid_30k/no_gesture')
raw_ok_dir = Path('data/raw/hagrid_30k/ok')
annotations_file = Path('data/raw/ann_train_val/ok.json')
output_dir = Path('data/processed_from_raw')

# Load annotations
print('\n1️⃣ Loading annotations...')
with open(annotations_file, 'r') as f:
    annotations = json.load(f)
print(f'   Loaded {len(annotations)} annotation entries')

# Get list of actual ok images that exist
existing_ok_images = set(p.stem for p in raw_ok_dir.glob('*.jpg'))
print(f'   Found {len(existing_ok_images)} actual ok images')

# Filter annotations to only existing images
valid_annotations = {k: v for k, v in annotations.items() if k in existing_ok_images}
print(f'   Valid annotations: {len(valid_annotations)}')

# Split 80/20 train/val
print('\n2️⃣ Splitting data 80/20 train/val...')
ok_image_ids = list(valid_annotations.keys())
random.shuffle(ok_image_ids)
train_size = int(len(ok_image_ids) * 0.8)
train_ok_ids = ok_image_ids[:train_size]
val_ok_ids = ok_image_ids[train_size:]
print(f'   Train: {len(train_ok_ids)} ok images')
print(f'   Val: {len(val_ok_ids)} ok images')

# Sample no-gesture images
print('\n3️⃣ Sampling no-gesture images...')
all_no_gesture = list(raw_no_gesture_dir.glob('*.jpeg'))
print(f'   Found {len(all_no_gesture)} total no-gesture images')
sampled = random.sample(all_no_gesture, 5000)
no_gesture_train_size = int(len(sampled) * 0.8)
train_no_gesture = sampled[:no_gesture_train_size]
val_no_gesture = sampled[no_gesture_train_size:]
print(f'   Sampled 5,000 images (train: {len(train_no_gesture)}, val: {len(val_no_gesture)})')

# Create output directories
print('\n4️⃣ Creating output directories...')
for split in ['train', 'val']:
    (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

# Function to convert bboxes to YOLO format labels
def create_yolo_labels(image_id, annotation, label_dir):
    """Create YOLO format label file for ok bboxes only"""
    bboxes = annotation['bboxes']
    labels = annotation['labels']
    
    # Filter for ok labels only (class 0)
    ok_lines = []
    for bbox, label in zip(bboxes, labels):
        if label == 'ok':
            # bbox is [x_center, y_center, width, height] - already normalized
            x_center, y_center, width, height = bbox
            ok_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Write label file
    label_path = label_dir / f"{image_id}.txt"
    if ok_lines:
        with open(label_path, 'w') as f:
            f.writelines(ok_lines)
    else:
        # No ok bboxes - create empty file (background)
        label_path.touch()

def copy_ok_images_with_labels(ok_ids, annotations, split):
    """Copy ok images and create their labels"""
    img_dir = output_dir / 'images' / split
    label_dir = output_dir / 'labels' / split
    copied = 0
    
    for image_id in ok_ids:
        # Copy image
        src_img = raw_ok_dir / f"{image_id}.jpg"
        dst_img = img_dir / f"{image_id}.jpg"
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
            # Create label file
            create_yolo_labels(image_id, annotations[image_id], label_dir)
            copied += 1
    
    return copied

def copy_no_gesture_samples(no_gesture_list, split):
    """Copy no-gesture samples and create empty labels"""
    img_dir = output_dir / 'images' / split
    label_dir = output_dir / 'labels' / split
    copied = 0
    
    for img_path in no_gesture_list:
        # Convert .jpeg to .jpg for consistency
        dst_img = img_dir / f"{img_path.stem}.jpg"
        shutil.copy2(img_path, dst_img)
        # Create empty label file (background image)
        label_path = label_dir / f"{img_path.stem}.txt"
        label_path.touch()
        copied += 1
    
    return copied

# Copy ok images with labels
print('\n5️⃣ Copying ok images and generating labels...')
ok_train_copied = copy_ok_images_with_labels(train_ok_ids, valid_annotations, 'train')
ok_val_copied = copy_ok_images_with_labels(val_ok_ids, valid_annotations, 'val')
print(f'   Train: {ok_train_copied} images')
print(f'   Val: {ok_val_copied} images')

# Copy no-gesture samples
print('\n6️⃣ Copying no-gesture samples...')
no_train_copied = copy_no_gesture_samples(train_no_gesture, 'train')
no_val_copied = copy_no_gesture_samples(val_no_gesture, 'val')
print(f'   Train: {no_train_copied} images')
print(f'   Val: {no_val_copied} images')

# Create dataset.yaml
print('\n7️⃣ Creating dataset.yaml...')
yaml_path = output_dir / 'dataset.yaml'
yaml_content = f"""path: {output_dir.absolute()}
train: images/train
val: images/val
nc: 2
names: ['ok', 'no_gesture']
"""
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

# Print summary
print('\n' + '='*50)
print('✅ Dataset preparation complete!')
print('='*50)
total_train = ok_train_copied + no_train_copied
total_val = ok_val_copied + no_val_copied
print(f"Train: {ok_train_copied} ok + {no_train_copied} no_gesture = {total_train} total")
print(f"Val: {ok_val_copied} ok + {no_val_copied} no_gesture = {total_val} total")
print(f"\nDataset config: {yaml_path}")
print("\nClass mapping:")
print("  Class 0: ok")
print("  Class 1: no_gesture")
```

### Cell 3: Update Data Path
**File**: `train_yolo_local.ipynb` (cell index 2)

Change line:
```python
data='D:/work/trashrobot/data/processed_neg/dataset.yaml'
```

To:
```python
data='data/processed_from_raw/dataset.yaml'
```

### Cell 4: Update Results Path
**File**: `train_yolo_local.ipynb` (cell index 3)

Change line:
```python
train_dir = Path('runs/detect/runs/detect/train')
```

To:
```python
train_dir = Path('runs/detect/train')
```

### Cell 5: Update Test Inference Paths
**File**: `train_yolo_local.ipynb` (cell index 4)

Change lines:
```python
best_model = YOLO('runs/detect/runs/detect/train/weights/best.pt')
val_images = list(Path('data/processed/images/val').glob('*.jpg'))
```

To:
```python
best_model = YOLO('runs/detect/train/weights/best.pt')
val_images = list(Path('data/processed_from_raw/images/val').glob('*.jpg'))
```

### Cell 6: Update Export Path
**File**: `train_yolo_local.ipynb` (cell index 5)

Change line:
```python
best_model = YOLO('runs/detect/train/weights/best.pt')
```

To:
```python
best_model = YOLO('runs/detect/train/weights/best.pt')
```
(Note: This is actually the same, but good to verify)

## Expected Outcome
After execution:
- `data/processed_from_raw/images/train/` - ~5,400 images (ok + no_gesture)
- `data/processed_from_raw/images/val/` - ~1,350 images
- `data/processed_from_raw/labels/train/` - YOLO .txt files for ok images, empty for no_gesture
- `data/processed_from_raw/labels/val/` - Same structure
- `data/processed_from_raw/dataset.yaml` - YOLO dataset configuration

## Testing
After notebook runs:
1. Check generated label files exist: `ls data/processed_from_raw/labels/train/*.txt | head`
2. Verify YOLO format: First line should be `0 0.xxxxx 0.xxxxx 0.xxxxx 0.xxxxx`
3. Check dataset.yaml exists and has correct paths

## Dependencies
No new dependencies required. Uses existing: `os`, `shutil`, `random`, `json`, `pathlib`

## Notes
- The JSON annotations use normalized coordinates (0-1), which is already YOLO format
- Only "ok" labeled bboxes are extracted (class 0)
- No-gesture images get empty label files (background samples)
- Output directory is `data/processed_from_raw/` to avoid conflicts with existing `data/processed/`
