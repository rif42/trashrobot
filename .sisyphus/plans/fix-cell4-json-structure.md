# Fix Cell 4 - HAGRID JSON Structure

## Problem
The JSON structure in `ok.json` is different than expected:

**Actual Structure** (dict with image IDs as keys):
```json
{
  "000dfde3-a3a2-41b3-a3eb-e52744bf3ac4": {
    "bboxes": [[0.21, 0.27, 0.22, 0.20], [0.30, 0.75, 0.22, 0.23]],
    "labels": ["ok", "no_gesture"],
    "leading_hand": "right",
    "leading_conf": 1.0,
    "user_id": "..."
  },
  "001b114a-d181-44bd-af8c-a883f1cc482e": {
    ...
  }
}
```

**Key Findings**:
1. JSON is a dictionary (not a list)
2. Keys are image UUIDs (used as filenames)
3. Each image has `bboxes` (list) and `labels` (list)
4. **Bboxes are already normalized!** (values 0-1)
5. Images can have multiple gestures (need to filter for 'ok')

**Current Error**:
- `all_annotations` became a single value (5 items) instead of a list
- Slicing with `[:MAX_SAMPLES]` failed because it was a dict, not a list

## Solution

Update Cell 4 to:
1. Use `ann_data.items()` to get (image_id, image_data) pairs
2. Convert to list for proper slicing and shuffling
3. Filter bboxes where label == 'ok' (images can have multiple gestures)
4. Skip normalization (bboxes already normalized)
5. Handle images with no 'ok' gesture gracefully

## Fixed Cell 4 Code

```python
# Cell 4: Process Data - Filter 'ok' Class and Convert Annotations
import json
import shutil
from pathlib import Path
from tqdm.notebook import tqdm
from PIL import Image
import random

# Configuration
MAX_SAMPLES = 1000  # Maximum 'ok' samples to use
TARGET_CLASS = "ok"  # Only keep this class
TRAIN_SPLIT = 0.8   # 80% train, 20% test

# Paths - note the nested folder structure from Kaggle download
base_dir = Path("/content/hagrid_download/hagrid-sample-30k-384p")
ann_file = base_dir / 'ann_train_val' / f'{TARGET_CLASS}.json'
img_dir = base_dir / 'hagrid_30k' / f'train_val_{TARGET_CLASS}'
processed_dir = Path("/content/hagrid_processed")

print(f"🔍 Processing '{TARGET_CLASS}' class...")
print(f"📁 Annotation file: {ann_file}")
print(f"📁 Image directory: {img_dir}")

# Check paths exist
if not ann_file.exists():
    raise FileNotFoundError(f"❌ Annotation file not found: {ann_file}")
if not img_dir.exists():
    raise FileNotFoundError(f"❌ Image directory not found: {img_dir}")

# Create output directories
for split in ['train', 'test']:
    (processed_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    (processed_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

# Load annotation JSON
print(f"📖 Loading annotations...")
with open(ann_file, 'r') as f:
    ann_data = json.load(f)

# HAGRID JSON structure: {image_id: {bboxes: [[x,y,w,h], ...], labels: ["ok", ...], ...}}
# bboxes are already normalized (0-1 range)
all_images = list(ann_data.items())
print(f"📊 Total '{TARGET_CLASS}' images found: {len(all_images)}")
print(f"📊 Using first {min(MAX_SAMPLES, len(all_images))} samples")

# Take first MAX_SAMPLES
images_data = all_images[:MAX_SAMPLES]

# Shuffle and split into train/test
random.seed(42)
random.shuffle(images_data)
split_idx = int(len(images_data) * TRAIN_SPLIT)
train_images = images_data[:split_idx]
test_images = images_data[split_idx:]

print(f"📊 Train: {len(train_images)}, Test: {len(test_images)}")

# Process each split
for split_name, split_images in [('train', train_images), ('test', test_images)]:
    print(f"\n📂 Processing {split_name} split ({len(split_images)} samples)...")
    
    for img_id, img_data in tqdm(split_images, desc=f"Processing {split_name}"):
        # img_id is the filename (UUID)
        img_filename = img_id + '.jpg'
        
        # Find image path
        img_path = img_dir / img_filename
        
        if not img_path.exists():
            # Try other extensions
            for ext in ['.jpeg', '.png']:
                test_path = img_dir / (img_id + ext)
                if test_path.exists():
                    img_path = test_path
                    break
        
        if not img_path.exists():
            continue
        
        # Copy image
        dst_img_path = processed_dir / 'images' / split_name / img_path.name
        shutil.copy(img_path, dst_img_path)
        
        # Get bboxes and labels
        bboxes = img_data.get('bboxes', [])
        labels = img_data.get('labels', [])
        
        # Find 'ok' gesture bboxes
        ok_bboxes = []
        for i, label in enumerate(labels):
            if label == TARGET_CLASS and i < len(bboxes):
                ok_bboxes.append(bboxes[i])
        
        if not ok_bboxes:
            continue
        
        # Convert to YOLO format
        # Note: HAGRID bboxes are already normalized [0-1]
        # Format: [x_center, y_center, width, height]
        yolo_lines = []
        for bbox in ok_bboxes:
            x, y, w, h = bbox
            # Ensure values are in [0, 1] range
            x_center = max(0, min(1, x))
            y_center = max(0, min(1, y))
            w_norm = max(0, min(1, w))
            h_norm = max(0, min(1, h))
            
            # Class 0 for 'ok'
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # Save YOLO label file
        label_filename = img_id + '.txt'
        label_file = processed_dir / 'labels' / split_name / label_filename
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_lines))

print(f"\n✅ Processing complete!")

# Show final counts
for split in ['train', 'test']:
    img_count = len(list((processed_dir / 'images' / split).glob('*')))
    lbl_count = len(list((processed_dir / 'labels' / split).glob('*.txt')))
    print(f"   📁 {split}: {img_count} images, {lbl_count} labels")
```

## Key Changes from Current Code

1. **JSON Parsing**: Use `list(ann_data.items())` instead of `list(ann_data.values())`
   - Gets (image_id, image_data) tuples
   - image_id is the filename (UUID)

2. **Filename Handling**: Use `img_id + '.jpg'` directly
   - img_id is the dictionary key (UUID)

3. **Multi-gesture Support**: Images can have multiple bboxes
   - Filter bboxes where label == 'ok'
   - Save all 'ok' bboxes to the label file

4. **No Normalization**: Bboxes already normalized (0-1 range)
   - Just clip to [0, 1] for safety
   - No division by image dimensions needed

5. **Flexible Extension**: Try .jpg, .jpeg, .png

## Acceptance Criteria

- [ ] Cell 4 uses `ann_data.items()` to get image_id and data
- [ ] Extracts filename from dictionary key (img_id)
- [ ] Filters bboxes where label == 'ok'
- [ ] Handles multiple 'ok' bboxes per image
- [ ] No normalization (bboxes already 0-1)
- [ ] Shows correct image count (thousands, not 5)
- [ ] Successfully processes train and test splits

## Estimated Effort
Quick (< 15 minutes) - Replace one cell in notebook
