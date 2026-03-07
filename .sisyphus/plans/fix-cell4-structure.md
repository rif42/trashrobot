# Fix Cell 4 - HAGRID Dataset Structure

## Issue
The current Cell 4 code expects:
```
hagrid_download/
├── ann_train_val/
│   ├── train/*.json   (one JSON per image)
│   └── test/*.json
└── hagrid_30k/
    ├── train/*.jpg
    └── test/*.jpg
```

But the actual structure is:
```
hagrid_download/
└── hagrid-sample-30k-384p/
    ├── ann_train_val/
    │   ├── ok.json          (one JSON per class, contains list of images)
    │   ├── call.json
    │   └── ... (16 other classes)
    └── hagrid_30k/
        ├── train_val_ok/    (folder with 'ok' images)
        ├── train_val_call/
        └── ... (16 other class folders)
```

## Fix Required

Update Cell 4 to:
1. Use correct base path: `/content/hagrid_download/hagrid-sample-30k-384p/`
2. Read `ann_train_val/ok.json` (contains list of image annotations)
3. Get images from `hagrid_30k/train_val_ok/`
4. Handle JSON format: list of dicts with `file_name`, `bbox`, etc.
5. Split into train/test (80/20) manually since original has no split

## Updated Cell 4 Code

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

# The JSON structure should have a list of image annotations
if isinstance(ann_data, list):
    all_annotations = ann_data
elif isinstance(ann_data, dict) and 'images' in ann_data:
    all_annotations = ann_data['images']
elif isinstance(ann_data, dict):
    all_annotations = list(ann_data.values())[0] if ann_data else []
else:
    raise ValueError(f"❌ Unexpected annotation format: {type(ann_data)}")

print(f"📊 Total '{TARGET_CLASS}' annotations found: {len(all_annotations)}")
print(f"📊 Using first {min(MAX_SAMPLES, len(all_annotations))} samples")

# Take first MAX_SAMPLES
annotations = all_annotations[:MAX_SAMPLES]

# Shuffle and split into train/test
random.seed(42)
random.shuffle(annotations)
split_idx = int(len(annotations) * TRAIN_SPLIT)
train_annotations = annotations[:split_idx]
test_annotations = annotations[split_idx:]

print(f"📊 Train: {len(train_annotations)}, Test: {len(test_annotations)}")

# Process each split
for split_name, split_annotations in [('train', train_annotations), ('test', test_annotations)]:
    print(f"\n📂 Processing {split_name} split ({len(split_annotations)} samples)...")
    
    for ann in tqdm(split_annotations, desc=f"Processing {split_name}"):
        # Get image filename - try different possible keys
        img_filename = None
        for key in ['file_name', 'image_id', 'filename', 'name', 'id']:
            if key in ann:
                img_filename = ann[key]
                if not img_filename.endswith(('.jpg', '.jpeg', '.png')):
                    img_filename += '.jpg'
                break
        
        if not img_filename:
            continue
        
        # Find image path
        img_path = img_dir / img_filename
        
        if not img_path.exists():
            # Try without extension variations
            base_name = img_filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
            for ext in ['.jpg', '.jpeg', '.png']:
                test_path = img_dir / (base_name + ext)
                if test_path.exists():
                    img_path = test_path
                    break
        
        if not img_path.exists():
            continue
        
        # Copy image
        dst_img_path = processed_dir / 'images' / split_name / img_path.name
        shutil.copy(img_path, dst_img_path)
        
        # Get image dimensions
        with Image.open(img_path) as img:
            img_width, img_height = img.size
        
        # Get bounding box
        bbox = None
        for key in ['bbox', 'box', 'bounding_box', 'coordinates']:
            if key in ann:
                bbox = ann[key]
                break
        
        if not bbox:
            continue
        
        # Convert COCO [x, y, w, h] to YOLO format
        x, y, w, h = bbox
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        
        # Clip to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w_norm = max(0, min(1, w_norm))
        h_norm = max(0, min(1, h_norm))
        
        # Class 0 for 'ok'
        yolo_line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        
        # Save YOLO label file
        label_filename = img_path.stem + '.txt'
        label_file = processed_dir / 'labels' / split_name / label_filename
        with open(label_file, 'w') as f:
            f.write(yolo_line)

print(f"\n✅ Processing complete!")

# Show final counts
for split in ['train', 'test']:
    img_count = len(list((processed_dir / 'images' / split).glob('*')))
    lbl_count = len(list((processed_dir / 'labels' / split).glob('*.txt')))
    print(f"   📁 {split}: {img_count} images, {lbl_count} labels")
```

## Key Changes

1. **Base path**: Now includes `hagrid-sample-30k-384p/` subdirectory
2. **Annotation source**: Single `ok.json` file with list of annotations
3. **Image source**: `train_val_ok/` folder with all 'ok' images
4. **Train/test split**: Manually created (80/20) with random seed for reproducibility
5. **Flexible field names**: Tries multiple keys for filename and bbox extraction

## Acceptance Criteria

- [ ] Cell 4 uses correct nested path structure
- [ ] Reads `ok.json` and extracts list of annotations
- [ ] Copies images from `train_val_ok/` folder
- [ ] Creates 80/20 train/test split
- [ ] Converts COCO bbox format to YOLO format correctly
- [ ] Handles missing images gracefully
