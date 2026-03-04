# OK Sign Gesture Detection - Phase 2: Data Collection

## Overview
This phase involves collecting and preparing training data for the OK sign gesture detection model.

## Directory Structure

```
data/
├── raw/                    # Raw collected images
│   ├── 1m_2m/             # Close distance images
│   ├── 3m_4m/             # Medium distance images
│   ├── 5m_7m/             # Far distance images
│   └── various_distances/ # Mixed distance images
├── processed/             # Organized dataset
│   ├── train/
│   │   ├── images/        # Training images
│   │   └── labels/        # YOLO format labels (.txt)
│   ├── val/
│   │   ├── images/        # Validation images
│   │   └── labels/        # Validation labels
│   └── test/
│       ├── images/        # Test images
│       └── labels/        # Test labels
└── annotations/           # Annotation project files
```

## Data Collection

### Option 1: Multi-Distance Collection (Recommended)

Collect data at multiple distances in one session:

```bash
cd ok-gesture-poc
python scripts/collect_data.py --mode multi --duration 30
```

This will guide you through:
1. **Close (1-2m)**: 30 seconds of OK signs at close range
2. **Medium (3-4m)**: 30 seconds at medium distance  
3. **Far (5-7m)**: 30 seconds at far distance

### Option 2: Single Distance Collection

Collect data at a specific distance:

```bash
# Close distance (1-2m)
python scripts/collect_data.py --distance close --duration 30

# Medium distance (3-4m)
python scripts/collect_data.py --distance medium --duration 30

# Far distance (5-7m)
python scripts/collect_data.py --distance far --duration 30
```

### Collection Tips

- **Hand positions**: Vary hand height, angle, and orientation
- **Lighting**: Collect in different lighting conditions if possible
- **Background**: Try different backgrounds (plain wall, room, etc.)
- **Multiple people**: If possible, have different people make OK signs
- **Duration**: 30 seconds per distance = ~60 images (1 per 0.5s)

## Dataset Preparation

### 1. Organize Data

After collection, organize into train/val/test splits:

```bash
python scripts/prepare_dataset.py
```

This creates:
- 80% train
- 10% validation
- 10% test

### 2. Annotate Images

You need to annotate the collected images with bounding boxes.

**Recommended Tools:**
- **Label Studio**: Web-based, supports YOLO export
  ```bash
  pip install label-studio
  label-studio start
  ```
  
- **LabelImg**: Desktop application
  ```bash
  pip install labelImg
  labelImg
  ```

**Annotation Guidelines:**
1. Draw bounding box around entire person (class 0: person)
2. Draw tight bounding box around hand making OK sign (class 1: ok_sign)
3. Label format: YOLO (class_id, x_center, y_center, width, height)
4. All values normalized to [0, 1]

### 3. Verify Dataset

Check dataset structure:

```bash
python scripts/prepare_dataset.py --verify
```

## Target Data Requirements

| Distance | Minimum Images | Hand Size in Frame |
|----------|---------------|-------------------|
| 1-2m | 100+ | Large (100-300px) |
| 3-4m | 100+ | Medium (50-100px) |
| 5-7m | 100+ | Small (20-50px) |
| **Total** | **300+** | Varies |

## Public Datasets (Optional)

If collecting your own data is insufficient, consider:

- **HaGRID**: Hand Gesture Recognition Image Dataset
  - Contains various hand gestures including OK sign
  - Available on GitHub: hukenovs/hagrid
  
- **HRI30**: Hand gesture dataset for human-robot interaction

- **Custom**: Download images from web and annotate

## Next Steps

1. Collect data using `collect_data.py`
2. Annotate images using Label Studio or LabelImg
3. Verify dataset structure
4. Proceed to Phase 3: Model Training

## Troubleshooting

### Camera not accessible
- Check if another application is using the camera
- Try different camera index: `--source 1` or `--source 2`
- Check webcam permissions

### Low frame rate during collection
- Reduce camera resolution in script
- Close other applications
- Use `--interval 1.0` to save less frequently

### Images too dark/bright
- Adjust room lighting
- Move closer to window or light source
- Check camera auto-exposure settings

## File Formats

### YOLO Label Format
Each image has a corresponding `.txt` file with same name:

```
0 0.5 0.5 0.3 0.6    # person at center, 30% width, 60% height
1 0.7 0.4 0.05 0.08  # ok_sign at (0.7, 0.4), 5% width, 8% height
```

Format: `class_id x_center y_center width height`
- All values normalized to [0.0, 1.0]
- Coordinates relative to image size
- One line per object
