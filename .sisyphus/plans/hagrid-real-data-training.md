# Local Training with Real HAGRID Data Plan

## TL;DR

> **Objective**: Create local Jupyter notebook that downloads 500 real thumbs-up images from HAGRID dataset via Kaggle API
> 
> **Deliverables**:
> - `train_local.ipynb` - Jupyter notebook
> - Automated HAGRID download (500 images, 'like' class only)
> - Kaggle API integration
> - Local GPU detection and training
> 
003e **Estimated Effort**: Medium (~1.5 hours)
> **Parallel Execution**: NO (sequential downloads + training)
> **Critical Path**: Download → Extract → Train

---

## Context

### Original Request
User wants to train with **real HAGRID dataset images** (not synthetic):
- Download from HAGRID dataset (Kaggle)
- 500 images of 'like' (thumbs up) class
- Jupyter notebook format for local execution
- GPU detection for local training

### HAGRID Dataset Details
- **Source**: https://www.kaggle.com/datasets/kapitanov/hagrid
- **Size**: ~40GB full dataset
- **Classes**: 18 gestures including 'like' (thumbs up)
- **Structure**: `user_id/gesture_class/image.jpg`
- **Annotations**: JSON format per user
- **License**: CC BY 4.0

### Approach
Instead of downloading 40GB, we'll:
1. Use Kaggle API to download dataset
2. Extract only 'like' class images
3. Sample 500 images (400 train + 100 val)
4. Convert annotations to YOLO format
5. Train locally

---

## Work Objectives

### Core Objective
Create `train_local.ipynb` that downloads 500 real thumbs-up images from HAGRID and trains YOLO locally.

### Concrete Deliverables
- `train_local.ipynb` with Kaggle download integration
- Automated download of HAGRID 'like' class subset
- 500 images split (400 train + 100 val)
- YOLO format label conversion from HAGRID JSON
- Local training with GPU detection

### Must Have
- Kaggle API setup and authentication
- Download only 'like' class images
- Exactly 500 images (or max available if < 500)
- Convert HAGRID JSON annotations to YOLO .txt format
- Person-stratified split (different users in train/val)
- Local directory structure
- GPU detection
- Training pipeline

### Must NOT Have
- No synthetic data generation
- No downloading full 40GB dataset
- No Colab-specific code
- No manual download steps

---

## Prerequisites

### Kaggle API Setup (User must do this first)
```bash
pip install kaggle
```

1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Save `kaggle.json` to `~/.kaggle/kaggle.json`
4. Run: `chmod 600 ~/.kaggle/kaggle.json`

---

## Execution Strategy

### Sequential Tasks (Dependencies)

```
Wave 1: Setup
└── Task 1: Create notebook with Kaggle download [quick]

Wave 2: Data Pipeline
└── Task 2: Add HAGRID extraction and conversion [medium]

Wave 3: Training
└── Task 3: Add local training pipeline [quick]
```

---

## TODOs

- [ ] 1. Create Notebook with Kaggle Download

  **What to do**:
  - Create `train_local.ipynb`
  - Cell 1: Check Kaggle API authentication
  - Cell 2: Download HAGRID dataset (`kaggle datasets download`)
  - Cell 3: Extract 'like' class images
  - Cell 4: Sample 500 images with person-stratified split
  - Cell 5: Convert HAGRID JSON annotations to YOLO format
  - Cell 6: Create dataset.yaml
  - Cell 7+: Training pipeline (same as before)

  **Must NOT do**:
  - No synthetic image generation
  - No downloading full dataset unnecessarily
  - No Colab-specific code

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **References**:
  - HAGRID Kaggle: https://www.kaggle.com/datasets/kapitanov/hagrid
  - HAGRID format: annotations in `annotations/{user_id}.json`
  - HAGRID structure: `images/{user_id}/{gesture}/{image}.jpg`

  **Acceptance Criteria**:
  - [ ] Notebook downloads HAGRID via Kaggle API
  - [ ] Extracts exactly 'like' class images
  - [ ] Gets 500 images (or explains if fewer available)
  - [ ] Converts JSON to YOLO format
  - [ ] Creates person-stratified train/val split
  - [ ] No manual download steps required

  **QA Scenarios**:
  ```
  Scenario: Kaggle API works
    Tool: Bash
    Steps:
      1. kaggle datasets list --user kapitanov
    Expected: Shows HAGRID dataset
    Evidence: .sisyphus/evidence/kaggle_api.txt
  
  Scenario: Download 'like' class
    Tool: Python (in notebook)
    Steps:
      1. Run download cell
      2. Check extracted images
    Expected: Images in ./data/hagrid/like/
    Evidence: ls -la ./data/hagrid/like/ | wc -l
  
  Scenario: YOLO format conversion
    Tool: Bash
    Steps:
      1. Check ./data/yolo/train/labels/
      2. Check format of .txt files
    Expected: YOLO format (class x_center y_center width height)
    Evidence: head -1 ./data/yolo/train/labels/*.txt
  
  Scenario: 500 images total
    Tool: Bash
    Steps:
      1. find ./data/yolo/train/images -name "*.jpg" | wc -l
      2. find ./data/yolo/val/images -name "*.jpg" | wc -l
    Expected: 400 + 100 = 500
    Evidence: .sisyphus/evidence/image_counts.txt
  ```

  **Commit**: YES
  - Message: `feat(data): add HAGRID download with 500 real thumbs-up images`
  - Files: `train_local.ipynb`

---

## HAGRID Data Processing Details

### HAGRID Structure
```
hagrid/
├── annotations/
│   ├── user_1234.json      # Annotations for user_1234
│   ├── user_5678.json
│   └── ...
└── images/
    ├── user_1234/
    │   ├── like/
    │   │   ├── image_001.jpg
    │   │   └── ...
    │   └── no_gesture/
    └── user_5678/
        └── ...
```

### JSON Format (per user)
```json
{
  "user_1234": {
    "labels": ["like", "like", "no_gesture"],
    "bboxes": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
  }
}
```

### YOLO Format (per image)
```
0 0.5 0.5 0.3 0.4
```
Where:
- `0` = class index (thumbs_up)
- `0.5 0.5` = center x, y (normalized)
- `0.3 0.4` = width, height (normalized)

### Conversion Logic
1. For each user, load their JSON annotation
2. Find indices where label == "like"
3. Get corresponding bounding boxes
4. Convert absolute bbox to YOLO normalized format:
   - x_center = (x1 + x2) / 2 / image_width
   - y_center = (y1 + y2) / 2 / image_height
   - width = (x2 - x1) / image_width
   - height = (y2 - y1) / image_height
5. Save as `{image_name}.txt`

---

## Success Criteria

### Verification Commands
```bash
# Check Kaggle auth
kaggle datasets list --user kapitanov | grep hagrid

# Count downloaded images
find ./data/hagrid/like -name "*.jpg" | wc -l

# Check YOLO labels created
find ./data/yolo/train/labels -name "*.txt" | wc -l
head -1 ./data/yolo/train/labels/*.txt

# Verify notebook runs
jupyter nbconvert --to notebook --execute train_local.ipynb
```

### Final Checklist
- [ ] Notebook downloads HAGRID automatically
- [ ] Extracts 'like' class (thumbs up)
- [ ] 500 images (400 train + 100 val)
- [ ] YOLO format labels created
- [ ] Person-stratified split
- [ ] Can start training immediately
