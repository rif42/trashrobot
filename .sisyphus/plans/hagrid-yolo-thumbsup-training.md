# HaGRID Thumbs Up Detection - YOLO26n Training Plan

## TL;DR

> **Objective**: Train YOLO26n to detect "thumbs up" gestures from HaGRID dataset, optimized for small/distant hand detection
> 
> **Deliverables**:
> - Trained model weights (`best.pt`) in YOLOv8 format
> - Dataset configuration (`dataset.yaml`) with person-stratified split
> - Colab training notebook with reproducible execution
> - Inference script with SAHI support for high-res images
> - Evaluation report with size-stratified metrics
> 
> **Estimated Effort**: Medium (2-3 hours on Colab T4)
> **Parallel Execution**: NO - Sequential pipeline with dependencies
> **Critical Path**: Data Download → Person-Stratified Split → Training → Evaluation

---

## Context

### Original Request
User wants to train YOLOv8n to detect thumbs up gesture from HaGRID dataset, specifically for detecting small hand signs at distance.

### Interview Summary
**Key Discussions**:
- **Target use case**: Detect small hand signs at distance (hand fills <20% of frame)
- **Gesture scope**: Single class - "thumbs up" only (HaGRID "like" class)
- **Environment**: Google Colab with free T4 GPU
- **Model choice**: YOLO26n (latest YOLO26 with ProgLoss+STAL for small object detection, NMS-free)
- **Training resolution**: 640×640 (imgsz=640)
- **Inference**: Various resolutions, potentially high-res with small hands

**Research Findings**:
- HaGRID dataset: 554K images across 18 gesture classes
- "like" class = thumbs up gesture (~30K images)
- YOLO26n includes ProgLoss + STAL loss functions specifically for small object detection (no P2 head needed)
- HaGRID includes `user_id` field - MUST use for stratified split to prevent data leakage
- Including "no_gesture" as negative class reduces false positives

### Metis Review
**Identified Gaps** (addressed in this plan):
- ✅ **Data leakage risk**: Person-stratified split implemented (not random)
- ✅ **Negative class strategy**: Include "no_gesture" at 10% ratio
- ✅ **Small object metrics**: Track mAP by object size (<64px, 64-192px, >192px)
- ✅ **Edge cases**: Define occlusion, orientation, lighting tolerance
- ✅ **Acceptance criteria**: Agent-executable metrics with exact thresholds
- ✅ **Scope guardrails**: Lock to single class, no architecture changes

---

## Work Objectives

### Core Objective
Train YOLO26n on HaGRID "like" class (thumbs up) with negative samples from "no_gesture", using person-stratified train/val split to achieve >0.75 mAP@0.5 on validation, with >0.60 mAP for small objects (<64px).

### Concrete Deliverables
1. **Dataset artifact**: `./hagrid-thumbsup/` with train/val split by person
2. **Config file**: `./dataset.yaml` with 2 classes (thumbs_up, no_gesture)
3. **Trained model**: `./runs/detect/train/weights/best.pt` and `last.pt`
4. **Training logs**: `results.csv`, `args.yaml`, TensorBoard/WandB logs
5. **Evaluation script**: `evaluate.py` with size-stratified metrics
6. **Inference demo**: `inference.py` supporting SAHI for high-res images
7. **Colab notebook**: `hagrid_thumbsup_training.ipynb` with full pipeline

### Definition of Done
- [ ] Dataset downloaded and split (80/20 by user_id)
- [ ] Training completes without OOM errors
- [ ] Final mAP@0.5 > 0.75 on validation set
- [ ] Small object mAP (<64px) > 0.60
- [ ] Inference speed >30 FPS on T4 at 640×640
- [ ] False positive rate on no_gesture <5%

### Must Have
- Person-stratified train/val split (prevent data leakage)
- Include "no_gesture" as negative class (10% of positive samples)
- YOLO26n architecture (ProgLoss + STAL for small objects, NMS-free)
- Training at imgsz=640 for 50 epochs with early stopping (patience=10)
- Mosaic augmentation disabled after epoch 40
- Batch size ≤16 (T4 GPU constraint)
- Save checkpoints to Google Drive (Colab persistence)

### Must NOT Have (Guardrails)
- Random train/val split (causes person leakage)
- Other HaGRID gesture classes (scope creep)
- Custom inference pipeline (use Ultralytics predict API)
- Keypoint detection or multi-task learning
- Training from scratch (MUST use pretrained weights)
- Batch size >16 without gradient accumulation
- Training >100 epochs without validation plateau evidence
- Custom loss functions or architecture modifications

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES - Google Colab with T4 GPU
- **Automated tests**: YES - Tests after implementation
- **Framework**: pytest for validation scripts
- **If TDD**: Each task includes validation as part of acceptance

### QA Policy
Every task includes agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/`.

**For this ML training project:**
- **Data validation**: Python script checks dataset integrity, split ratios, class distribution
- **Training monitoring**: Automated metric tracking, OOM detection
- **Model evaluation**: Precision, recall, mAP by object size
- **Inference testing**: Speed benchmark, accuracy on held-out test set

---

## Execution Strategy

### Sequential Pipeline (No Parallelism)

This is a sequential ML training pipeline - each stage depends on the previous:

```
Wave 1: Data Preparation (45 min)
├── Task 1: Setup Colab environment + mount Drive
├── Task 2: Download HaGRID dataset (like + no_gesture classes)
├── Task 3: Extract and organize images/annotations
└── Task 4: Person-stratified train/val split (80/20)

Wave 2: Data Validation (15 min)
├── Task 5: Validate dataset integrity (corruption check)
├── Task 6: Verify split ratios and class distribution
└── Task 7: Create dataset.yaml configuration

Wave 3: Model Training (45-60 min)
├── Task 8: Setup YOLO26n with pretrained weights
├── Task 9: Configure training hyperparameters
├── Task 10: Train model (50 epochs, early stopping)
└── Task 11: Save best checkpoint to Drive

Wave 4: Evaluation & Analysis (20 min)
├── Task 12: Evaluate on validation set (full metrics)
├── Task 13: Size-stratified evaluation (<64px, 64-192px, >192px)
├── Task 14: Confusion matrix analysis
└── Task 15: Inference speed benchmark

Wave 5: Deployment Artifacts (15 min)
├── Task 16: Create inference script with SAHI support
├── Task 17: Package model + config for download
└── Task 18: Generate final evaluation report

Wave FINAL: Verification (10 min)
├── Task F1: End-to-end pipeline test
└── Task F2: Verify all deliverables exist
```

**Critical Path**: T1 → T2 → T3 → T4 → T8 → T10 → T12 → T18

### Dependency Matrix

| Task | Dependencies | Blocks |
|------|--------------|--------|
| T1 (Setup) | - | T2 |
| T2 (Download) | T1 | T3 |
| T3 (Extract) | T2 | T4 |
| T4 (Split) | T3 | T5, T7 |
| T5 (Validate) | T4 | T6 |
| T6 (Verify) | T5 | T8 |
| T7 (Config) | T4 | T8 |
| T8 (Setup model) | T6, T7 | T9 |
| T9 (Configure) | T8 | T10 |
| T10 (Train) | T9 | T11, T12 |
| T11 (Save) | T10 | - |
| T12 (Evaluate) | T10 | T13, T14, T15 |
| T13 (Size eval) | T12 | T18 |
| T14 (Confusion) | T12 | T18 |
| T15 (Benchmark) | T12 | T18 |
| T16 (Inference) | T12 | T17 |
| T17 (Package) | T11, T16 | T18 |
| T18 (Report) | T13, T14, T15, T17 | F1, F2 |

### Agent Dispatch Summary

All tasks use `unspecified-high` category with Python/ML skills:

- **Wave 1**: Data engineering, file manipulation, dataset handling
- **Wave 2**: Data validation, quality checks
- **Wave 3**: Model training (PyTorch/Ultralytics)
- **Wave 4**: Evaluation, metrics computation
- **Wave 5**: Scripting, packaging, documentation

---

## TODOs

- [ ] 1. Setup Google Colab Environment

  **What to do**:
  - Create new Colab notebook
  - Mount Google Drive for persistence
  - Check GPU availability (`!nvidia-smi`)
  - Install dependencies: `ultralytics`, `roboflow`, `sahi`
  - Verify T4 GPU is active

  **Must NOT do**:
  - Use CPU-only runtime
  - Skip Drive mounting (risk losing work on disconnect)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: Simple setup task, standard Colab procedures

  **Parallelization**:
  - **Can Run In Parallel**: NO (first task)
  - **Blocks**: Task 2 (Download)

  **References**:
  - Colab docs: GPU runtime setup
  - Ultralytics install: `pip install ultralytics`

  **Acceptance Criteria**:
  - [ ] GPU detected: T4 or better
  - [ ] Drive mounted at `/content/drive`
  - [ ] Ultralytics imported without errors
  - [ ] Evidence: Screenshot of `!nvidia-smi` output

  **QA Scenarios**:
  ```
  Scenario: Verify GPU availability
    Tool: Bash
    Steps:
      1. Run `!nvidia-smi`
      2. Verify output shows "Tesla T4" or better
    Expected Result: GPU memory ~15GB, driver version shown
    Evidence: .sisyphus/evidence/task-1-gpu-check.txt
  ```

  **Commit**: NO

---

- [ ] 2. Download HaGRID Dataset

  **What to do**:
  - Download HaGRID from Kaggle or official source
  - Target classes: `like` (thumbs up) + `no_gesture` (negative)
  - Approximate sizes: ~30K images per class
  - Save to `./hagrid-raw/` directory

  **Must NOT do**:
  - Download all 18 classes (scope violation)
  - Skip annotation files (need bounding boxes)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Large file download, potential auth issues

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T1)
  - **Blocked By**: T1
  - **Blocks**: T3

  **References**:
  - Kaggle dataset: `kapitanov/hagrid`
  - Alternative: Hugging Face `neilrigaud/hagrid-subset`
  - HaGRID paper: arxiv.org/abs/2206.08219

  **Acceptance Criteria**:
  - [ ] `like` folder exists with ~30K images
  - [ ] `no_gesture` folder exists with ~3K images (10%)
  - [ ] Annotations downloaded in YOLO format
  - [ ] Total download size: ~5-10GB
  - [ ] Evidence: Directory listing with file counts

  **QA Scenarios**:
  ```
  Scenario: Verify dataset download
    Tool: Bash
    Steps:
      1. Count images in ./hagrid-raw/like/: `ls | wc -l`
      2. Count images in ./hagrid-raw/no_gesture/: `ls | wc -l`
      3. Verify annotation files exist
    Expected Result: like/ has 25K-35K images, no_gesture has 2.5K-3.5K
    Evidence: .sisyphus/evidence/task-2-download-counts.txt
  ```

  **Commit**: NO

---

- [ ] 3. Extract and Organize Dataset

  **What to do**:
  - Extract downloaded archives
  - Organize into structure:
    ```
    hagrid-raw/
    ├── like/
    │   ├── images/
    │   └── annotations/
    └── no_gesture/
        ├── images/
        └── annotations/
    ```
  - Verify image-annotation pairs match

  **Must NOT do**:
  - Mix images and annotations in same folder
  - Delete original archives until verified

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Blocked By**: T2
  - **Blocks**: T4

  **Acceptance Criteria**:
  - [ ] Directory structure matches specification
  - [ ] Every image has corresponding .txt annotation
  - [ ] No orphaned annotations or images
  - [ ] Evidence: Sample file listing (10 random pairs)

  **QA Scenarios**:
  ```
  Scenario: Verify image-annotation pairs
    Tool: Python
    Steps:
      1. List all images in like/images/
      2. Check corresponding annotation exists in like/annotations/
      3. Calculate match ratio
    Expected Result: 100% match, no orphans
    Evidence: .sisyphus/evidence/task-3-pair-verification.json
  ```

  **Commit**: NO

---

- [ ] 4. Person-Stratified Train/Val Split

  **What to do**:
  - **CRITICAL**: Use HaGRID's `user_id` field for splitting
  - Load metadata CSV with user_id annotations
  - Split: 80% users → train, 20% users → val
  - Copy images to:
    ```
    hagrid-thumbsup/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
    ```
  - Combine `like` (positive) and `no_gesture` (negative) in same folders

  **Must NOT do**:
  - Random split (causes person leakage, inflated metrics)
  - Split by image (same person in train and val)

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Critical logic, must prevent data leakage

  **Parallelization**:
  - **Blocked By**: T3
  - **Blocks**: T5, T7

  **References**:
  - HaGRID metadata format (user_id field)
  - Stratified sampling best practices

  **Acceptance Criteria**:
  - [ ] Split by user_id, not by image
  - [ ] No overlapping users between train/val
  - [ ] Train: ~24K like + ~2.4K no_gesture
  - [ ] Val: ~6K like + ~600 no_gesture
  - [ ] Evidence: User overlap check report

  **QA Scenarios**:
  ```
  Scenario: Verify no user leakage
    Tool: Python
    Steps:
      1. Extract user_ids from train set metadata
      2. Extract user_ids from val set metadata
      3. Check intersection: `set(train_users) & set(val_users)`
    Expected Result: Empty set (no overlap)
    Evidence: .sisyphus/evidence/task-4-no-leakage.txt
  
  Scenario: Verify split ratios
    Tool: Python
    Steps:
      1. Count images in train/ and val/
      2. Calculate ratio (should be ~80/20)
    Expected Result: 75-85% in train, 15-25% in val
    Evidence: .sisyphus/evidence/task-4-split-ratio.txt
  ```

  **Commit**: NO

---

- [ ] 5. Validate Dataset Integrity

  **What to do**:
  - Check for corrupted images (can't be opened)
  - Verify annotation format (YOLO: class x y w h)
  - Check for empty annotations
  - Validate bounding box coordinates (0-1 range)

  **Must NOT do**:
  - Skip validation (corrupted data = training failure)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Blocked By**: T4
  - **Blocks**: T6

  **Acceptance Criteria**:
  - [ ] 0 corrupted images
  - [ ] All annotations in valid YOLO format
  - [ ] All boxes within [0,1] range
  - [ ] Evidence: Validation report with error count

  **QA Scenarios**:
  ```
  Scenario: Check for corrupted images
    Tool: Python (PIL/OpenCV)
    Steps:
      1. Attempt to open each image
      2. Count failures
    Expected Result: 0 failures
    Evidence: .sisyphus/evidence/task-5-corruption-check.txt
  
  Scenario: Validate annotation format
    Tool: Python
    Steps:
      1. Parse all .txt annotation files
      2. Check format: 5 values (class, x, y, w, h)
      3. Check all values in [0,1]
    Expected Result: All files valid
    Evidence: .sisyphus/evidence/task-5-annotation-validation.txt
  ```

  **Commit**: NO

---

- [ ] 6. Verify Split Ratios and Class Distribution

  **What to do**:
  - Calculate exact train/val split ratio
  - Verify class balance (thumbs_up vs no_gesture)
  - Check for class imbalance issues
  - Document statistics

  **Must NOT do**:
  - Proceed with imbalanced dataset (>10:1 ratio)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Blocked By**: T5
  - **Blocks**: T8

  **Acceptance Criteria**:
  - [ ] Train/val ratio between 75:25 and 85:15
  - [ ] Class ratio (thumbs_up:no_gesture) between 8:1 and 12:1
  - [ ] Evidence: Distribution statistics JSON

  **QA Scenarios**:
  ```
  Scenario: Calculate class distribution
    Tool: Python
    Steps:
      1. Count class 0 (thumbs_up) in train labels
      2. Count class 1 (no_gesture) in train labels
      3. Calculate ratio
    Expected Result: ~10:1 ratio (acceptable for binary detection)
    Evidence: .sisyphus/evidence/task-6-class-distribution.json
  ```

  **Commit**: NO

---

- [ ] 7. Create dataset.yaml Configuration

  **What to do**:
  - Create `dataset.yaml` for Ultralytics:
    ```yaml
    path: /content/hagrid-thumbsup
    train: train/images
    val: val/images
    names:
      0: thumbs_up
      1: no_gesture
    ```
  - Save to project root

  **Must NOT do**:
  - Use absolute paths that won't work on other machines
  - Skip validation of yaml syntax

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Blocked By**: T4
  - **Blocks**: T8

  **References**:
  - Ultralytics dataset format: https://docs.ultralytics.com/datasets/

  **Acceptance Criteria**:
  - [ ] YAML file created and valid
  - [ ] Paths point to correct directories
  - [ ] 2 classes defined with correct names
  - [ ] Evidence: YAML file content + load test

  **QA Scenarios**:
  ```
  Scenario: Validate YAML syntax
    Tool: Python
    Steps:
      1. Load yaml file with `yaml.safe_load()`
      2. Verify required keys: path, train, val, names
      3. Verify names dict has 2 entries
    Expected Result: No exceptions, valid structure
    Evidence: .sisyphus/evidence/task-7-yaml-validation.txt
  ```

  **Commit**: NO

---

- [ ] 8. Setup YOLO26n Model

  **What to do**:
  - Load YOLO26n from Ultralytics:
  ```python
  model = YOLO('yolo26n.pt')  # Downloads automatically
    ```
  - Verify model architecture (should have 4 heads: P2, P3, P4, P5)
  - Print model summary to confirm

  **Must NOT do**:
  - Use standard YOLOv8n (missing P2 head for small objects)
  - Use YOLOv8n-p6 (wrong direction - for very large objects)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Blocked By**: T6, T7
  - **Blocks**: T9

  **References**:
  - YOLOv8-p2 docs: P2 detection head for small objects

  **Acceptance Criteria**:
  - [ ] Model loaded successfully
  - [ ] Model has 4 detection heads (P2, P3, P4, P5)
  - [ ] ~3.4M parameters, ~17.4 GFLOPs
  - [ ] Evidence: Model summary output

  **QA Scenarios**:
  ```
  Scenario: Verify P2 architecture
    Tool: Python
    Steps:
      1. Load model and print architecture
      2. Search for "Detect" layers in model
      3. Count detection heads
    Expected Result: 4 heads detected, includes P2
    Evidence: .sisyphus/evidence/task-8-architecture-verify.txt
  ```

  **Commit**: NO

---

- [ ] 9. Configure Training Hyperparameters

  **What to do**:
  - Set training arguments:
    ```python
    args = {
        'data': 'dataset.yaml',
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,
        'patience': 10,  # Early stopping
        'save': True,
        'device': 0,
        'workers': 8,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'close_mosaic': 10,  # Disable mosaic for last 10 epochs
    }
    ```

  **Must NOT do**:
  - Batch size >16 (OOM risk on T4)
  - Skip early stopping (wastes time on plateau)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Blocked By**: T8
  - **Blocks**: T10

  **References**:
  - Ultralytics training args: https://docs.ultralytics.com/usage/cfg/

  **Acceptance Criteria**:
  - [ ] All hyperparameters defined
  - [ ] Batch size ≤16
  - [ ] Early stopping configured (patience=10)
  - [ ] close_mosaic set to 10 (disables at epoch 40)
  - [ ] Evidence: Config dict printed

  **QA Scenarios**:
  ```
  Scenario: Verify training config
    Tool: Python
    Steps:
      1. Print all training arguments
      2. Check batch size
      3. Verify patience value
    Expected Result: batch=16, patience=10, close_mosaic=10
    Evidence: .sisyphus/evidence/task-9-config-verify.txt
  ```

  **Commit**: NO

---

- [ ] 10. Train Model

  **What to do**:
  - Start training:
    ```python
    results = model.train(**args)
    ```
  - Monitor metrics in real-time
  - Handle potential OOM errors
  - Save progress to Drive every 10 epochs

  **Must NOT do**:
  - Train on CPU (too slow)
  - Ignore OOM warnings
  - Skip checkpointing (Colab can disconnect)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Long-running task, needs monitoring

  **Parallelization**:
  - **Blocked By**: T9
  - **Blocks**: T11, T12

  **Acceptance Criteria**:
  - [ ] Training starts without errors
  - [ ] GPU utilization >80%
  - [ ] Loss decreasing over first 10 epochs
  - [ ] No OOM errors
  - [ ] Evidence: Training logs (first 10 epochs)

  **QA Scenarios**:
  ```
  Scenario: Monitor training start
    Tool: Python
    Steps:
      1. Check GPU memory: `!nvidia-smi`
      2. Verify training loop starts
      3. Check first epoch loss values
    Expected Result: Training running, loss <10 initially
    Evidence: .sisyphus/evidence/task-10-training-start.txt
  
  Scenario: Check for OOM
    Tool: Bash
    Steps:
      1. Monitor GPU memory during training
      2. Alert if memory >14GB
    Expected Result: Memory stable, no OOM
    Evidence: .sisyphus/evidence/task-10-memory-usage.txt
  ```

  **Commit**: NO

---

- [ ] 11. Save Checkpoint to Google Drive

  **What to do**:
  - Copy best.pt and last.pt to Drive:
    ```bash
    !cp runs/detect/train/weights/*.pt /content/drive/MyDrive/hagrid-thumbsup/
    ```
  - Also save training logs and config

  **Must NOT do**:
  - Skip this step (Colab disconnect = lose work)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Blocked By**: T10
  - **Blocks**: None (can parallel with T12)

  **Acceptance Criteria**:
  - [ ] best.pt copied to Drive
  - [ ] last.pt copied to Drive
  - [ ] results.csv copied to Drive
  - [ ] Evidence: Drive directory listing

  **QA Scenarios**:
  ```
  Scenario: Verify checkpoint saved
    Tool: Bash
    Steps:
      1. List Drive directory
      2. Check file sizes
    Expected Result: Files exist, size >5MB each
    Evidence: .sisyphus/evidence/task-11-drive-backup.txt
  ```

  **Commit**: NO

---

- [ ] 12. Evaluate on Validation Set

  **What to do**:
  - Run validation:
    ```python
    metrics = model.val()
    ```
  - Extract key metrics:
    - mAP@0.5 (primary)
    - mAP@0.5:0.95
    - Precision, Recall
    - Per-class metrics
  - Generate confusion matrix

  **Must NOT do**:
  - Skip validation (need metrics for acceptance)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Blocked By**: T10
  - **Blocks**: T13, T14, T15

  **Acceptance Criteria**:
  - [ ] mAP@0.5 > 0.75
  - [ ] Validation completes without errors
  - [ ] Metrics saved to JSON
  - [ ] Evidence: Full metrics report

  **QA Scenarios**:
  ```
  Scenario: Check mAP threshold
    Tool: Python
    Steps:
      1. Run model.val()
      2. Extract metrics.box.map50
      3. Compare to threshold 0.75
    Expected Result: map50 >= 0.75
    Evidence: .sisyphus/evidence/task-12-map50-validation.txt
  ```

  **Commit**: NO

---

- [ ] 13. Size-Stratified Evaluation

  **What to do**:
  - Evaluate model on different object sizes:
    - Small: <64px
    - Medium: 64-192px
    - Large: >192px
  - Calculate mAP for each size category

  **Must NOT do**:
  - Report only overall mAP (hides small object performance)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Blocked By**: T12
  - **Blocks**: T18

  **Acceptance Criteria**:
  - [ ] mAP@0.5 for small (<64px) > 0.60
  - [ ] mAP@0.5 for medium (>64px) > 0.75
  - [ ] Evidence: Size-stratified metrics JSON

  **QA Scenarios**:
  ```
  Scenario: Evaluate small objects
    Tool: Python
    Steps:
      1. Filter val set for boxes <64px
      2. Run evaluation on subset
      3. Extract mAP@0.5
    Expected Result: mAP >= 0.60 for small objects
    Evidence: .sisyphus/evidence/task-13-small-object-map.json
  ```

  **Commit**: NO

---

- [ ] 14. Confusion Matrix Analysis

  **What to do**:
  - Generate confusion matrix
  - Identify common confusions:
    - Thumbs_up vs no_gesture
    - False positives on similar gestures (if any in val)
  - Calculate false positive rate on no_gesture

  **Must NOT do**:
  - Ignore false positive rate (important for deployment)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Blocked By**: T12
  - **Blocks**: T18

  **Acceptance Criteria**:
  - [ ] Confusion matrix generated
  - [ ] FP rate on no_gesture < 5%
  - [ ] Evidence: Confusion matrix plot + FP rate

  **QA Scenarios**:
  ```
  Scenario: Calculate FP rate
    Tool: Python
    Steps:
      1. Count predictions on no_gesture class
      2. Count false positives (predicted thumbs_up on no_gesture)
      3. Calculate FP rate
    Expected Result: FP rate < 0.05
    Evidence: .sisyphus/evidence/task-14-fp-rate.txt
  ```

  **Commit**: NO

---

- [ ] 15. Inference Speed Benchmark

  **What to do**:
  - Benchmark inference speed on T4:
    ```python
    import time
    times = []
    for _ in range(100):
        start = time.time()
        model.predict(img, imgsz=640)
        times.append(time.time() - start)
    fps = 1 / np.mean(times)
    ```
  - Test at 640×640 and 1280×1280

  **Must NOT do**:
  - Benchmark on CPU (not representative)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Blocked By**: T12
  - **Blocks**: T18

  **Acceptance Criteria**:
  - [ ] FPS > 30 at 640×640
  - [ ] Evidence: Benchmark results with mean/std

  **QA Scenarios**:
  ```
  Scenario: Measure inference speed
    Tool: Python
    Steps:
      1. Run 100 inference iterations
      2. Calculate mean time per image
      3. Convert to FPS
    Expected Result: FPS >= 30
    Evidence: .sisyphus/evidence/task-15-fps-benchmark.txt
  ```

  **Commit**: NO

---

- [ ] 16. Create Inference Script with SAHI Support

  **What to do**:
  - Create `inference.py` script:
    ```python
    from ultralytics import YOLO
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    
    def predict(image_path, use_sahi=False):
        if use_sahi and image_large:
            # Use SAHI for high-res images with small objects
            detection_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path='best.pt',
                confidence_threshold=0.5
            )
            result = get_sliced_prediction(
                image_path,
                detection_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )
        else:
            model = YOLO('best.pt')
            result = model.predict(image_path, imgsz=640)
        return result
    ```

  **Must NOT do**:
  - Skip SAHI support (needed for small objects in high-res images)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Blocked By**: T12
  - **Blocks**: T17

  **References**:
  - SAHI docs: https://github.com/obss/sahi

  **Acceptance Criteria**:
  - [ ] Script runs without errors
  - [ ] Supports both standard and SAHI inference
  - [ ] Configurable confidence threshold
  - [ ] Evidence: Test run on sample image

  **QA Scenarios**:
  ```
  Scenario: Test inference script
    Tool: Python
    Steps:
      1. Run inference on sample image
      2. Verify detections returned
      3. Check bounding box format
    Expected Result: Valid predictions returned
    Evidence: .sisyphus/evidence/task-16-inference-test.txt
  ```

  **Commit**: NO

---

- [ ] 17. Package Model for Download

  **What to do**:
  - Create zip file with:
    - `best.pt` (trained model)
    - `dataset.yaml` (dataset config)
    - `inference.py` (inference script)
    - `requirements.txt` (dependencies)
    - `README.md` (usage instructions)
  - Save to Drive for easy download

  **Must NOT do**:
  - Include training data (too large)
  - Skip requirements.txt (dependencies unclear)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Blocked By**: T11, T16
  - **Blocks**: T18

  **Acceptance Criteria**:
  - [ ] Zip file created with all required files
  - [ ] Zip size <100MB (model is ~6MB)
  - [ ] Saved to Drive
  - [ ] Evidence: Zip file listing

  **QA Scenarios**:
  ```
  Scenario: Verify package contents
    Tool: Bash
    Steps:
      1. Unzip package
      2. List contents
      3. Verify required files present
    Expected Result: All 5 files present
    Evidence: .sisyphus/evidence/task-17-package-contents.txt
  ```

  **Commit**: NO

---

- [ ] 18. Generate Final Evaluation Report

  **What to do**:
  - Create comprehensive report with:
    - Dataset statistics (size, split, class distribution)
    - Training configuration (hyperparameters, epochs)
    - Final metrics (mAP@0.5, mAP@0.5:0.95, precision, recall)
    - Size-stratified metrics (small/medium/large)
    - Confusion matrix analysis
    - Inference speed benchmark
    - Sample predictions (visualizations)
  - Save as `evaluation_report.md` and `evaluation_report.pdf`

  **Must NOT do**:
  - Skip visualizations (important for understanding)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Blocked By**: T13, T14, T15, T17
  - **Blocks**: F1, F2

  **Acceptance Criteria**:
  - [ ] Report contains all required sections
  - [ ] All metrics from acceptance criteria documented
  - [ ] Visualizations included
  - [ ] Saved to Drive
  - [ ] Evidence: Report preview (first page)

  **QA Scenarios**:
  ```
  Scenario: Verify report completeness
    Tool: Python
    Steps:
      1. Check all required sections exist
      2. Verify metrics meet thresholds
      3. Confirm visualizations present
    Expected Result: All sections complete, all thresholds met
    Evidence: .sisyphus/evidence/task-18-report-complete.txt
  ```

  **Commit**: NO

---

## Final Verification Wave

### F1. End-to-End Pipeline Test

**What to do**:
- Run complete pipeline from data download to inference
- Test on held-out sample images
- Verify all steps execute without errors

**Acceptance Criteria**:
- [ ] Pipeline completes without manual intervention
- [ ] All 18 tasks execute successfully
- [ ] Final model can detect thumbs up in test images
- [ ] Evidence: Pipeline execution log

### F2. Verify All Deliverables Exist

**What to do**:
- Check all required deliverables:
  - [ ] Trained model (`best.pt`)
  - [ ] Dataset config (`dataset.yaml`)
  - [ ] Inference script (`inference.py`)
  - [ ] Evaluation report (`evaluation_report.md`)
  - [ ] Packaged model (`hagrid-thumbsup-model.zip`)
- All saved to Google Drive

**Acceptance Criteria**:
- [ ] All 5 deliverables present
- [ ] All in Google Drive
- [ ] File sizes reasonable
- [ ] Evidence: Deliverables checklist

---

## Commit Strategy

All work in Colab - no git commits. Progress saved to Google Drive:
- Checkpoints every 10 epochs
- Final model and artifacts
- Training logs and metrics

---

## Success Criteria

### Verification Commands
```bash
# Check GPU availability
!nvidia-smi
# Expected: Tesla T4, ~15GB memory

# Verify dataset
python -c "import yaml; d=yaml.safe_load(open('dataset.yaml')); print(f'Classes: {len(d[\"names\"])}')"
# Expected: Classes: 2

# Check model metrics
python -c "from ultralytics import YOLO; m=YOLO('best.pt'); r=m.val(); print(f'mAP50: {r.box.map50:.3f}')"
# Expected: mAP50: 0.750+ (or 0.750 if exactly 0.75)

# Test inference speed
python inference.py --benchmark
# Expected: FPS > 30
```

### Final Checklist
- [ ] mAP@0.5 > 0.75 on validation
- [ ] Small object mAP (<64px) > 0.60
- [ ] Inference FPS > 30 on T4
- [ ] FP rate on no_gesture < 5%
- [ ] All deliverables saved to Drive
- [ ] End-to-end pipeline tested

---

## Notes for Executor

### Critical Warnings
1. **Person-stratified split is MANDATORY** - Random split causes data leakage and inflated metrics
2. **Use YOLO26n** - Latest model with ProgLoss + STAL specifically for small object detection, NMS-free for simpler deployment
3. **Include no_gesture as negative class** - Reduces false positives significantly
4. **Save to Drive frequently** - Colab disconnects are common
5. **Monitor GPU memory** - Batch size 16 is maximum for T4

### Colab-Specific Tips
- Use `!pip install` at start of notebook
- Mount Drive with `from google.colab import drive; drive.mount('/content/drive')`
- Use `%cd` to navigate directories
- Monitor GPU with `!nvidia-smi` in separate cell
- Download large datasets with `!wget` or `!gdown`

### Troubleshooting
- **OOM errors**: Reduce batch size to 8, enable gradient accumulation
- **Slow training**: Verify GPU is active (not CPU)
- **Colab disconnect**: Restart from checkpoint saved in Drive
- **Poor metrics**: Check if person leakage occurred (redo split)

### Resources
- HaGRID paper: https://arxiv.org/abs/2206.08219
- Ultralytics docs: https://docs.ultralytics.com
- SAHI docs: https://github.com/obss/sahi
- Small object detection guide: https://y-t-g.github.io/tutorials/yolov8-increase-accuracy
