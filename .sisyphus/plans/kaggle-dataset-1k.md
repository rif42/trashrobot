# Work Plan: Integrate Kaggle HaGRID Dataset (1k Images)

## TL;DR

> **Objective**: Download and integrate 1,000 images from Kaggle HaGRID dataset (`innominate817/hagrid-sample-30k-384p`) for training OK sign detection.
>
> **Deliverables**:
> - Download script that fetches 1k images (500 ok + 500 other gestures)
> - COCO-to-YOLO annotation converter
> - Updated `data.yaml` with single-class configuration (ok_sign only)
> - Organized train/val/test splits (800/100/100)
>
> **Estimated Effort**: Short (2-4 hours)
> **Parallel Execution**: NO - sequential workflow with dependencies
> **Critical Path**: Download → Inspect → Convert → Sample → Split → Verify

---

## Context

### Original Request
Switch training dataset to use Kaggle repository `innominate817/hagrid-sample-30k-384p` instead of current local dataset.

### Interview Summary
**Key Decisions**:
- **Dataset size**: Download only 1,000 images (not full 30k) for quick training/testing
- **Classes**: Single class - `ok_sign` only (remove `person` class)
- **Sampling strategy**: 500 images with "ok" gesture + 500 images with other gestures (as negative examples)
- **Current dataset handling**: Replace completely - clear `data/processed/` and populate with new data
- **Split ratios**: 80/10/10 (train/val/test)
- **Data location**: `data/processed/` (replace existing)

**Research Findings** (from Librarian Agent):
- **Total Images**: ~31,833 images in the full dataset
- **Annotation Format**: **CSV** (train.csv or annotations.csv) - NOT COCO JSON
- **Image Format**: JPEG (.jpg), 384p resolution  
- **Bounding Boxes**: **Already normalized** [x, y, width, height] (0.0 to 1.0 range)
  - *Note*: Need to convert to YOLO format [x_center, y_center, width, height]
- **Folder Structure**: Organized by gesture class:
  ```
  hagrid-sample-30k-384p/
  ├── train/
  │   ├── call/
  │   ├── dislike/
  │   ├── ok/              <-- Our target class
  │   └── [16 other gesture folders]/
  └── annotations.csv
  ```
- **Classes**: 18 gestures + `no_gesture` (includes "ok" which we need)
- **Kaggle API credentials**: Already configured in `kaggle.json`

### Metis Review
**Identified Gaps** (addressed):
- **Critical**: HaGRID has hand gesture annotations, NOT full person bounding boxes → Resolved: Use single class (ok_sign only)
- **Risk**: Annotation format mismatch (COCO vs YOLO) → Resolved: Create converter script
- **Risk**: Sampling strategy may produce imbalanced dataset → Resolved: Stratified sampling (500 ok + 500 others)
- **Risk**: Data integrity (corrupted downloads) → Resolved: Add validation checks

---

## Work Objectives

### Core Objective
Create a complete data pipeline that downloads 1,000 images from Kaggle HaGRID dataset, converts annotations from COCO to YOLO format, samples 500 "ok" + 500 other gesture images, splits into train/val/test, and updates configuration for single-class training.

### Concrete Deliverables
1. `scripts/download_hagrid_kaggle.py` - Download script with resume capability
2. `scripts/convert_coco_to_yolo.py` - COCO-to-YOLO annotation converter
3. `scripts/prepare_hagrid.py` - Sampling and splitting script
4. Updated `ok-gesture-poc/config/data.yaml` - Single-class configuration
5. Organized dataset in `data/processed/` (train/val/test splits)
6. Dataset verification report

### Definition of Done
- [ ] Download completes successfully (1,000 images + annotations)
- [ ] All images pass integrity check (not corrupted)
- [ ] Annotations converted to YOLO format (.txt files)
- [ ] Train/val/test splits created with no data leakage
- [ ] `data.yaml` updated with correct paths and single-class config
- [ ] Verification script reports: "Dataset ready for training"

### Must Have
- Exactly 1,000 images (500 ok + 500 other gestures)
- YOLO format annotations (.txt files with normalized coordinates)
- 80/10/10 train/val/test split
- No corrupted images
- Updated data.yaml with single class (ok_sign)

### Must NOT Have (Guardrails)
- ❌ Full 30k dataset download (limit to 1k)
- ❌ Person class in training (single class only)
- ❌ Data leakage between splits
- ❌ Corrupted images without handling
- ❌ Modified training pipeline code (data integration only)

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES (current project has test scripts)
- **Automated tests**: Tests-after (verify after implementation)
- **Framework**: Python scripts with assertions

### QA Policy
Every task MUST include agent-executed QA scenarios:
- **Download verification**: File existence, size checks, integrity validation
- **Conversion verification**: Format validation, coordinate bounds checking
- **Sampling verification**: Class distribution, count validation
- **Split verification**: No overlap between splits, correct proportions
- **Config verification**: YAML parsing, path validation

---

## Execution Strategy

### Sequential Workflow (Dependencies Required)

```
Wave 1 (Foundation - Download & Setup):
├── Task 1: Create download script
├── Task 2: Download HaGRID dataset from Kaggle
└── Task 3: Verify download integrity

Wave 2 (Processing - Conversion & Sampling):
├── Task 4: Inspect annotation format
├── Task 5: Create COCO-to-YOLO converter
├── Task 6: Convert annotations
├── Task 7: Sample 1k images (500 ok + 500 others)
└── Task 8: Create train/val/test splits

Wave 3 (Integration - Configuration & Verification):
├── Task 9: Update data.yaml for single class
├── Task 10: Verify dataset structure
└── Task 11: Test training pipeline integration

Wave FINAL (Review):
├── Task F1: Plan compliance audit (oracle)
└── Task F2: End-to-end verification

Critical Path: Task 1 → Task 2 → Task 4 → Task 5 → Task 6 → Task 7 → Task 8 → Task 9 → Task 10 → F1-F2
```

### Agent Dispatch Summary
- **Wave 1**: 3 tasks → `unspecified-high` (download, verification)
- **Wave 2**: 5 tasks → `unspecified-high` (conversion, sampling, splitting)
- **Wave 3**: 3 tasks → `quick` (config update, verification)
- **FINAL**: 2 tasks → `oracle`, `unspecified-high` (compliance audit, E2E test)

---

## TODOs

> **A task WITHOUT QA Scenarios is INCOMPLETE.**

- [ ] 1. Create Kaggle download script

  **What to do**:
  - Create `scripts/download_hagrid_kaggle.py`
  - Use `kaggle` CLI or `kagglehub` library to download dataset
  - Implement resume capability for interrupted downloads
  - Download to temporary location first, then extract
  - Set random seed for reproducibility

  **Must NOT do**:
  - Do NOT download full 30k dataset
  - Do NOT extract directly to data/processed/
  - Do NOT skip Kaggle API credential validation

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required
  - **Justification**: Download script involves file I/O, subprocess calls to kaggle CLI, and error handling. Needs careful implementation but no specialized domain knowledge.

  **Parallelization**:
  - **Can Run In Parallel**: NO (Wave 1 start)
  - **Blocks**: Task 2 (actual download)
  - **Blocked By**: None

  **References**:
  - `kaggle.json` in project root - API credentials
  - Kaggle dataset: `innominate817/hagrid-sample-30k-384p`
  - Example: `scripts/prepare_dataset.py` - shows directory structure patterns
  - Kaggle API docs: `https://github.com/Kaggle/kaggle-api`

  **Acceptance Criteria**:
  - [ ] Script exists at `scripts/download_hagrid_kaggle.py`
  - [ ] Script validates Kaggle API credentials before download
  - [ ] Script accepts `--output-dir` argument
  - [ ] Script has `--resume` flag for interrupted downloads
  - [ ] Script returns exit code 0 on success, non-zero on failure

  **QA Scenarios**:
  ```
  Scenario: Script validates Kaggle credentials
    Tool: Bash
    Steps:
      1. Run: python scripts/download_hagrid_kaggle.py --check-credentials
    Expected Result: Script reports "Kaggle API credentials valid" and exits 0
    Failure Indicators: "Kaggle API credentials not found" or exit code != 0
    Evidence: .sisyphus/evidence/task-1-creds-check.log

  Scenario: Script handles missing credentials gracefully
    Tool: Bash
    Preconditions: Temporarily rename kaggle.json to kaggle.json.bak
    Steps:
      1. Run: python scripts/download_hagrid_kaggle.py --check-credentials
      2. Restore kaggle.json
    Expected Result: Script reports error and exits with non-zero code
    Evidence: .sisyphus/evidence/task-1-creds-missing.log
  ```

  **Commit**: YES
  - Message: `feat(scripts): add Kaggle download script for HaGRID dataset`
  - Files: `scripts/download_hagrid_kaggle.py`

- [ ] 2. Download HaGRID dataset from Kaggle

  **What to do**:
  - Execute the download script
  - Download `innominate817/hagrid-sample-30k-384p` dataset
  - Extract to `data/hagrid_raw/` (temporary location)
  - Verify download completed (check file counts, sizes)
  - Estimate: 30k images ~8-15GB download

  **Must NOT do**:
  - Do NOT move files to data/processed/ yet
  - Do NOT delete existing data until Task 8

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required
  - **Justification**: Network I/O intensive task. Requires monitoring download progress and handling network interruptions.

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 1)
  - **Blocks**: Task 3, Task 4
  - **Blocked By**: Task 1

  **References**:
  - `scripts/download_hagrid_kaggle.py` (from Task 1)
  - `kaggle.json` - API credentials

  **Acceptance Criteria**:
  - [ ] Download completes without errors
  - [ ] `data/hagrid_raw/` directory exists with extracted contents
  - [ ] At least 30,000 image files present
  - [ ] Annotation files present (JSON format)
  - [ ] Download log saved to `.sisyphus/evidence/task-2-download.log`

  **QA Scenarios**:
  ```
  Scenario: Download completes successfully
    Tool: Bash
    Steps:
      1. Run: python scripts/download_hagrid_kaggle.py --output-dir data/hagrid_raw/
      2. Wait for completion
      3. Count images: find data/hagrid_raw/ -name "*.jpg" | wc -l
    Expected Result: Image count >= 30000, exit code 0
    Evidence: .sisyphus/evidence/task-2-download.log

  Scenario: Resume interrupted download
    Tool: Bash
    Preconditions: Partial download exists in data/hagrid_raw/
    Steps:
      1. Interrupt download mid-way (Ctrl+C)
      2. Run with --resume flag
    Expected Result: Download resumes from where it left off
    Evidence: .sisyphus/evidence/task-2-resume.log
  ```

  **Commit**: NO (download is temporary, don't commit large files)
  - Files: None (data goes to .gitignore)

- [ ] 3. Verify download integrity

  **What to do**:
  - Check all downloaded images can be opened (not corrupted)
  - Verify annotation files are valid JSON
  - Count total images and annotations match
  - Report any corrupted files
  - Create manifest of valid files

  **Must NOT do**:
  - Do NOT delete corrupted files automatically (log them for review)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required
  - **Justification**: File validation task. Requires OpenCV or PIL for image verification and JSON parsing.

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 2)
  - **Blocks**: Task 4
  - **Blocked By**: Task 2

  **References**:
  - `data/hagrid_raw/` - Downloaded data location
  - Example: `scripts/prepare_dataset.py:verify_dataset()` - verification pattern

  **Acceptance Criteria**:
  - [ ] All images pass integrity check (can be opened by OpenCV)
  - [ ] Annotation JSON is valid and parseable
  - [ ] Image count matches annotation count (±5% tolerance for missing pairs)
  - [ ] Corruption report saved if any issues found

  **QA Scenarios**:
  ```
  Scenario: All images pass integrity check
    Tool: Bash
    Steps:
      1. Run: python scripts/verify_download.py --data-dir data/hagrid_raw/
    Expected Result: "All images valid: 30000/30000", exit code 0
    Evidence: .sisyphus/evidence/task-3-integrity-report.json

  Scenario: Corrupted images detected and reported
    Tool: Bash
    Preconditions: Manually corrupt one test image
    Steps:
      1. Run verification script
    Expected Result: Script reports corrupted files but continues, exit code 0
    Evidence: .sisyphus/evidence/task-3-corruption-report.txt
  ```

  **Commit**: NO
  - Files: None

- [ ] 4. Inspect annotation format

  **What to do**:
  - Read annotation JSON files
  - Document structure: fields, types, coordinate system
  - Identify class names and IDs used in HaGRID
  - Map HaGRID classes to our single class (ok_sign)
  - Document bounding box format (COCO: [x, y, width, height])

  **Must NOT do**:
  - Do NOT assume annotation format without inspection
  - Do NOT modify annotation files

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required
  - **Justification**: Data inspection task. Requires JSON parsing and documentation.

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 3)
  - **Blocks**: Task 5
  - **Blocked By**: Task 3

  **References**:
  - `data/hagrid_raw/` - Annotation files location
  - COCO format spec: https://cocodataset.org/#format-data

  **Acceptance Criteria**:
  - [ ] Annotation structure documented in `.sisyphus/evidence/task-4-annotation-structure.md`
  - [ ] Class mapping table created (HaGRID class → our class)
  - [ ] Bounding box coordinate system identified
  - [ ] Sample annotations printed for verification

  **QA Scenarios**:
  ```
  Scenario: Annotation structure inspected
    Tool: Bash
    Steps:
      1. Run: python scripts/inspect_annotations.py --data-dir data/hagrid_raw/
    Expected Result: Outputs JSON structure summary, class list, sample annotations
    Evidence: .sisyphus/evidence/task-4-annotation-structure.md
  ```

  **Commit**: YES
  - Message: `docs: document HaGRID annotation format`
  - Files: `.sisyphus/evidence/task-4-annotation-structure.md`

- [ ] 5. Create COCO-to-YOLO converter

  **What to do**:
  - Create `scripts/convert_coco_to_yolo.py`
  - Read COCO JSON annotations
  - Convert to YOLO format (one .txt file per image)
  - YOLO format: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0-1)
  - Handle class remapping (HaGRID "ok" → class 0)
  - Handle images with no annotations (negative examples)

  **Must NOT do**:
  - Do NOT use pixel coordinates (must normalize)
  - Do NOT create empty label files for positive examples

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required
  - **Justification**: Data transformation task. Requires coordinate math and file I/O.

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 4)
  - **Blocks**: Task 6
  - **Blocked By**: Task 4

  **References**:
  - Task 4 output: annotation structure documentation
  - `scripts/prepare_dataset.py` - shows YOLO label file creation
  - YOLO format spec: https://docs.ultralytics.com/datasets/detect/

  **Acceptance Criteria**:
  - [ ] Script exists at `scripts/convert_coco_to_yolo.py`
  - [ ] Script accepts `--input-dir` and `--output-dir` arguments
  - [ ] Script accepts `--class-map` argument for remapping
  - [ ] Script outputs .txt files in YOLO format
  - [ ] Coordinates are normalized (0-1 range)

  **QA Scenarios**:
  ```
  Scenario: Converter creates valid YOLO annotations
    Tool: Bash
    Steps:
      1. Run: python scripts/convert_coco_to_yolo.py --input-dir data/hagrid_raw/ --output-dir data/hagrid_yolo/
      2. Check output: head -5 data/hagrid_yolo/labels/train/00001.txt
    Expected Result: Lines contain 5 space-separated numbers (class x y w h), all in [0,1] range
    Evidence: .sisyphus/evidence/task-5-sample-labels.txt

  Scenario: Class remapping works correctly
    Tool: Bash
    Steps:
      1. Run converter with --class-map '{"ok": 0}'
      2. Check that "ok" gestures have class_id 0 in output
    Expected Result: All ok gestures map to class 0
    Evidence: .sisyphus/evidence/task-5-class-mapping.json
  ```

  **Commit**: YES
  - Message: `feat(scripts): add COCO to YOLO annotation converter`
  - Files: `scripts/convert_coco_to_yolo.py`

- [ ] 6. Convert annotations to YOLO format

  **What to do**:
  - Execute the converter script
  - Process all 30k annotations
  - Create YOLO-format label files
  - Verify conversion accuracy (sample check)
  - Handle errors gracefully

  **Must NOT do**:
  - Do NOT skip corrupted annotations (log and continue)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required
  - **Justification**: Batch processing task. Requires monitoring progress and error handling.

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 5)
  - **Blocks**: Task 7
  - **Blocked By**: Task 5

  **References**:
  - `scripts/convert_coco_to_yolo.py`
  - `data/hagrid_raw/` - Input
  - `data/hagrid_yolo/` - Output (temporary)

  **Acceptance Criteria**:
  - [ ] All 30k images have corresponding .txt label files
  - [ ] Label files are in YOLO format
  - [ ] Class 0 = ok_sign (remapped from HaGRID "ok")
  - [ ] No errors in conversion log

  **QA Scenarios**:
  ```
  Scenario: Conversion completes successfully
    Tool: Bash
    Steps:
      1. Run: python scripts/convert_coco_to_yolo.py --input-dir data/hagrid_raw/ --output-dir data/hagrid_yolo/
      2. Count output files: find data/hagrid_yolo/labels/ -name "*.txt" | wc -l
    Expected Result: File count >= 30000, exit code 0
    Evidence: .sisyphus/evidence/task-6-conversion.log

  Scenario: YOLO format validation
    Tool: Bash
    Steps:
      1. Sample 10 random label files
      2. Verify each line has 5 values
      3. Verify all values are in [0,1] range
    Expected Result: All 10 files pass validation
    Evidence: .sisyphus/evidence/task-6-format-validation.log
  ```

  **Commit**: NO
  - Files: None (intermediate data)

- [ ] 7. Sample 1k images (500 ok + 500 other gestures)

  **What to do**:
  - Select 500 images with "ok" gesture (positive examples)
  - Select 500 images with other gestures (negative examples)
  - Use stratified random sampling with seed for reproducibility
  - Copy selected images and labels to `data/sampled/`
  - Create manifest of selected files

  **Must NOT do**:
  - Do NOT include images without any gesture (if using other gestures as negatives)
  - Do NOT sample without stratification (must ensure 500/500 split)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required
  - **Justification**: Data sampling task. Requires random sampling with constraints.

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 6)
  - **Blocks**: Task 8
  - **Blocked By**: Task 6

  **References**:
  - `data/hagrid_yolo/` - Converted annotations
  - Task 4 output - class mapping information

  **Acceptance Criteria**:
  - [ ] Exactly 1,000 images selected
  - [ ] 500 images with "ok" gesture (class 0)
  - [ ] 500 images with other gestures (classes 1-17, treated as background)
  - [ ] Random seed set for reproducibility
  - [ ] Manifest saved: `.sisyphus/evidence/task-7-sampled-manifest.json`

  **QA Scenarios**:
  ```
  Scenario: Sampling produces correct distribution
    Tool: Bash
    Steps:
      1. Run: python scripts/sample_dataset.py --input-dir data/hagrid_yolo/ --output-dir data/sampled/ --n-ok 500 --n-other 500 --seed 42
      2. Count ok images: grep -l "^0 " data/sampled/labels/*.txt | wc -l
      3. Count other images: ls data/sampled/images/ | wc -l
    Expected Result: 500 ok images, 1000 total images
    Evidence: .sisyphus/evidence/task-7-sampling-report.json

  Scenario: Reproducible sampling
    Tool: Bash
    Steps:
      1. Run sampling twice with same seed
      2. Compare manifests
    Expected Result: Identical file lists
    Evidence: .sisyphus/evidence/task-7-reproducibility-check.txt
  ```

  **Commit**: NO
  - Files: None (intermediate data)

- [ ] 8. Create train/val/test splits

  **What to do**:
  - Clear existing `data/processed/` directory
  - Create new train/val/test structure:
    - `data/processed/train/images/` and `data/processed/train/labels/`
    - `data/processed/val/images/` and `data/processed/val/labels/`
    - `data/processed/test/images/` and `data/processed/test/labels/`
  - Split 1,000 images: 800 train, 100 val, 100 test
  - Use stratified split (maintain 50/50 ok/other ratio in each split)
  - Ensure no data leakage (no image appears in multiple splits)

  **Must NOT do**:
  - Do NOT keep existing data in processed/ (clear completely)
  - Do NOT create random splits without stratification
  - Do NOT allow duplicate images across splits

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required
  - **Justification**: Data splitting task. Requires careful file operations and validation.

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 7)
  - **Blocks**: Task 9
  - **Blocked By**: Task 7

  **References**:
  - `data/sampled/` - 1k sampled images
  - `scripts/prepare_dataset.py:organize_dataset()` - shows split logic

  **Acceptance Criteria**:
  - [ ] `data/processed/` cleared and recreated
  - [ ] 800 train images, 100 val images, 100 test images
  - [ ] Each split has ~50% ok, ~50% other gesture images
  - [ ] No image appears in multiple splits
  - [ ] All images have corresponding label files

  **QA Scenarios**:
  ```
  Scenario: Splits created with correct proportions
    Tool: Bash
    Steps:
      1. Run: python scripts/split_dataset.py --input-dir data/sampled/ --output-dir data/processed/ --split 0.8 0.1 0.1 --seed 42
      2. Count files in each split
    Expected Result: 800 train, 100 val, 100 test
    Evidence: .sisyphus/evidence/task-8-split-counts.json

  Scenario: No data leakage between splits
    Tool: Bash
    Steps:
      1. Extract image names from each split
      2. Check for overlaps
    Expected Result: Zero overlaps (intersection is empty)
    Evidence: .sisyphus/evidence/task-8-leakage-check.txt

  Scenario: Stratification maintained
    Tool: Bash
    Steps:
      1. Count ok images in each split
      2. Verify ~50% ok in train, val, test
    Expected Result: Each split has ~50% ok images (400 train, 50 val, 50 test)
    Evidence: .sisyphus/evidence/task-8-stratification-report.json
  ```

  **Commit**: NO
  - Files: None (data goes to .gitignore)

- [ ] 9. Update data.yaml for single class

  **What to do**:
  - Update `ok-gesture-poc/config/data.yaml`:
    - Change `path` to `./data/processed`
    - Remove `person` class (keep only `ok_sign`)
    - Set `nc: 1` (single class)
    - Keep train/val/test paths
  - Verify YAML syntax is valid
  - Document the change

  **Must NOT do**:
  - Do NOT keep person class in config (single class only)
  - Do NOT modify other config files

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None required
  - **Justification**: Simple configuration file edit. Low complexity.

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 8)
  - **Blocks**: Task 10
  - **Blocked By**: Task 8

  **References**:
  - `ok-gesture-poc/config/data.yaml` - current config
  - YOLO data.yaml spec: https://docs.ultralytics.com/datasets/detect/

  **Acceptance Criteria**:
  - [ ] `data.yaml` has `nc: 1`
  - [ ] `data.yaml` has only `ok_sign` class
  - [ ] `data.yaml` has correct paths to processed/
  - [ ] YAML parses without errors

  **QA Scenarios**:
  ```
  Scenario: data.yaml updated correctly
    Tool: Bash
    Steps:
      1. Read: ok-gesture-poc/config/data.yaml
      2. Verify: nc: 1
      3. Verify: names: [ok_sign]
    Expected Result: Single class configuration confirmed
    Evidence: .sisyphus/evidence/task-9-data-yaml.txt

  Scenario: YAML syntax valid
    Tool: Bash
    Steps:
      1. Run: python -c "import yaml; yaml.safe_load(open('ok-gesture-poc/config/data.yaml'))"
    Expected Result: No errors, exit code 0
    Evidence: .sisyphus/evidence/task-9-yaml-validation.log
  ```

  **Commit**: YES
  - Message: `config: update data.yaml for single-class (ok_sign) training`
  - Files: `ok-gesture-poc/config/data.yaml`

- [ ] 10. Verify dataset structure

  **What to do**:
  - Run verification script on new dataset
  - Check image/label pair completeness
  - Verify YOLO format in all label files
  - Check class distribution
  - Generate verification report

  **Must NOT do**:
  - Do NOT skip verification (required before training)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required
  - **Justification**: Validation task. Comprehensive checks required.

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 9)
  - **Blocks**: Task 11
  - **Blocked By**: Task 9

  **References**:
  - `scripts/prepare_dataset.py:verify_dataset()` - existing verification logic
  - `data/processed/` - new dataset location

  **Acceptance Criteria**:
  - [ ] All images have corresponding labels
  - [ ] No orphaned images or labels
  - [ ] All label files in valid YOLO format
  - [ ] Class distribution matches expected (50/50 ok/other)
  - [ ] Verification report generated

  **QA Scenarios**:
  ```
  Scenario: Dataset passes verification
    Tool: Bash
    Steps:
      1. Run: python scripts/verify_dataset.py --data-dir data/processed/
    Expected Result: "Dataset verification PASSED", exit code 0
    Evidence: .sisyphus/evidence/task-10-verification-report.json

  Scenario: Dataset statistics reported
    Tool: Bash
    Steps:
      1. Run verification script
      2. Check output for image counts per split
      3. Check output for class distribution
    Expected Result: Statistics match expected (800/100/100 split, 50/50 class balance)
    Evidence: .sisyphus/evidence/task-10-dataset-stats.json
  ```

  **Commit**: NO
  - Files: None

- [ ] 11. Test training pipeline integration

  **What to do**:
  - Run a quick training test (1 epoch) to verify pipeline works
  - Verify YOLO can load the dataset
  - Check for any data loading errors
  - Confirm model trains without errors

  **Must NOT do**:
  - Do NOT run full training (just 1 epoch test)
  - Do NOT modify training script

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None required
  - **Justification**: Integration test. Requires running actual training code.

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 10)
  - **Blocks**: FINAL wave
  - **Blocked By**: Task 10

  **References**:
  - `ok-gesture-poc/scripts/train.py` - training script
  - `ok-gesture-poc/config/data.yaml` - updated config

  **Acceptance Criteria**:
  - [ ] Training starts without errors
  - [ ] 1 epoch completes successfully
  - [ ] Model saves checkpoint
  - [ ] No data loading errors

  **QA Scenarios**:
  ```
  Scenario: Training pipeline integration test
    Tool: Bash
    Steps:
      1. Run: cd ok-gesture-poc && python scripts/train.py --epochs 1 --model n --data config/data.yaml
      2. Wait for epoch 1 to complete
    Expected Result: "Training Complete!" message, exit code 0
    Evidence: .sisyphus/evidence/task-11-training-test.log

  Scenario: Dataset loads correctly
    Tool: Bash
    Steps:
      1. Check training log for "Dataset:", "Images:", "Labels:" counts
    Expected Result: 800 train images, 100 val images loaded
    Evidence: .sisyphus/evidence/task-11-data-loading.log
  ```

  **Commit**: NO
  - Files: None (test artifacts)

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. Verify all tasks completed:
  - Download script exists and works
  - 1,000 images downloaded and converted
  - Splits created (800/100/100)
  - data.yaml updated to single class
  - Training pipeline integration test passed
  Check evidence files exist in .sisyphus/evidence/.
  Output: `Must Have [5/5] | Must NOT Have [5/5] | Tasks [11/11] | VERDICT: APPROVE/REJECT`

- [ ] F2. **End-to-End Verification** — `unspecified-high`
  Run complete pipeline from scratch (clean state):
  1. Delete data/hagrid_raw/, data/hagrid_yolo/, data/sampled/, data/processed/
  2. Run all tasks in sequence
  3. Verify final dataset structure
  4. Run 1-epoch training test
  Save comprehensive report to `.sisyphus/evidence/final-e2e-report.md`.
  Output: `E2E Test [PASS/FAIL] | VERDICT`

---

## Commit Strategy

- **Task 1**: `feat(scripts): add Kaggle download script for HaGRID dataset`
- **Task 4**: `docs: document HaGRID annotation format`
- **Task 5**: `feat(scripts): add COCO to YOLO annotation converter`
- **Task 9**: `config: update data.yaml for single-class (ok_sign) training`
- **F1-F2**: `chore: integrate Kaggle HaGRID dataset (1k images)` (final commit after verification)

---

## Success Criteria

### Verification Commands
```bash
# Check dataset structure
ls data/processed/train/images/ | wc -l  # Expected: 800
ls data/processed/val/images/ | wc -l    # Expected: 100
ls data/processed/test/images/ | wc -l   # Expected: 100

# Check data.yaml
cat ok-gesture-poc/config/data.yaml      # Expected: nc: 1, names: [ok_sign]

# Quick training test
cd ok-gesture-poc && python scripts/train.py --epochs 1 --model n
```

### Final Checklist
- [ ] All 11 tasks completed
- [ ] Final verification wave passed (F1, F2)
- [ ] Dataset ready for training (800 train, 100 val, 100 test)
- [ ] Single-class configuration (ok_sign only)
- [ ] Training pipeline integration verified
- [ ] Evidence files saved in .sisyphus/evidence/
