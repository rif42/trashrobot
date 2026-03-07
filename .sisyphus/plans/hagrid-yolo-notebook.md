# Fix HAGRID YOLO Training Notebook

## TL;DR

> **Objective**: Fix `train_local copy 2.ipynb` to work with local HAGRID data instead of downloading from GitHub (which fails with 404).
> 
> **Key Fixes**:
> - Load local annotations from `data/ann_train_val/ok.json` (27,999 entries)
> - Match UUIDs to existing images in `data/hagrid_30k/train_val_ok/` (1,753 files)
> - Filter bboxes to only "ok" class, use others as negative samples
> - Create 80/20 train/val split with fixed seed 42
> - Generate YOLO format labels and dataset.yaml
> - Include comprehensive verification cells
> 
> **Deliverable**: New notebook `train_local_FIXED.ipynb`
> 
> **Estimated Effort**: Short (single notebook, ~8 cells)
> **Parallel Execution**: NO - sequential notebook cells
> **Critical Path**: Cell 1 → Cell 8 (must run in order)

---

## Context

### Original Request
User has HAGRID dataset locally with:
- 1,753 images in `data/hagrid_30k/train_val_ok/` (UUID-named JPGs)
- 27,999 annotations in `data/ann_train_val/ok.json`

Current notebook tries to download "like" class from GitHub releases, fails with 404, and produces 0 images for training.

### Interview Summary
**Key Discussions**:
- Data structure analyzed: UUID filenames, normalized bbox format [x_center,y_center,width,height]
- Metis review identified gaps: stratification edge cases, negative samples, output handling

**Decisions Made**:
1. **Stratification**: Use simple random 80/20 split (not strict user-stratified) to keep all images
2. **Negative samples**: Include images with non-"ok" bboxes as negative samples (empty .txt files)
3. **Output file**: Create new `train_local_FIXED.ipynb` (preserve original)
4. **Random seed**: Fixed seed 42 for reproducibility
5. **Verification**: Include comprehensive automated tests

### Research Findings
- Annotation format: {"uuid": {"bboxes": [[x,y,w,h],...], "labels": ["ok",...], "user_id": "..."}}
- Bboxes already normalized 0-1, just need to write as YOLO format
- Only ~6% of annotated images exist locally (1,753 of 27,999)
- Some images have multiple bboxes with mixed labels ("ok" + "no_gesture")

### Metis Review
**Identified Gaps (addressed)**:
1. ✅ Images without "ok" bboxes → Decision: Use as negative samples
2. ✅ Single-image users → Decision: Use random split instead of stratification
3. ✅ Output handling → Decision: Create new notebook file
4. ✅ Verification criteria → Included in plan
5. ✅ Edge cases (malformed bboxes, missing files) → Handled in code

---

## Work Objectives

### Core Objective
Create a working Jupyter notebook that loads local HAGRID data, processes it for YOLO training, and verifies the output is ready for `model.train()`.

### Concrete Deliverables
- `train_local_FIXED.ipynb` - Complete revised notebook (8 cells)
- `data/processed/` directory with YOLO structure:
  - `images/train/` - Training images (80%)
  - `images/val/` - Validation images (20%)
  - `labels/train/` - YOLO format labels (one .txt per image)
  - `labels/val/` - YOLO format labels
  - `dataset.yaml` - Ultralytics dataset configuration
  - `exclusion_report.txt` - Log of excluded images and reasons

### Definition of Done
- [ ] Notebook runs end-to-end without errors
- [ ] All verification cells pass (automated QA)
- [ ] Ultralytics can successfully load the dataset
- [ ] File counts match expected (train/val split ~80/20)

### Must Have
- Load local ok.json annotations
- Match UUIDs to existing JPG files
- Filter to only "ok" class bboxes
- Create 80/20 train/val split with seed 42
- Generate YOLO format labels (class x_center y_center width height)
- Create dataset.yaml for Ultralytics
- Include verification cells for data integrity
- Create exclusion report
- Use pathlib for cross-platform compatibility

### Must NOT Have (Guardrails)
- NO downloading from GitHub/Kaggle
- NO data augmentation
- NO model training code beyond setup
- NO handling of classes beyond "ok"
- NO modification of original data files
- NO complex error recovery (fail fast with clear messages)

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (Jupyter + Python environment)
- **Automated tests**: YES - Verification cells in notebook (Cells 2, 7, 8)
- **Framework**: Python assertions + Ultralytics integration test
- **QA Method**: Agent-executed scenarios using notebook cells

### QA Policy
All verification is automated within notebook cells:
- Cell 2: Validate data loading (assertions on file existence, counts)
- Cell 7: Validate YOLO format (sample label file parsing)
- Cell 8: Full verification (file counts, split ratio, Ultralytics load test)

Evidence saved to notebook output cells (execution results).

---

## Execution Strategy

### Parallel Execution Waves

**This is a sequential notebook - cells must run in order.**

```
Sequential Execution (Notebook Cells):
├── Cell 1: Imports and Configuration [quick]
├── Cell 2: Load Annotations and Validate [quick]
├── Cell 3: Filter and Match Images [quick]
├── Cell 4: Create Train/Val Split [quick]
├── Cell 5: Convert to YOLO Format [quick]
├── Cell 6: Create Directory Structure [quick]
├── Cell 7: Generate dataset.yaml [quick]
└── Cell 8: Final Verification [quick]

Critical Path: Cell 1 → Cell 8 (all sequential)
Parallel Speedup: N/A - Notebook format requires sequential execution
Max Concurrent: 1 (single notebook file)
```

### Agent Dispatch Summary

- **1**: **1** - Complete notebook → `quick` agent

---

## TODOs

- [ ] 1. Create Cell 1: Imports and Configuration

  **What to do**:
  - Import required libraries (json, pathlib, shutil, yaml, random, PIL)
  - Define configuration variables (paths, seed, split ratio)
  - Create output directories
  - Print configuration summary

  **Must NOT do**:
  - Hardcode absolute Windows paths
  - Import training libraries (ultralytics) - not needed for data prep

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None needed
  
  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 2
  - **Blocked By**: None

  **References**:
  - Current notebook Cell 1-2 for import patterns
  - HAGRID docs for expected paths

  **Acceptance Criteria**:
  - [ ] All imports successful
  - [ ] Configuration variables defined and printed
  - [ ] Output directories created

  **QA Scenarios**:
  ```
  Scenario: Configuration loads correctly
    Tool: Bash (python)
    Preconditions: None
    Steps:
      1. Run Cell 1 in notebook
      2. Verify no import errors
      3. Check printed paths are correct
    Expected Result: All imports succeed, paths printed, directories exist
    Evidence: Notebook execution output cell
  ```

  **Commit**: YES
  - Message: `feat(notebook): Cell 1 - imports and configuration`

---

- [ ] 2. Create Cell 2: Load Annotations and Validate

  **What to do**:
  - Load `data/ann_train_val/ok.json`
  - Validate JSON structure (dict with UUID keys)
  - Count total annotations
  - Verify required fields exist (bboxes, labels, user_id)
  - Sample and verify bbox format (normalized 0-1)
  - Find all existing JPG images in `data/hagrid_30k/train_val_ok/`
  - Report statistics

  **Must NOT do**:
  - Modify original annotation file
  - Assume all annotated images exist (only ~6% do)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 3
  - **Blocked By**: Task 1

  **References**:
  - ok.json structure analyzed: {"uuid": {"bboxes": [...], "labels": [...], "user_id": "..."}}
  - 1,753 images exist vs 27,999 annotations

  **Acceptance Criteria**:
  - [ ] JSON loaded successfully
  - [ ] Statistics printed (total annos, unique users, sample bbox)
  - [ ] Image directory scanned
  - [ ] Validation assertions pass

  **QA Scenarios**:
  ```
  Scenario: Data loads and validates
    Tool: Bash (python)
    Preconditions: Cell 1 completed
    Steps:
      1. Run Cell 2
      2. Check output shows: "Total annotations: 27999"
      3. Check output shows: "Found images: ~1753"
      4. Verify sample bbox values are 0-1 normalized
    Expected Result: All validations pass, stats printed
    Evidence: Notebook output cell with statistics
  ```

  **Commit**: YES
  - Message: `feat(notebook): Cell 2 - load and validate annotations`

---

- [ ] 3. Create Cell 3: Filter and Match Images

  **What to do**:
  - Match annotation UUIDs to existing JPG files (case-insensitive)
  - For each matched image:
    - Extract bboxes where label == "ok"
    - Track images with only non-"ok" labels (for negative samples)
  - Build data structure: {uuid: {"bboxes": [...], "user_id": "...", "is_negative": bool}}
  - Log excluded images (orphaned JPGs, orphaned annotations)
  - Print summary statistics

  **Must NOT do**:
  - Copy/move original images
  - Filter out negative samples (keep them with empty label list)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 4
  - **Blocked By**: Task 2

  **References**:
  - Decision: Keep negative samples (images with no "ok" bboxes)
  - 6,530 images have multiple labels ("ok" + "no_gesture")

  **Acceptance Criteria**:
  - [ ] Matched images count printed (~1,753)
  - [ ] Positive samples count (images with "ok" bboxes)
  - [ ] Negative samples count (images without "ok" bboxes)
  - [ ] Excluded counts logged

  **QA Scenarios**:
  ```
  Scenario: Image matching works correctly
    Tool: Bash (python)
    Preconditions: Cell 2 completed
    Steps:
      1. Run Cell 3
      2. Verify matched_images length matches JPG count
      3. Check positive + negative + excluded = total annotations
      4. Verify no UUID mismatches
    Expected Result: All images categorized correctly
    Evidence: Notebook output showing categories
  ```

  **Commit**: YES
  - Message: `feat(notebook): Cell 3 - filter and match images`

---

- [ ] 4. Create Cell 4: Create Train/Val Split

  **What to do**:
  - Set random seed 42 for reproducibility
  - Shuffle all matched image UUIDs
  - Split 80/20 into train/val lists
  - Print split statistics
  - Save split info to JSON for reference

  **Must NOT do**:
  - Use stratified split (user requested simple random)
  - Exclude any images (keep all 1,753)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 5
  - **Blocked By**: Task 3

  **References**:
  - Decision: Random 80/20 split with seed 42
  - ~1,400 train, ~350 val expected

  **Acceptance Criteria**:
  - [ ] Split created with seed 42
  - [ ] Train count printed (~80%)
  - [ ] Val count printed (~20%)
  - [ ] Split info saved to JSON

  **QA Scenarios**:
  ```
  Scenario: Split is correct
    Tool: Bash (python)
    Preconditions: Cell 3 completed
    Steps:
      1. Run Cell 4
      2. Verify train + val = total matched images
      3. Check ratio is ~80/20 (between 75-85%)
      4. Verify split reproducible (same seed)
    Expected Result: Split created, stats printed
    Evidence: Notebook output with counts
  ```

  **Commit**: YES
  - Message: `feat(notebook): Cell 4 - train/val split`

---

- [ ] 5. Create Cell 5: Convert to YOLO Format

  **What to do**:
  - For each image in train and val:
    - Copy image to output images/{split}/ directory
    - Create label file labels/{split}/{uuid}.txt
    - Write YOLO format: "0 {x_center} {y_center} {width} {height}"
    - Handle multiple bboxes (one line per bbox)
    - Handle negative samples (create empty .txt file)
  - Print progress and statistics

  **Must NOT do**:
  - Move original images (copy only)
  - Modify bbox values (already normalized)
  - Skip negative samples (create empty label files)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 6
  - **Blocked By**: Task 4

  **References**:
  - YOLO format: class_id x_center y_center width height (all 0-1)
  - Class 0 = "ok"
  - Negative samples = empty .txt file

  **Acceptance Criteria**:
  - [ ] All images copied to output directories
  - [ ] All label files created
  - [ ] YOLO format verified (5 values per line)
  - [ ] Negative samples have empty files

  **QA Scenarios**:
  ```
  Scenario: YOLO labels created correctly
    Tool: Bash (python)
    Preconditions: Cell 4 completed
    Steps:
      1. Run Cell 5
      2. Check train/images/ and val/images/ have JPGs
      3. Check train/labels/ and val/labels/ have TXTs
      4. Sample random label file and parse
    Expected Result: All files created, format correct
    Evidence: File system listing + sample label content
  ```

  **Commit**: YES
  - Message: `feat(notebook): Cell 5 - YOLO format conversion`

---

- [ ] 6. Create Cell 6: Create Directory Structure

  **What to do**:
  - Verify output directory structure:
    - data/processed/images/train/
    - data/processed/images/val/
    - data/processed/labels/train/
    - data/processed/labels/val/
  - Print directory tree
  - Verify file counts match

  **Must NOT do**:
  - Create unnecessary subdirectories
  - Change structure from YOLO standard

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 7
  - **Blocked By**: Task 5

  **References**:
  - YOLO standard structure: images/ and labels/ with train/val subdirs

  **Acceptance Criteria**:
  - [ ] All directories exist
  - [ ] File counts match expected
  - [ ] Directory tree printed

  **QA Scenarios**:
  ```
  Scenario: Directory structure is correct
    Tool: Bash (ls/tree)
    Preconditions: Cell 5 completed
    Steps:
      1. Run Cell 6
      2. Verify 4 subdirectories exist
      3. Check images count = labels count in each split
    Expected Result: Structure validated, counts match
    Evidence: Directory tree output
  ```

  **Commit**: YES
  - Message: `feat(notebook): Cell 6 - directory structure`

---

- [ ] 7. Create Cell 7: Generate dataset.yaml

  **What to do**:
  - Create dataset.yaml with:
    - path: relative path to processed directory
    - train: images/train
    - val: images/val
    - nc: 1
    - names: ['ok']
  - Use relative paths (not absolute) for portability
  - Save to data/processed/dataset.yaml
  - Print yaml content for verification

  **Must NOT do**:
  - Use absolute Windows paths (breaks on other systems)
  - Include test split (not needed)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 8
  - **Blocked By**: Task 6

  **References**:
  - Ultralytics dataset.yaml format
  - Relative paths for cross-platform compatibility

  **Acceptance Criteria**:
  - [ ] dataset.yaml created
  - [ ] YAML structure valid
  - [ ] Paths are relative
  - [ ] Content printed

  **QA Scenarios**:
  ```
  Scenario: YAML is valid
    Tool: Bash (python)
    Preconditions: Cell 6 completed
    Steps:
      1. Run Cell 7
      2. Load YAML with yaml.safe_load()
      3. Verify required keys exist
      4. Check paths are relative
    Expected Result: YAML valid, paths correct
    Evidence: YAML content output
  ```

  **Commit**: YES
  - Message: `feat(notebook): Cell 7 - dataset.yaml generation`

---

- [ ] 8. Create Cell 8: Final Verification

  **What to do**:
  - Comprehensive verification:
    - File counts: train_images == train_labels
    - Split ratio: train / total ≈ 0.8
    - Label format: sample files have correct structure
    - Negative samples: empty .txt files exist
    - Ultralytics integration test: try loading dataset
  - Create exclusion_report.txt with:
    - Orphaned annotations count
    - Orphaned images count
    - Filtered bboxes count
  - Print final success message

  **Must NOT do**:
  - Skip any verification step
  - Hide errors (fail fast with clear messages)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None
  - **Blocked By**: Task 7

  **References**:
  - Metis acceptance criteria
  - Ultralytics YOLO dataset loading

  **Acceptance Criteria**:
  - [ ] All file counts match
  - [ ] Split ratio ~80/20
  - [ ] Label format validated
  - [ ] Ultralytics can load dataset
  - [ ] Exclusion report created
  - [ ] All assertions pass

  **QA Scenarios**:
  ```
  Scenario: Full verification passes
    Tool: Bash (python)
    Preconditions: All previous cells completed
    Steps:
      1. Run Cell 8
      2. Verify all assertions pass
      3. Check exclusion_report.txt exists
      4. Confirm Ultralytics loads dataset without error
    Expected Result: All checks pass, "✅ Dataset ready for training"
    Evidence: Verification output + exclusion report
  ```

  **Commit**: YES
  - Message: `feat(notebook): Cell 8 - final verification`

---

## Final Verification Wave (MANDATORY)

> Run the complete notebook end-to-end and verify all outputs

- [ ] F1. **End-to-End Notebook Execution** — `quick`
  Run all 8 cells in order in a fresh kernel. Verify no errors, all outputs as expected. Check exclusion_report.txt created.
  
  **Agent-Executed QA**:
  ```
  Scenario: Complete notebook execution
    Tool: Bash (jupyter nbconvert --execute)
    Steps:
      1. Execute notebook: jupyter nbconvert --to notebook --execute train_local_FIXED.ipynb
      2. Verify exit code 0 (no errors)
      3. Check output notebook has 8 execution outputs
      4. Verify exclusion_report.txt exists
    Expected Result: Notebook executes successfully
    Evidence: Executed notebook file + report
  ```

- [ ] F2. **YOLO Dataset Load Test** — `quick`
  Test that Ultralytics can actually load the dataset.
  
  **Agent-Executed QA**:
  ```
  Scenario: Ultralytics integration
    Tool: Bash (python)
    Steps:
      1. python -c "from ultralytics import YOLO; model = YOLO('yolov8n.yaml'); model.train(data='data/processed/dataset.yaml', epochs=1, imgsz=640)"
      2. Verify training starts without dataset errors
    Expected Result: YOLO loads dataset successfully
    Evidence: Training starts, shows batch loading
  ```

- [ ] F3. **File Structure Validation** — `quick`
  Verify output structure matches YOLO requirements.
  
  **Agent-Executed QA**:
  ```
  Scenario: Directory structure check
    Tool: Bash (ls/find)
    Steps:
      1. tree data/processed/ or ls -R
      2. Verify: images/train/, images/val/, labels/train/, labels/val/, dataset.yaml
      3. Check counts: find images/train -name '*.jpg' | wc -l
      4. Verify images count == labels count in each split
    Expected Result: Structure matches YOLO spec
    Evidence: Directory listing output
  ```

- [ ] F4. **Label Format Verification** — `quick`
  Sample random label files and verify format.
  
  **Agent-Executed QA**:
  ```
  Scenario: Label format check
    Tool: Bash (python)
    Steps:
      1. Sample 5 random .txt files from labels/train/
      2. Parse each: should have 5 values (class x y w h)
      3. Verify class = 0, all values 0-1
      4. Check multi-bbox files have multiple lines
    Expected Result: All labels valid YOLO format
    Evidence: Sample label contents
  ```

---

## Commit Strategy

- **1**: `feat(notebook): Complete HAGRID YOLO data preparation notebook`
  - All 8 cells
  - train_local_FIXED.ipynb
  - Pre-commit: Run notebook end-to-end, verify no errors

---

## Success Criteria

### Verification Commands
```bash
# 1. Notebook executes without errors
jupyter nbconvert --to notebook --execute train_local_FIXED.ipynb

# 2. YOLO can load dataset
python -c "from ultralytics import YOLO; YOLO('yolov8n.yaml').train(data='data/processed/dataset.yaml', epochs=1)"

# 3. File counts match
train_img=$(find data/processed/images/train -name '*.jpg' | wc -l)
train_lbl=$(find data/processed/labels/train -name '*.txt' | wc -l)
echo "Train: $train_img images, $train_lbl labels"
[ "$train_img" -eq "$train_lbl" ] && echo "✓ Match" || echo "✗ Mismatch"

# 4. Split ratio
val_img=$(find data/processed/images/val -name '*.jpg' | wc -l)
total=$((train_img + val_img))
ratio=$(echo "scale=2; $train_img / $total" | bc)
echo "Split ratio: $ratio (should be ~0.80)"
```

### Final Checklist
- [ ] All 8 notebook cells execute successfully
- [ ] data/processed/ has correct structure
- [ ] Train/val split is ~80/20
- [ ] All images have corresponding labels
- [ ] YOLO format is correct (verified by sampling)
- [ ] dataset.yaml is valid
- [ ] Ultralytics can load dataset
- [ ] exclusion_report.txt created
- [ ] Original data files untouched
- [ ] New notebook file created (not overwritten)
