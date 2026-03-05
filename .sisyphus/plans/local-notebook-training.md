# Local Jupyter Notebook Training Plan

## TL;DR

> **Objective**: Create local Jupyter notebook (`train_local.ipynb`) adapted from Colab version with 500-image dataset and local GPU detection
> 
> **Deliverables**:
> - `train_local.ipynb` - Jupyter notebook for local training
> - 500 synthetic training images (400 train + 100 val)
> - Automatic GPU/CPU detection for local machines
> - No Colab-specific code (Google Drive, shell commands)
> 
> **Estimated Effort**: Quick (~45 min)
> **Parallel Execution**: NO
> **Critical Path**: Task 1 (notebook creation) → Task 2 (validation)

---

## Context

### Original Request
User wants to adapt the Colab notebook (`hagrid_thumbsup_training.ipynb`) to:
1. **Keep .ipynb format** (Jupyter notebook, not Python script)
2. **Run locally** on their own computer (not Google Colab)
3. **Generate 500 images** instead of 120 (more training data)
4. **Auto-detect GPU** availability on local machine

### Changes from Colab Version
- ❌ Remove: `!nvidia-smi` shell commands
- ❌ Remove: Google Drive mounting (`from google.colab import drive`)
- ❌ Remove: `/content/` paths
- ✅ Add: Local GPU detection with `torch.cuda.is_available()`
- ✅ Add: Local file paths (`./data/`, `./work/`)
- ✅ Add: 500 images (400 train + 100 val instead of 100+20)
- ✅ Keep: All training logic, hyperparameters, visualization

---

## Work Objectives

### Core Objective
Create `train_local.ipynb` - a Jupyter notebook that runs the complete YOLO training pipeline locally with 500 synthetic images and automatic GPU detection.

### Concrete Deliverables
- `train_local.ipynb` with all cells ported from Colab
- Cell 1: Local environment setup + GPU detection
- Cell 2-3: Dataset generation (500 images: 400 train + 100 val)
- Cell 4: Dataset verification
- Cell 5-7: Configuration (dataset.yaml)
- Cell 8: Model download/load
- Cell 9: Training configuration
- Cell 10-11: Training execution
- Cell 12-15: Evaluation
- Cell 16: Inference script generation
- Cell 17: Packaging
- Cell 18: Final report

### Must Have
- Jupyter notebook format (.ipynb)
- Local GPU detection and device selection
- 500 synthetic images (400 train, 100 val)
- All original training hyperparameters preserved
- Local directory structure (./work/, ./data/)
- Progress bars and visualization outputs
- Model checkpointing to local disk

### Must NOT Have
- No Colab-specific imports or magic commands
- No `!` shell commands
- No Google Drive integration
- No hardcoded `/content/` paths

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Single Task):
└── Task 1: Create train_local.ipynb [quick]

Wave 2 (Verification):
└── Task 2: Validate notebook structure [quick]
```

---

## TODOs

- [ ] 1. Create Local Jupyter Notebook

  **What to do**:
  - Create `train_local.ipynb` in project root
  - Port all cells from `hagrid_thumbsup_training.ipynb`
  - Replace Colab-specific code:
    - `!nvidia-smi` → `torch.cuda.is_available()` check with print
    - Google Drive mount → local directory creation
    - `/content/` paths → `./work/` relative paths
  - Increase dataset size: 400 train + 100 val = 500 total (was 100+20)
  - Add local GPU detection cell at start
  - Ensure all cells are executable independently
  - Add markdown explanations for each section
  - Update all file paths to use `./` or `Path.home()`

  **Must NOT do**:
  - No `!` shell command cells
  - No `from google.colab import drive`
  - No `/content/` absolute paths
  - No Colab-specific metadata

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **References**:
  - `hagrid_thumbsup_training.ipynb` - Source notebook to adapt
  - `train_local.py` (if exists) - For logic patterns
  - Jupyter notebook format specification

  **Acceptance Criteria**:
  - [ ] Notebook file created: `train_local.ipynb`
  - [ ] Valid JSON structure (can be opened in Jupyter)
  - [ ] Cell count: ~18-20 cells matching original structure
  - [ ] First cell: GPU detection with `torch.cuda.is_available()`
  - [ ] Dataset generation cell: creates 400 train + 100 val images
  - [ ] No Colab imports present
  - [ ] No `!` shell commands in code cells
  - [ ] All paths use relative `./` or `Path()` objects

  **QA Scenarios**:
  ```
  Scenario: Notebook structure is valid
    Tool: Bash
    Steps:
      1. python -c "import json; json.load(open('train_local.ipynb'))"
    Expected: No JSON decode errors
    Evidence: .sisyphus/evidence/notebook_json_valid.txt
  
  Scenario: No Colab-specific code present
    Tool: Bash
    Steps:
      1. grep -n "google.colab" train_local.ipynb || echo "No Colab refs found"
      2. grep -n '"!"' train_local.ipynb | head -5 || echo "No shell commands"
    Expected: No matches for Colab imports or shell commands
    Evidence: .sisyphus/evidence/no_colab_refs.txt
  
  Scenario: Dataset size updated to 500
    Tool: Grep
    Steps:
      1. grep -n "400.*train\|100.*val\|n_train.*400\|n_val.*100" train_local.ipynb
    Expected: Shows 400 train and 100 val configuration
    Evidence: .sisyphus/evidence/dataset_size_500.txt
  
  Scenario: GPU detection cell present
    Tool: Grep
    Steps:
      1. grep -n "cuda.is_available\|get_device_name" train_local.ipynb
    Expected: Shows GPU detection code
    Evidence: .sisyphus/evidence/gpu_detection.txt
  ```

  **Commit**: YES
  - Message: `feat(training): add local jupyter notebook with 500 images and gpu detection`
  - Files: `train_local.ipynb`
  - Pre-commit: Verify JSON structure

---

## Data Size Justification

### Original Colab Version
- 100 train + 20 val = 120 total images
- Very small for actual training (testing only)

### New Local Version
- 400 train + 100 val = 500 total images
- Better for actual model training
- Still synthetic/quick to generate
- Can be replaced with real HaGRID data later

### Trade-offs
- Generation time: ~2-3 seconds for 500 images (still fast)
- Disk space: ~15-20 MB for 500 images
- Training time: ~10-20% longer than 120 images

---

## Success Criteria

### Verification Commands
```bash
# Validate JSON structure
python -c "import json; nb = json.load(open('train_local.ipynb')); print(f'Cells: {len(nb[\"cells\"])}')"

# Check for Colab references
grep -c "google.colab" train_local.ipynb || echo "0 Colab refs"

# Check for shell commands
grep -c '"!"' train_local.ipynb || echo "0 shell commands"

# Verify dataset size configuration
grep -n "n_train\|n_val" train_local.ipynb | head -5
```

### Final Checklist
- [ ] Notebook created with .ipynb extension
- [ ] Valid Jupyter notebook JSON structure
- [ ] ~18-20 cells matching original flow
- [ ] GPU detection in first code cell
- [ ] 400 train + 100 val images (500 total)
- [ ] No Colab imports
- [ ] No shell commands (`!`)
- [ ] Local relative paths only
- [ ] Can be opened in Jupyter Lab/Notebook
