# Local Training Adaptation Plan

## TL;DR

> **Objective**: Adapt Google Colab notebook (`hagrid_thumbsup_training.ipynb`) to run on local computer with optional GPU detection
> 
003e **Deliverables**:
003e - `train_local.py` - Standalone training script for local execution
003e - Automatic GPU/CPU detection
003e - Same functionality as Colab version but without Google Drive integration
003e 
003e **Estimated Effort**: Quick (~45 min)
003e **Parallel Execution**: NO
003e **Critical Path**: Task 1 (script creation) → Task 2 (testing)

---

## Context

### Original Request
User wants to run the HaGRID thumbs up training notebook on their local computer instead of Google Colab. The notebook currently:
- Uses `!nvidia-smi` (Colab shell commands)
- Mounts Google Drive for storage
- Assumes GPU is always available
- Uses Colab-specific paths (`/content/`)

### Requirements
- Remove Colab-specific code
- Add automatic GPU detection (use CUDA if available, fallback to CPU)
- Make it a runnable Python script (not notebook)
- Keep all training functionality
- Local file paths instead of Google Drive

---

## Work Objectives

### Core Objective
Create a standalone Python script `train_local.py` that replicates the Colab training pipeline but runs on local machines with automatic GPU/CPU detection.

### Concrete Deliverables
- `train_local.py` - Complete training script
- GPU detection with graceful CPU fallback
- Local directory structure (no Google Drive)
- Command-line interface for configuration
- Same training hyperparameters as Colab version

### Must Have
- GPU availability check using PyTorch (`torch.cuda.is_available()`)
- Same dataset generation (synthetic test data)
- Same training configuration (epochs, batch size, optimizer)
- Model download (YOLO11n or YOLOv8n)
- Training loop with progress output
- Model evaluation post-training
- Save best model locally

### Must NOT Have
- No Google Drive mounting
- No `!` shell commands (Colab-specific)
- No hardcoded Colab paths
- No manual GPU runtime selection

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Single Task):
└── Task 1: Create train_local.py with all functionality [quick]

Wave 2 (Verification):
└── Task 2: Test syntax and verify imports [quick]
```

---

## TODOs

- [ ] 1. Create Local Training Script

  **What to do**:
  - Create `train_local.py` in project root
  - Implement GPU detection function
  - Implement directory setup (local paths)
  - Port dataset generation from notebook
  - Port training configuration
  - Add CLI argument parsing
  - Implement training loop
  - Add evaluation and benchmarking
  - Add results saving

  **Must NOT do**:
  - No `!` shell commands
  - No Google Colab imports (`from google.colab import drive`)
  - No hardcoded `/content/` paths
  - No assumption that GPU is always available

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **References**:
  - `hagrid_thumbsup_training.ipynb` - Source material to port
  - `output/inference_simple.py` - Pattern for CLI structure
  - Ultralytics docs for YOLO training API

  **Acceptance Criteria**:
  - [ ] Script runs without syntax errors: `python -m py_compile train_local.py`
  - [ ] GPU detection works and prints correct device
  - [ ] All imports resolve (ultralytics, torch, etc.)
  - [ ] CLI help works: `python train_local.py --help`
  - [ ] Creates expected directory structure
  - [ ] Can generate test data: `python train_local.py --test-data-only`

  **QA Scenarios**:
  ```
  Scenario: Script syntax is valid
    Tool: Bash
    Steps:
      1. python -m py_compile train_local.py
    Expected: No errors
    Evidence: .sisyphus/evidence/train_local_syntax.txt
  
  Scenario: GPU detection works
    Tool: Bash
    Steps:
      1. python -c "import torch; print('CUDA:', torch.cuda.is_available())"
    Expected: Shows True/False correctly
    Evidence: .sisyphus/evidence/gpu_check.txt
  
  Scenario: CLI help displays
    Tool: Bash
    Steps:
      1. python train_local.py --help
    Expected: Shows argument descriptions
    Evidence: .sisyphus/evidence/train_local_help.txt
  ```

  **Commit**: YES
  - Message: `feat(training): add local training script with GPU detection`
  - Files: `train_local.py`
  - Pre-commit: `python -m py_compile train_local.py`

---

## Success Criteria

### Verification Commands
```bash
# Test syntax
python -m py_compile train_local.py

# Test help
python train_local.py --help

# Test data generation
python train_local.py --work-dir ./test-work --test-data-only

# Check created directories
ls -la ./test-work/
```

### Final Checklist
- [ ] Script created and syntactically valid
- [ ] GPU detection implemented
- [ ] CPU fallback works
- [ ] CLI arguments functional
- [ ] Directory creation works
- [ ] Test data generation works
