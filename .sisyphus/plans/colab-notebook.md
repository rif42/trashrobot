# Google Colab YOLO Training Notebook

## TL;DR

> **Quick Summary**: Create a new Jupyter notebook (`train_yolo_colab.ipynb`) for Google Colab that trains YOLO on HAGRID 'ok' gesture dataset using Google Drive for data storage and persistence.
>
> **Deliverables**: 
> - `train_yolo_colab.ipynb` - Complete Colab notebook with Drive integration
>
> **Estimated Effort**: Quick (< 30 minutes)
> **Parallel Execution**: NO - single sequential task
> **Critical Path**: Task 1 → Complete

---

## Context

### Original Request
User wants to create a Google Colab notebook for training YOLO on their local HAGRID 'ok' gesture dataset. They will upload the `data/processed/` folder to Google Drive and want the notebook to work seamlessly in Colab.

### Existing Assets
- `train_yolo.ipynb` - Local training notebook with 6 cells:
  1. Install and Import (ultralytics, torch)
  2. Load Model (YOLOv8n)
  3. Train (hardcoded Windows paths)
  4. View Results
  5. Test Inference
  6. Export Model (ONNX)

- `data/processed/` - Dataset folder with:
  - `dataset.yaml` - Uses absolute Windows paths (needs adaptation)
  - `images/train/` - ~1680 training images
  - `images/val/` - ~771 validation images
  - `labels/train/` - Training labels (YOLO format)
  - `labels/val/` - Validation labels

### Colab Requirements
1. Mount Google Drive
2. Configure paths for Drive structure
3. Generate dataset.yaml with correct paths
4. Leverage GPU acceleration (Colab provides T4)
5. Save results back to Drive for persistence
6. Handle runtime disconnections with resume capability
7. Validate data existence before training

---

## Work Objectives

### Core Objective
Create a complete, ready-to-use Google Colab notebook that can train YOLO on the HAGRIG 'ok' gesture dataset stored in Google Drive.

### Concrete Deliverables
- `train_yolo_colab.ipynb` - Full notebook with:
  - Google Drive mounting
  - Configurable data path (user sets their Drive folder)
  - Automatic dataset.yaml generation with correct paths
  - GPU detection and usage
  - Data validation before training
  - Training with proper hyperparameters
  - Results saved back to Drive
  - Model export to ONNX

### Definition of Done
- [ ] Notebook can be opened in Google Colab
- [ ] Mounts Drive and validates data exists
- [ ] Generates correct dataset.yaml
- [ ] Uses GPU if available
- [ ] Completes training (or has resume capability)
- [ ] Saves results to Drive
- [ ] Can run inference and export

### Must Have
- Google Drive mounting cell
- Configurable BASE_PATH variable
- Dynamic dataset.yaml generation
- GPU detection and info display
- Data validation (check images/labels exist)
- Training with epochs=100 (same as local)
- Results persistence to Drive
- Test inference cell
- ONNX export cell

### Must NOT Have (Guardrails)
- No hardcoded Windows paths
- No dependency on local file system
- No manual dataset.yaml editing required
- No results lost on disconnect

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES - Google Colab environment
- **Automated tests**: NO - manual validation in Colab
- **Validation method**: Open notebook in Colab and verify cells run

### QA Policy
- Agent will create notebook with all required cells
- Notebook will be validated by checking JSON structure
- Each cell will be complete and runnable

---

## Execution Strategy

### Parallel Execution Waves

This is a single sequential task - creating one notebook file.

```
Wave 1 (Single Task - Create Notebook):
└── Task 1: Create train_yolo_colab.ipynb
    ├── Create notebook structure (JSON format)
    ├── Cell 1: Mount Google Drive
    ├── Cell 2: Configure paths & generate dataset.yaml
    ├── Cell 3: Install packages & verify GPU
    ├── Cell 4: Load model
    ├── Cell 5: Train model (with resume support)
    ├── Cell 6: Copy results to Drive
    ├── Cell 7: Test inference
    └── Cell 8: Export to ONNX

Wave FINAL (Validation):
└── Validate notebook JSON structure
```

---

## TODOs

- [ ] 1. Create Google Colab YOLO Training Notebook

  **What to do**:
  Create `train_yolo_colab.ipynb` with the following cells:
  
  **Cell 1: Mount Google Drive**
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  print("✅ Drive mounted at /content/drive")
  ```
  
  **Cell 2: Configure Paths & Generate Dataset YAML**
  ```python
  import os
  import yaml
  from pathlib import Path
  
  # === CONFIGURE THIS ===
  # Set the path to your data folder in Drive
  # Example: If you uploaded data/processed/ to MyDrive/ML_Training/
  BASE_PATH = "/content/drive/MyDrive/ML_Training/processed"
  # =====================
  
  # Verify path exists
  if not os.path.exists(BASE_PATH):
      raise FileNotFoundError(f"❌ Path not found: {BASE_PATH}\nPlease update BASE_PATH to your data folder")
  
  # Check for required subdirectories
  required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
  for dir_name in required_dirs:
      dir_path = Path(BASE_PATH) / dir_name
      if not dir_path.exists():
          raise FileNotFoundError(f"❌ Missing directory: {dir_path}")
      file_count = len(list(dir_path.glob('*')))
      print(f"✅ {dir_name}: {file_count} files")
  
  # Generate dataset.yaml with correct paths
  dataset_config = {
      'path': BASE_PATH,
      'train': str(Path(BASE_PATH) / 'images' / 'train'),
      'val': str(Path(BASE_PATH) / 'images' / 'val'),
      'nc': 1,
      'names': ['ok']
  }
  
  yaml_path = Path(BASE_PATH) / 'dataset.yaml'
  with open(yaml_path, 'w') as f:
      yaml.dump(dataset_config, f, default_flow_style=False)
  
  print(f"\n✅ Created dataset.yaml at: {yaml_path}")
  print(f"📊 Dataset config:\n{yaml.dump(dataset_config, default_flow_style=False)}")
  ```
  
  **Cell 3: Install Packages & Verify GPU**
  ```python
  import subprocess
  import sys
  
  print("Installing ultralytics...")
  subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"])
  
  from ultralytics import YOLO
  import torch
  
  print(f"\n{'='*50}")
  print("System Info:")
  print(f"{'='*50}")
  print(f"✅ PyTorch: {torch.__version__}")
  print(f"✅ CUDA Available: {torch.cuda.is_available()}")
  
  if torch.cuda.is_available():
      print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
      print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
      device = '0'
  else:
      print("⚠️  Using CPU (slower)")
      device = 'cpu'
  ```
  
  **Cell 4: Load Model**
  ```python
  print("\nLoading YOLO model...")
  
  # Choose model size: n (nano), s (small), m (medium), l (large), x (xlarge)
  # n = fastest, x = most accurate
  model = YOLO('yolov8n.pt')
  
  print(f"✅ Model loaded: yolov8n")
  print(f"   Task: {model.task}")
  print(f"   Classes: {model.names}")
  ```
  
  **Cell 5: Train Model**
  ```python
  from pathlib import Path
  
  print("\nStarting training...")
  print("💡 Tip: If Colab disconnects, re-run all cells to resume from last checkpoint")
  print("⏱️  This will take 5-10 minutes on GPU\n")
  
  # Set up paths
  yaml_path = Path(BASE_PATH) / 'dataset.yaml'
  results_dir = Path(BASE_PATH) / 'training_results'
  
  # Training configuration
  results = model.train(
      data=str(yaml_path),
      epochs=100,
      imgsz=640,
      batch=16,
      device=device,
      patience=10,
      save=True,
      project=str(results_dir),
      name='train',
      exist_ok=True,
      resume=True  # Allows resuming if interrupted
  )
  
  print("\n✅ Training complete!")
  ```
  
  **Cell 6: View & Copy Results**
  ```python
  from pathlib import Path
  import shutil
  
  print("\nTraining Results:")
  print("="*50)
  
  # Get training directory
  train_dir = Path(BASE_PATH) / 'training_results' / 'train'
  
  if train_dir.exists():
      print(f"\n📁 Results saved to: {train_dir}")
      
      # List files
      weights_dir = train_dir / 'weights'
      if weights_dir.exists():
          print(f"\n💾 Model weights:")
          for w in weights_dir.glob('*.pt'):
              size_mb = w.stat().st_size / (1024*1024)
              print(f"   - {w.name} ({size_mb:.1f} MB)")
      
      # Show results image if exists
      results_img = train_dir / 'results.png'
      if results_img.exists():
          print(f"\n📊 Results plot found")
          from IPython.display import Image, display
          display(Image(filename=str(results_img)))
          
      # Copy final model to root of data folder for easy access
      best_model = weights_dir / 'best.pt'
      if best_model.exists():
          dest = Path(BASE_PATH) / 'best_model.pt'
          shutil.copy(best_model, dest)
          print(f"\n✅ Best model copied to: {dest}")
  else:
      print("❌ Training directory not found")
  ```
  
  **Cell 7: Test Inference**
  ```python
  print("\nTesting trained model...")
  
  # Load best model
  best_model_path = Path(BASE_PATH) / 'training_results' / 'train' / 'weights' / 'best.pt'
  
  if not best_model_path.exists():
      print("❌ Best model not found. Did training complete?")
  else:
      best_model = YOLO(str(best_model_path))
      
      # Test on validation image
      import random
      
      val_dir = Path(BASE_PATH) / 'images' / 'val'
      val_images = list(val_dir.glob('*.jpg')) + list(val_dir.glob('*.png'))
      
      if val_images:
          test_img = random.choice(val_images)
          print(f"\n🖼️  Testing on: {test_img.name}")
          
          # Run inference
          results = best_model(str(test_img))
          
          # Display results
          results[0].show()
          
          # Print detections
          boxes = results[0].boxes
          if boxes:
              print(f"\n✅ Detected {len(boxes)} 'ok' gesture(s)")
              for i, box in enumerate(boxes):
                  conf = float(box.conf)
                  print(f"   Detection {i+1}: confidence={conf:.2%}")
          else:
              print(f"\n⚠️  No 'ok' gesture detected in this image")
      else:
          print("❌ No validation images found")
  ```
  
  **Cell 8: Export Model**
  ```python
  print("\nExporting model...")
  
  # Load best model
  best_model_path = Path(BASE_PATH) / 'training_results' / 'train' / 'weights' / 'best.pt'
  
  if not best_model_path.exists():
      print("❌ Best model not found. Did training complete?")
  else:
      best_model = YOLO(str(best_model_path))
      
      # Export to ONNX format (for deployment)
      best_model.export(format='onnx')
      
      # Copy ONNX to data folder root
      onnx_path = Path(BASE_PATH) / 'training_results' / 'train' / 'weights' / 'best.onnx'
      if onnx_path.exists():
          dest = Path(BASE_PATH) / 'best_model.onnx'
          shutil.copy(onnx_path, dest)
          print(f"\n✅ Model exported to ONNX")
          print(f"   Location: {dest}")
          print(f"   Size: {dest.stat().st_size / (1024*1024):.1f} MB")
  ```
  
  **Usage Instructions** (as markdown cell at top):
  ```
  # Train YOLO on HAGRID 'ok' Gesture Dataset (Google Colab)
  
  ## Setup Instructions:
  1. Upload your `data/processed/` folder to Google Drive (e.g., to `MyDrive/ML_Training/`)
  2. Open this notebook in Google Colab
  3. Update `BASE_PATH` in Cell 2 to point to your data folder
  4. Run all cells (Runtime → Run all)
  5. Training will save results back to your Drive
  
  ## Features:
  - ✅ GPU acceleration (if available)
  - ✅ Resume training after disconnections
  - ✅ Results saved to Google Drive
  - ✅ Model export to ONNX
  ```

  **Must NOT do**:
  - Do NOT use hardcoded Windows paths
  - Do NOT depend on local file system
  - Do NOT require manual dataset.yaml editing
  - Do NOT use absolute paths without BASE_PATH configuration

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file creation, straightforward JSON formatting task
  - **Skills**: []
    - No special skills needed - just Python code writing

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None
  - **Blocked By**: None (can start immediately)

  **References** (CRITICAL):
  - `train_yolo.ipynb` - Reference for original cell structure and code patterns
  - `data/processed/dataset.yaml` - Reference for dataset configuration format
  - Google Colab docs: https://colab.research.google.com/notebooks/io.ipynb
  - Ultralytics YOLO docs: https://docs.ultralytics.com/modes/train/

  **Acceptance Criteria**:
  - [ ] File `train_yolo_colab.ipynb` exists and is valid JSON
  - [ ] Notebook has 9 cells (1 markdown + 8 code cells)
  - [ ] Each cell contains complete, runnable code
  - [ ] No hardcoded local paths
  - [ ] BASE_PATH is configurable
  - [ ] Includes data validation checks
  - [ ] Includes GPU detection
  - [ ] Includes resume capability
  - [ ] Results save to Drive

  **QA Scenarios**:
  ```
  Scenario: Validate notebook structure
    Tool: Read
    Steps:
      1. Read train_yolo_colab.ipynb
      2. Verify it's valid JSON
      3. Count cells (should be 9)
      4. Check each cell has source content
    Expected Result: Valid notebook with 9 cells
    Evidence: Screenshot of file structure
  ```

  **Commit**: NO (no commits needed - single file creation)

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Verify notebook exists, has correct structure, and follows all requirements.
  Output: `VERDICT: APPROVE/REJECT`

---

## Commit Strategy

- NO commits needed - this creates a single standalone notebook file

---

## Success Criteria

### Verification Commands
```bash
# Check notebook exists and is valid JSON
python -c "import json; json.load(open('train_yolo_colab.ipynb')); print('✅ Valid notebook')"

# Count cells
python -c "import json; nb=json.load(open('train_yolo_colab.ipynb')); print(f'Cells: {len(nb[\"cells\"])}')"
```

### Final Checklist
- [ ] Notebook file created: `train_yolo_colab.ipynb`
- [ ] Valid Jupyter notebook format (JSON)
- [ ] Contains 9 cells (1 markdown + 8 code)
- [ ] Each cell has complete code
- [ ] No hardcoded Windows paths
- [ ] BASE_PATH configurable
- [ ] Includes usage instructions
