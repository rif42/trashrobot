# Google Colab YOLO Training Notebook - Kaggle Version

## TL;DR

> **Quick Summary**: Modify `train_yolo_colab.ipynb` to download HAGRID dataset from Kaggle, filter only 'ok' class (first 1000 samples), convert annotations, and save processed data to Google Drive.
>
> **Deliverables**: 
> - Updated `train_yolo_colab.ipynb` with new data pipeline
>
> **Estimated Effort**: Quick (< 30 minutes)
> **Parallel Execution**: NO - single sequential task
> **Critical Path**: Task 1 → Complete

---

## Context

### Original Request (Modified)
User wants to change the data source from Google Drive (local data) to **Kaggle download**. The notebook should:
1. Download from https://www.kaggle.com/datasets/innominate817/hagrid-sample-30k-384p
2. Use Kaggle API with manual `kaggle.json` upload
3. Filter only 'ok' class
4. Use first 1000 'ok' samples
5. Keep original train/test split structure
6. Save processed data to Google Drive for persistence

### Current Notebook Structure (9 cells)
1. Markdown: Setup instructions
2. Mount Google Drive
3. Configure paths & generate dataset.yaml
4. Install packages & verify GPU
5. Load model
6. Train model
7. View results
8. Test inference
9. Export model

### New Notebook Structure (12 cells)
1. Markdown: Setup instructions (updated)
2. Mount Google Drive
3. **NEW**: Setup Kaggle API
4. **NEW**: Download dataset from Kaggle
5. **NEW**: Process data - filter 'ok' class, convert annotations
6. **NEW**: Save processed data to Drive
7. Install packages & verify GPU
8. Generate dataset.yaml
9. Load model
10. Train model
11. Test inference
12. Export model

---

## Work Objectives

### Core Objective
Modify the existing notebook to download from Kaggle instead of using pre-uploaded data, with proper filtering and processing pipeline.

### Concrete Deliverables
- Updated `train_yolo_colab.ipynb` with:
  - Kaggle API setup (upload kaggle.json)
  - Dataset download from Kaggle
  - 'ok' class filtering (first 1000 samples)
  - Annotation conversion (JSON to YOLO format)
  - Original train/test split preserved
  - Processed data saved to Drive
  - All original training functionality intact

### Definition of Done
- [ ] Notebook downloads from Kaggle successfully
- [ ] Filters only 'ok' class (1000 samples max)
- [ ] Converts JSON annotations to YOLO format
- [ ] Saves processed data to Drive
- [ ] Training works with new data
- [ ] Results persist to Drive

### Must Have
- Kaggle API setup with kaggle.json upload
- Download from specified Kaggle URL
- Filter 'ok' class only
- Take first 1000 'ok' samples
- Keep train/test split
- Convert annotations from COCO JSON to YOLO txt format
- Save processed images and labels to Drive
- Data validation before training
- All original cells functional

### Must NOT Have (Guardrails)
- Do NOT use hardcoded Kaggle credentials
- Do NOT download entire dataset (too large)
- Do NOT skip annotation conversion
- Do NOT lose processed data on disconnect

---

## HAGRID Dataset Technical Details

### Dataset Structure
```
hagrid-sample-30k-384p/
├── annotations/
│   ├── train/          # JSON annotation files (one per image)
│   └── test/           # JSON annotation files (one per image)
├── images/
│   ├── train/          # ~25k training images
│   └── test/           # ~5k test images
└── metadata/           # Additional metadata files
```

### Annotation Format (JSON)
Each annotation file contains:
```json
{
  "annotations": [
    {
      "label": "ok",
      "bbox": [x, y, width, height]  // COCO format (x,y top-left)
    }
  ]
}
```

### YOLO Format Conversion
YOLO format requires normalized center coordinates:
```
class_id center_x center_y width height
```
All values normalized to [0, 1] range.

### Classes in HAGRID
18 classes total: call, dislike, fist, four, like, mute, **ok**, one, palm, peace, peace_inverted, rock, stop, stop_inverted, three, three2, two_up, two_up_inverted

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES - Colab environment
- **Automated tests**: NO - manual validation
- **Validation method**: Open notebook in Colab and verify full pipeline

### QA Policy
- Notebook must complete full pipeline without errors
- Verify 'ok' samples count matches expected (≤1000)
- Verify annotations converted to YOLO format

---

## Execution Strategy

### Parallel Execution Waves

Single task - modify the existing notebook file.

```
Wave 1 (Update Notebook):
└── Task 1: Modify train_yolo_colab.ipynb
    ├── Update Cell 1 (Markdown): New setup instructions
    ├── Keep Cell 2: Mount Google Drive
    ├── NEW Cell 3: Setup Kaggle API
    ├── NEW Cell 4: Download dataset
    ├── NEW Cell 5: Process data (filter + convert)
    ├── NEW Cell 6: Save to Drive
    ├── Update Cell 7: Install packages (renumbered)
    ├── Update Cell 8: Generate YAML (renumbered)
    ├── Keep Cell 9: Load model (renumbered)
    ├── Keep Cell 10: Train (renumbered)
    ├── Keep Cell 11: Results (renumbered)
    ├── Keep Cell 12: Test (renumbered)
    └── Keep Cell 13: Export (renumbered)

Wave FINAL (Validation):
└── Validate notebook JSON structure and cell count
```

---

## TODOs

- [ ] 1. Update Google Colab Notebook for Kaggle Pipeline

  **What to do**:
  Modify `train_yolo_colab.ipynb` with the following cell changes:
  
  **Cell 1** (Update Markdown):
  ```
  # Train YOLO on HAGRID 'ok' Gesture Dataset (Google Colab - Kaggle Version)
  
  ## Setup Instructions:
  1. **Upload Kaggle API credentials**:
     - Go to https://www.kaggle.com/account, click "Create New API Token"
     - Download `kaggle.json` to your computer
     - Upload it in the next cell (files.upload())
  2. Run all cells (Runtime → Run all)
  3. The notebook will:
     - Download HAGRID dataset from Kaggle (~30k images)
     - Filter only 'ok' gestures (first 1000 samples)
     - Convert annotations to YOLO format
     - Train YOLO model with GPU acceleration
     - Save results to Google Drive
  
  ## Features:
  - ✅ Downloads from Kaggle automatically
  - ✅ Filters only 'ok' class (1000 samples)
  - ✅ Keeps original train/test split
  - ✅ GPU acceleration (if available)
  - ✅ Results saved to Google Drive
  - ✅ Model export to ONNX
  ```
  
  **Cell 2** (Keep existing): Mount Google Drive
  ```python
  # Cell 2: Mount Google Drive
  from google.colab import drive
  drive.mount('/content/drive')
  print("✅ Drive mounted at /content/drive")
  
  # Configure where to save processed data and results
  SAVE_PATH = "/content/drive/MyDrive/hagrid_ok_training"
  print(f"📁 Data will be saved to: {SAVE_PATH}")
  ```
  
  **Cell 3** (NEW): Setup Kaggle API
  ```python
  # Cell 3: Setup Kaggle API
  from google.colab import files
  import os
  import shutil
  
  print("📤 Please upload your kaggle.json file")
  print("(Download from https://www.kaggle.com/account)")
  uploaded = files.upload()
  
  # Move kaggle.json to correct location
  kaggle_path = list(uploaded.keys())[0]
  os.makedirs('/root/.kaggle', exist_ok=True)
  shutil.move(kaggle_path, '/root/.kaggle/kaggle.json')
  os.chmod('/root/.kaggle/kaggle.json', 0o600)
  
  print("✅ Kaggle API configured")
  
  # Install kaggle package
  import subprocess
  import sys
  subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
  print("✅ Kaggle package installed")
  ```
  
  **Cell 4** (NEW): Download Dataset
  ```python
  # Cell 4: Download HAGRID Dataset from Kaggle
  import subprocess
  import os
  from pathlib import Path
  
  # Create download directory
  download_dir = "/content/hagrid_download"
  os.makedirs(download_dir, exist_ok=True)
  os.chdir(download_dir)
  
  print("📥 Downloading HAGRID dataset from Kaggle...")
  print("This may take 5-10 minutes...")
  
  # Download the dataset
  subprocess.run([
      "kaggle", "datasets", "download",
      "-d", "innominate817/hagrid-sample-30k-384p",
      "--unzip"
  ], check=True)
  
  print("✅ Dataset downloaded and extracted")
  
  # Show directory structure
  import glob
  print("\n📁 Downloaded files:")
  for item in sorted(glob.glob("*")):
      if os.path.isdir(item):
          print(f"   📂 {item}/")
      else:
          print(f"   📄 {item}")
  ```
  
  **Cell 5** (NEW): Process Data - Filter 'ok' Class
  ```python
  # Cell 5: Process Data - Filter 'ok' Class and Convert Annotations
  import json
  import shutil
  from pathlib import Path
  from tqdm.notebook import tqdm
  import glob
  
  # Configuration
  MAX_SAMPLES = 1000  # Maximum 'ok' samples to use
  TARGET_CLASS = "ok"  # Only keep this class
  
  # Paths
  raw_dir = Path("/content/hagrid_download")
  processed_dir = Path("/content/hagrid_processed")
  
  # Create output directories
  for split in ['train', 'test']:
      (processed_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
      (processed_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
  
  print(f"🔍 Processing data to extract '{TARGET_CLASS}' class...")
  print(f"📊 Maximum samples: {MAX_SAMPLES}")
  
  ok_count = 0
  skipped_count = 0
  
  for split in ['train', 'test']:
      print(f"\n📂 Processing {split} split...")
      
      ann_dir = raw_dir / 'annotations' / split
      img_dir = raw_dir / 'images' / split
      
      # Get all annotation files
      ann_files = sorted(ann_dir.glob('*.json'))
      
      for ann_file in tqdm(ann_files, desc=f"Processing {split}"):
          if ok_count >= MAX_SAMPLES:
              break
          
          # Load annotation
          with open(ann_file, 'r') as f:
              ann_data = json.load(f)
          
          # Check if image has 'ok' class
          ok_annotations = [a for a in ann_data.get('annotations', []) 
                           if a.get('label') == TARGET_CLASS]
          
          if not ok_annotations:
              skipped_count += 1
              continue
          
          # Get image filename
          img_filename = ann_file.stem + '.jpg'
          img_path = img_dir / img_filename
          
          if not img_path.exists():
              skipped_count += 1
              continue
          
          # Copy image to processed directory
          shutil.copy(img_path, processed_dir / 'images' / split / img_filename)
          
          # Convert annotations to YOLO format
          # Get image dimensions
          from PIL import Image
          with Image.open(img_path) as img:
              img_width, img_height = img.size
          
          # Create YOLO format labels
          yolo_lines = []
          for ann in ok_annotations:
              bbox = ann['bbox']  # [x, y, width, height] COCO format
              x, y, w, h = bbox
              
              # Convert to YOLO format (normalized center coordinates)
              x_center = (x + w / 2) / img_width
              y_center = (y + h / 2) / img_height
              w_norm = w / img_width
              h_norm = h / img_height
              
              # Class 0 for 'ok'
              yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
          
          # Save YOLO label file
          label_file = processed_dir / 'labels' / split / (ann_file.stem + '.txt')
          with open(label_file, 'w') as f:
              f.write('\n'.join(yolo_lines))
          
          ok_count += 1
  
  print(f"\n✅ Processing complete!")
  print(f"   📊 Total 'ok' samples: {ok_count}")
  print(f"   ⏭️  Skipped (other classes): {skipped_count}")
  
  # Show split distribution
  for split in ['train', 'test']:
      img_count = len(list((processed_dir / 'images' / split).glob('*.jpg')))
      print(f"   📁 {split}: {img_count} images")
  ```
  
  **Cell 6** (NEW): Save to Drive
  ```python
  # Cell 6: Save Processed Data to Google Drive
  import shutil
  from pathlib import Path
  import os
  
  save_path = Path(SAVE_PATH)
  processed_dir = Path("/content/hagrid_processed")
  
  print(f"💾 Saving processed data to Drive...")
  print(f"   Destination: {save_path}")
  
  # Create directories
  save_path.mkdir(parents=True, exist_ok=True)
  
  # Copy processed data
  for split in ['train', 'test']:
      # Copy images
      src_img = processed_dir / 'images' / split
      dst_img = save_path / 'images' / split
      if src_img.exists():
          shutil.copytree(src_img, dst_img, dirs_exist_ok=True)
          img_count = len(list(dst_img.glob('*.jpg')))
          print(f"   ✅ Images/{split}: {img_count} files")
      
      # Copy labels
      src_lbl = processed_dir / 'labels' / split
      dst_lbl = save_path / 'labels' / split
      if src_lbl.exists():
          shutil.copytree(src_lbl, dst_lbl, dirs_exist_ok=True)
          lbl_count = len(list(dst_lbl.glob('*.txt')))
          print(f"   ✅ Labels/{split}: {lbl_count} files")
  
  print(f"\n✅ All data saved to: {save_path}")
  
  # Free up Colab storage
  print(f"\n🧹 Cleaning up temporary files...")
  shutil.rmtree("/content/hagrid_download", ignore_errors=True)
  shutil.rmtree("/content/hagrid_processed", ignore_errors=True)
  print(f"✅ Temporary files cleaned")
  ```
  
  **Cell 7** (Update existing Cell 3): Install packages
  - Add tqdm and pillow to pip install
  
  **Cell 8** (Update existing Cell 4): Generate dataset.yaml
  - Update paths to use SAVE_PATH
  - Use 'test' instead of 'val' for validation (to match HAGRID structure)
  
  **Cell 9** (Update existing Cell 5): Load model
  - Keep as is
  
  **Cell 10** (Update existing Cell 6): Train model
  - Update paths to use SAVE_PATH
  
  **Cell 11** (Update existing Cell 7): Results
  - Update paths to use SAVE_PATH
  
  **Cell 12** (Update existing Cell 8): Test inference
  - Update paths to use SAVE_PATH
  - Use 'test' split instead of 'val'
  
  **Cell 13** (Update existing Cell 9): Export
  - Update paths to use SAVE_PATH

  **Must NOT do**:
  - Do NOT hardcode Kaggle credentials
  - Do NOT download more than necessary
  - Do NOT skip the 'ok' filter
  - Do NOT use wrong annotation format
  - Do NOT break existing functionality

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Notebook modification, single file task
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None
  - **Blocked By**: None (can start immediately)

  **References** (CRITICAL):
  - Current `train_yolo_colab.ipynb` - Base file to modify
  - HAGRID dataset docs: https://github.com/hukenovs/hagrid
  - YOLO format docs: https://docs.ultralytics.com/datasets/detect/

  **Acceptance Criteria**:
  - [ ] Notebook has 13 cells total
  - [ ] Cell 3: Kaggle API setup with file upload
  - [ ] Cell 4: Downloads from Kaggle successfully
  - [ ] Cell 5: Filters only 'ok' class, max 1000 samples
  - [ ] Cell 5: Converts JSON annotations to YOLO format correctly
  - [ ] Cell 6: Saves processed data to Drive
  - [ ] Cells 7-13: Original training functionality preserved
  - [ ] All paths use SAVE_PATH variable
  - [ ] Valid notebook JSON format

  **QA Scenarios**:
  ```
  Scenario: Validate notebook structure
    Tool: Read + Python validation
    Steps:
      1. Read train_yolo_colab.ipynb
      2. Verify it's valid JSON
      3. Count cells (should be 13)
      4. Check Cell 3 has files.upload()
      5. Check Cell 5 has annotation conversion logic
    Expected Result: 13 cells with proper structure
  ```

  **Commit**: NO

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Verify notebook has correct structure and all cells functional.
  Output: `VERDICT: APPROVE/REJECT`

---

## Commit Strategy

- NO commits needed - modifies single file

---

## Success Criteria

### Verification Commands
```bash
# Check notebook exists and is valid JSON
python -c "import json; nb=json.load(open('train_yolo_colab.ipynb')); print(f'✅ Valid notebook with {len(nb[\"cells\"])} cells')"

# Count cells
python -c "import json; nb=json.load(open('train_yolo_colab.ipynb')); print(f'Cells: {len(nb[\"cells\"])} (expected: 13)')"
```

### Final Checklist
- [ ] Notebook file modified: `train_yolo_colab.ipynb`
- [ ] Valid Jupyter notebook format (JSON)
- [ ] Contains 13 cells (4 new + 9 updated)
- [ ] Cell 3: Kaggle API setup with upload
- [ ] Cell 4: Kaggle download logic
- [ ] Cell 5: 'ok' filter + annotation conversion
- [ ] Cell 6: Save to Drive logic
- [ ] Original training cells preserved and updated
