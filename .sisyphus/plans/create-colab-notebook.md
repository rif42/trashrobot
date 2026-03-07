# Create Google Colab Notebook

## Summary
Create a new Jupyter notebook `train_yolo_colab.ipynb` configured for Google Colab that uses the training data from Google Drive.

## Background
- User has uploaded `data/processed_from_raw/` to Google Drive
- Need Colab-specific setup (Drive mounting, paths, GPU detection)
- Should be similar to local notebook but adapted for Colab environment

## Changes Required

### Create File: train_yolo_colab.ipynb

**Cell 1: Setup (Mount Drive + Install)**
```python
# Mount Google Drive and Install Dependencies
from google.colab import drive
import subprocess
import sys

# Mount Google Drive
drive.mount('/content/drive')

# Install ultralytics
subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"])

from ultralytics import YOLO
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

**Cell 2: Configure Paths**
```python
# IMPORTANT: Update this path to where data is in Google Drive
DRIVE_PATH = "/content/drive/MyDrive/data/processed_from_raw"

# Verify data exists
data_path = Path(DRIVE_PATH)
Check: dataset.yaml, images/train, images/val, labels/train, labels/val
Print counts of images and labels
```

**Cell 3: Display Sample Image**
```python
# Load random validation image
# Load corresponding YOLO label file
# Draw bounding boxes using YOLO format [x_center, y_center, width, height]
# Display image with annotations
```

**Cell 4: Load Model**
```python
# Load YOLO model (yolo11n.pt for Colab free tier)
model = YOLO('yolo11n.pt')
```

**Cell 5: Train**
```python
# Create temporary dataset.yaml pointing to Google Drive path
# Train with settings optimized for Colab:
#   - epochs=20 (adjustable)
#   - batch=16 (or 8 if OOM)
#   - device='0' (GPU) or 'cpu'
#   - Save to /content/runs/detect (local, faster)

yaml_content = f"""
path: {data_path.absolute()}
train: images/train
val: images/val
nc: 2
names: ['ok', 'no_gesture']
"""

results = model.train(
    data=str(temp_yaml),
    epochs=20,
    imgsz=640,
    batch=16,
    device='0' if torch.cuda.is_available() else 'cpu',
    patience=10,
    save=True,
    project='/content/runs/detect',
    name='train',
    exist_ok=True
)
```

**Cell 6: View Results**
```python
# Display results.png, confusion_matrix.png, val_batch0_pred.jpg
# List model weights (best.pt, last.pt)
```

**Cell 7: Test Inference**
```python
# Load best model from /content/runs/detect/train/weights/best.pt
# Test on random validation image
# Show detections with confidence scores
```

**Cell 8: Save to Drive (Optional)**
```python
# Copy best.pt, results.png, confusion_matrix.png to Google Drive
# Location: {DRIVE_PATH}/../training_results/
```

**Cell 9: Export (Optional)**
```python
# Export model to ONNX format
# Can copy to Drive using Cell 8
```

## Key Differences from Local Notebook:
1. **Drive mounting**: Required to access data
2. **Path configuration**: Uses Google Drive paths
3. **Temporary yaml**: Points to Drive location
4. **Save location**: /content/runs (local Colab storage, faster than Drive)
5. **GPU detection**: Automatically uses GPU if available
6. **Export results**: Optional cell to copy results back to Drive

## Output
- File: `train_yolo_colab.ipynb` in workspace root
- Ready to upload to Google Colab
- User just needs to update DRIVE_PATH in Cell 2

## Testing
After creation, verify:
- Valid JSON structure
- All cells have correct format
- No syntax errors
- Paths are properly configured
