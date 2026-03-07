# Create Webcam Inference Script

## Summary
Create `main.py` to run real-time inference using the trained YOLO model (`result/best.pt`) on webcam feed.

## Requirements
- Load model from `result/best.pt`
- Open webcam (ID 0)
- Run inference on each frame
- Display bounding boxes and class labels
- Show FPS and detection count
- Controls: 'q' to quit, 's' to save screenshot

## Code Structure

### main.py Content:

```python
#!/usr/bin/env python3
"""
Real-time Webcam Inference with Trained YOLO Model
"""

import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
import time

# Configuration
MODEL_PATH = Path("result/best.pt")
CONFIDENCE_THRESHOLD = 0.5
WEBCAM_ID = 0
FRAME_SKIP = 1

def load_model():
    """Load trained YOLO model."""
    print(f"Loading model from: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model = YOLO(str(MODEL_PATH))
    print(f"✅ Model loaded: {model.names}")
    
    if torch.cuda.is_available():
        print(f"   Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   Using CPU")
    
    return model

def run_webcam_inference():
    """Main webcam inference loop."""
    model = load_model()
    
    print(f"\n📷 Opening webcam (ID: {WEBCAM_ID})...")
    cap = cv2.VideoCapture(WEBCAM_ID)
    
    if not cap.isOpened():
        print(f"❌ Failed to open webcam")
        return
    
    print(f"✅ Webcam opened: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
    print(f"\nControls: 'q' = Quit, 's' = Screenshot")
    print(f"🎥 Starting inference...\n")
    
    frame_count = 0
    fps_time = time.time()
    current_fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
            
            # Draw detections (Ultralytics built-in plotting)
            annotated_frame = results[0].plot()
            
            # Calculate FPS
            if time.time() - fps_time >= 1.0:
                current_fps = frame_count
                frame_count = 0
                fps_time = time.time()
            
            # Add overlay text
            cv2.putText(annotated_frame, f"FPS: {current_fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Detections: {len(results[0].boxes)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('YOLO Webcam Inference', annotated_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Quitting...")
                break
            elif key == ord('s'):
                screenshot_path = f"webcam_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam released")

if __name__ == "__main__":
    run_webcam_inference()
```

## Dependencies
- opencv-python (cv2)
- ultralytics
- torch

## Usage
```bash
python main.py
```

## Output Location
`D:
work	rashrobotesultest.pt` (model to load)
`D:
work	rashrobotesultest.onnx` (optional ONNX export)

## Output
- Real-time webcam display with bounding boxes
- FPS counter
- Detection count
- Screenshots saved to working directory

## Notes
- Model loads from `result/best.pt`
- Uses GPU if available, falls back to CPU
- Press 'q' to quit, 's' to save screenshot
- Adjust `CONFIDENCE_THRESHOLD` to filter detections
- Change `WEBCAM_ID` if multiple webcams (0, 1, 2, ...)
