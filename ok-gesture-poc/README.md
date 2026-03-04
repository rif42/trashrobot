# OK Sign Gesture Detection - POC

Real-time OK sign (👌) detection from webcam with ground position mapping for robot navigation.

## Features

- ✅ **Gesture Detection**: Detects OK sign at 1-7m range using YOLO
- ✅ **Person Tracking**: Maintains person ID across frames using ByteTrack
- ✅ **Ground Mapping**: Maps detected gestures to real-world coordinates
- ✅ **Small Object Detection**: SAHI integration for distant hands
- ✅ **Calibration Tool**: Interactive 4-point ground plane calibration
- ✅ **Performance**: 24+ FPS target on Ryzen 7 and Intel N100

## Quick Start

### 1. Install Dependencies

```bash
cd ok-gesture-poc

# Option A: Full installation (recommended when network stable)
pip install -r requirements.txt

# Option B: Minimal installation (core packages only)
pip install -r requirements-minimal.txt
pip install ultralytics sahi boxmot  # Install separately if needed
```

### 2. Verify Environment

```bash
python scripts/test_environment.py
```

### 3. Calibrate Camera

```bash
python -m src.calibration.calibrator --interactive
```

Follow instructions to click 4 points on ground and enter real-world coordinates.

### 4. Collect Training Data

```bash
# Collect at all distances (recommended)
python scripts/collect_data.py --mode multi --duration 30

# Or collect at specific distance
python scripts/collect_data.py --distance close --duration 30
```

### 5. Prepare Dataset

```bash
python scripts/prepare_dataset.py
```

### 6. Train Model

```bash
python scripts/train.py --model n --epochs 150 --export
```

### 7. Run Pipeline

```bash
# Run with existing calibration
python -m src.pipeline

# Run with calibration first
python -m src.pipeline --calibrate
```

## Project Structure

```
ok-gesture-poc/
├── config/                  # Configuration files
│   ├── config.yaml         # Main configuration
│   └── data.yaml           # Dataset configuration
├── data/                   # Data directory
│   ├── raw/               # Collected images
│   ├── processed/         # Train/val/test splits
│   └── annotations/       # Label files
├── models/                # Model storage
│   ├── pre-trained/       # Base YOLO models
│   └── trained/          # Our trained models
├── src/                   # Source code
│   ├── pipeline.py       # Main pipeline
│   ├── detection/        # Detection modules
│   │   └── sahi_detector.py
│   ├── tracking/         # Tracking modules
│   │   └── tracker.py
│   └── calibration/      # Calibration modules
│       └── calibrator.py
├── scripts/              # Utility scripts
│   ├── collect_data.py   # Data collection
│   ├── prepare_dataset.py
│   ├── train.py          # Training script
│   ├── test_environment.py
│   └── test_camera.py
├── requirements.txt
└── README.md
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Model settings
model:
  name: "yolov8n"  # n, s, or m

# Training settings
training:
  epochs: 150
  batch_size: 16
  imgsz: 640

# SAHI settings (small object detection)
sahi:
  enabled: true
  slice_height: 640
  slice_width: 640
  overlap_height_ratio: 0.25

# Tracking settings
tracking:
  track_thresh: 0.5
  match_thresh: 0.8

# Camera settings
camera:
  width: 1280
  height: 720
  fps: 30
```

## Output Format

When OK sign is detected, the pipeline outputs:

```json
{
  "person_id": 1,
  "ground_x": 2.5,
  "ground_y": 3.2,
  "timestamp": "2026-03-04T10:30:00",
  "confidence": 0.94,
  "person_bbox": [100, 200, 300, 500],
  "gesture_bbox": [200, 250, 250, 300]
}
```

- `person_id`: Persistent ID of the person
- `ground_x, ground_y`: Position in meters (from calibration origin)
- `confidence`: Detection confidence (0-1)
- `bbox`: Bounding box coordinates [x1, y1, x2, y2]

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| OK Detection (1-3m) | >90% | Close range, easy |
| OK Detection (5-7m) | >70% | Far range, requires SAHI |
| Ground Position Error | ±30cm | Center of frame |
| FPS (Ryzen 7) | >24 | GPU acceleration |
| FPS (Intel N100) | >24 | OpenVINO optimized |

## Troubleshooting

### "ultralytics not found"
```bash
pip install ultralytics
```

### "Camera not accessible"
- Close other camera applications
- Try different camera index: edit `config.yaml` -> `camera: source: 1`

### "Calibrator not calibrated"
Run calibration first:
```bash
python -m src.calibration.calibrator --interactive
```

### Low FPS
- Reduce SAHI overlap ratio in config
- Use smaller model (yolov8n instead of yolov8s)
- Reduce camera resolution

## Next Steps (Future Development)

- [ ] Multi-camera support
- [ ] Robot communication protocol (MQTT/WebSocket)
- [ ] Obstacle detection
- [ ] Multi-robot coordination
- [ ] ROS2 integration

## Requirements

**Hardware:**
- Webcam (720p minimum)
- Ryzen 7 8845HS (dev) or Intel N100 (target)
- 8GB+ RAM
- 10GB disk space

**Software:**
- Python 3.10+
- CUDA (optional, for GPU acceleration)
- Windows/Linux/macOS

## License

MIT License - See LICENSE file

## Credits

- YOLOv8 by Ultralytics
- SAHI for small object detection
- ByteTrack for person tracking
- OpenCV for computer vision
