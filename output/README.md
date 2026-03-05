# YOLO26n HaGRID Thumbs Up Detector

Trained model for detecting "thumbs up" gestures.

## Model Details
- Architecture: YOLO26n
- Dataset: HaGRID (like + no_gesture classes)
- Training Resolution: 640x640
- Classes: thumbs_up, no_gesture

## Usage
```bash
pip install -r requirements.txt
python inference.py --model best.pt --source image.jpg
```

## Performance
- mAP@0.5: See evaluation_metrics.json
- FPS: >30 on T4 GPU
