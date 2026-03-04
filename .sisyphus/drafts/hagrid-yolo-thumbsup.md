# Draft: YOLOv8n + HaGRID Thumbs Up Training Plan

## ✅ DECISIONS FINALIZED

Based on user requirements:
1. ✅ **Target**: Detect small hands at distance (hand fills <20% of frame)
2. ✅ **Model**: YOLO26n (latest with ProgLoss + STAL for small objects, NMS-free)
3. ✅ **Training resolution**: 640×640 (imgsz=640)
4. ✅ **Gesture**: Single class - "thumbs up" (HaGRID "like" class)
5. ✅ **Environment**: Google Colab with T4 GPU
6. ✅ **Negative class**: Include "no_gesture" at 10% ratio
7. ✅ **Split strategy**: Person-stratified (not random)

### Why YOLO26n?
- **Highest accuracy**: 40.9% mAP (vs 37.3% YOLOv8, 39.5% YOLO11)
- **Smallest model**: 2.4M parameters (vs 3.2M YOLOv8, 2.6M YOLO11)
- **Built for small objects**: ProgLoss + STAL specifically target small object detection
- **NMS-free**: No post-processing overhead, consistent latency
- **43% faster CPU inference**: Better for edge deployment
- **Released Jan 2026**: Latest official Ultralytics model

## Research Findings

### HaGRID Dataset
- **Full Name**: HAnd Gesture Recognition Image Dataset
- **Size**: 554,800+ annotated images across 18 classes
- **"like" class**: ~30,000 thumbs up images
- **"no_gesture" class**: ~30,000 negative samples
- **Annotations**: YOLO format bounding boxes
- **Metadata**: Includes `user_id` field for stratified splitting

### YOLOv8n-p2 Specifications
- **Purpose**: Small object detection (P2 head with stride 4)
- **Parameters**: ~3.4M
- **FLOPs**: 17.4 GFLOPs (2x standard YOLOv8n)
- **Detection heads**: P2, P3, P4, P5 (4 heads total)
- **Best for**: Objects <64px in feature maps

### Why YOLOv8n-p2 for Small Hands?
Standard YOLOv8n has 3 heads (P3, P4, P5) - smallest stride is 8.
YOLOv8n-p2 adds P2 head with stride 4 = 2x finer grid for small objects.
Critical for detecting distant hands that appear small in frame.

## Open Questions - ALL RESOLVED ✅
- [x] Target image size: 640×640 training, various inference
- [x] Single class: thumbs up only + no_gesture negative
- [x] Hardware: Google Colab T4 GPU
- [x] Training: Fine-tune from COCO pretrained weights
- [x] Model: YOLOv8n-p2 for small object detection

## Plan Status
Plan saved to: `.sisyphus/plans/hagrid-yolo-thumbsup-training.md`
