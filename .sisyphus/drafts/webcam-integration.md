# Draft: Webcam Integration for YOLO Thumbs Up Model

## Current State
- Model: `best.pt` (YOLO26n trained on HaGRID thumbs up)
- Existing inference script: `output/inference.py`
- Dependencies: ultralytics, opencv-python, numpy, Pillow

## Current Capabilities
- Image inference: `python inference.py --source image.jpg`
- Video inference: `python inference.py --source video.mp4`
- Missing: Webcam support

## Requirements Confirmed
- [x] Webcam index: 0 (default camera)
- [x] Display: Mirror image + Show FPS + Show detection confidence
- [x] Save output: Yes, save video with detections overlaid
- [x] Confidence threshold: 0.25 (25%)
- [x] Floor visible: Yes, visible in camera view
- [x] Output: Pixels on ground image (bird's eye view / top-down view)

## Research Findings - Ground Plane Calibration

### Approach: Planar Homography (Perspective Transform)
Standard CV technique for mapping image coordinates (u,v) to ground plane coordinates (x,y).

### Key OpenCV Functions
- `cv2.getPerspectiveTransform(src_pts, dst_pts)` - Compute 3x3 homography matrix from 4 points
- `cv2.findHomography()` - Alternative for >4 points (RANSAC outlier rejection)
- `cv2.perspectiveTransform(pts, matrix)` - Map image point to ground coordinate
- `cv2.warpPerspective(img, matrix, size)` - Create bird's eye view
- `cv2.FileStorage` - Save/load calibration matrix to YAML/JSON

### Calibration UX Pattern
1. Display live camera feed
2. User clicks 4 corners of known rectangle on floor (e.g., corners of a rug, tape markers)
3. User enters real-world dimensions (width in meters/centimeters, height in meters/centimeters)
4. Compute homography matrix from image points → real-world points
5. Save matrix to calibration file (e.g., `calibration.yml`)
6. Visual feedback: Show warped bird's eye view to verify calibration

### Point Order Convention
Must be consistent: Top-Left → Top-Right → Bottom-Right → Bottom-Left (clockwise) for both source and destination

### Coordinate Mapping for Detections
Use **bottom-center** of bounding box as "contact point":
```python
bottom_center_x = (x1 + x2) / 2
bottom_center_y = y2  # bottom of box
point = np.array([[[bottom_center_x, bottom_center_y]]], dtype=np.float32)
ground_point = cv2.perspectiveTransform(point, matrix)
```

### Critical Gotchas
1. **Z-height error**: Homography assumes Z=0 (points ON ground). Hand raised 1m off floor = projection appears further from camera. Acceptable for approximate positioning.
2. **Lens distortion**: Cheap webcams introduce barrel distortion. For high precision, need `cv2.undistort()` with camera matrix.
3. **Extrapolation**: Error increases exponentially outside calibrated rectangle. Stay within bounds.
4. **Point order mismatch**: Swapping TL↔TR gives mirrored/inverted coordinates.

## Requirements Confirmed
- [x] Webcam index: 0 (default camera)
- [x] Display: Mirror image + Show FPS + Show detection confidence
- [x] Save output: Yes, save video with detections
- [x] Confidence threshold: 0.25 (25%)
