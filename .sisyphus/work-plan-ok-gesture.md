# Work Plan: OK Sign Gesture Detection + Ground Position Mapping

**Project:** POC for detecting OK sign (👌) from webcam, outputting person ground coordinates  
**Hardware:** Ryzen 7 8845HS (dev) → Intel N100 (target)  
**Camera:** Laptop webcam (720p @ 30fps)  
**Range:** 1-7 meters  
**Target FPS:** 24+  
**Output:** `{person_id, ground_x, ground_y, timestamp, confidence}`  

---

## Phase 1: Environment Setup & Dependencies

**Goal:** Set up development environment with all required libraries

**Tasks:**
1. Create project structure:
   ```
   ok-gesture-poc/
   ├── data/
   │   ├── raw/                 # Downloaded datasets
   │   ├── annotations/         # Label files
   │   └── processed/           # Train/val splits
   ├── models/
   │   ├── pre-trained/         # YOLO base models
   │   └── trained/             # Our trained models
   ├── src/
   │   ├── calibration/         # Ground plane calibration
   │   ├── detection/           # YOLO + SAHI inference
   │   ├── tracking/            # ByteTrack integration
   │   └── pipeline.py          # Main orchestrator
   ├── config/
   │   ├── training.yaml        # Training hyperparameters
   │   └── inference.yaml       # Runtime config
   ├── scripts/
   │   ├── collect_data.py      # Data collection from webcam
   │   ├── train.py             # Training script
   │   ├── evaluate.py          # Model evaluation
   │   └── demo.py              # Real-time demo
   └── requirements.txt
   ```

2. Install dependencies:
   - `ultralytics` (YOLOv8)
   - `sahi` (slicing aided inference)
   - `boxmot` or `ByteTrack` (person tracking)
   - `opencv-python` (webcam capture)
   - `numpy`, `scipy` (math operations)
   - `pyyaml` (config management)
   - `tqdm` (progress bars)
   - `label-studio` or `labelImg` (annotation - optional if using pre-annotated data)

3. Verify GPU availability and PyTorch CUDA installation

4. Test basic webcam capture and display at 720p 30fps

**Deliverables:**
- [ ] Working Python environment
- [ ] Successful webcam capture test
- [ ] All imports working without errors
- [ ] Project structure created

**Estimated Time:** 2-3 hours

---

## Phase 2: Data Collection & Preparation

**Goal:** Acquire and prepare training data for OK sign detection at 1-7m range

**Tasks:**

1. **Source Training Data:**
   - Search for public OK sign datasets (Hagrid, HaGRID 2.0, or similar gesture datasets)
   - Download COCO person dataset (for person detection pre-training)
   - If limited OK sign data available, plan data augmentation strategy

2. **Collect Custom Data (if needed):**
   - Record webcam footage at varying distances (1m, 2m, 3m, 5m, 7m)
   - Different lighting conditions (bright, dim, mixed)
   - Different backgrounds (plain wall, cluttered room, outdoor)
   - Different people (various hand sizes, skin tones)
   - Target: 500-1000 OK sign examples at various distances

3. **Data Annotation:**
   - Label format: YOLO format (class_id, x_center, y_center, width, height)
   - Classes: `[person, ok_sign]`
   - For OK sign: Tight bounding box around hand making OK gesture
   - For person: Full body bounding box
   - Use label-studio, CVAT, or labelImg

4. **Dataset Organization:**
   - Split: 80% train, 10% val, 10% test
   - Create `data.yaml` for YOLO training:
     ```yaml
     path: ./data/processed
     train: train/images
     val: val/images
     test: test/images
     nc: 2
     names: ['person', 'ok_sign']
     ```

5. **Data Augmentation Strategy:**
   - YOLO built-in: Mosaic, MixUp, random affine
   - Custom: Add motion blur (simulate webcam movement)
   - Add JPEG compression artifacts (simulate streaming)
   - Random brightness/contrast (lighting variation)
   - Small object augmentation (copy-paste small OK signs onto backgrounds)

**Deliverables:**
- [ ] 500+ labeled OK sign images
- [ ] 1000+ labeled person images (can use COCO)
- [ ] Proper train/val/test split
- [ ] `data.yaml` config file
- [ ] Data augmentation pipeline configured

**Estimated Time:** 6-8 hours (depends on data availability)

---

## Phase 3: Model Training (YOLO OK Sign Detector)

**Goal:** Train YOLOv8n to detect both person and OK sign

**Tasks:**

1. **Base Model Selection:**
   - Start with YOLOv8n (nano) - fastest, smallest
   - Alternative: YOLOv8s (small) if nano accuracy insufficient
   - Download pre-trained COCO weights

2. **Training Configuration:**
   - Input resolution: 640x640 (or 1280x1280 if GPU allows)
   - Batch size: 16 (adjust based on VRAM)
   - Epochs: 100-150
   - Optimizer: SGD with momentum (YOLO default)
   - Learning rate: 0.01 (cosine annealing)
   - Augmentations: Enable mosaic, mixup, copy-paste
   
   ```python
   # Key training args
   model = YOLO('yolov8n.pt')
   results = model.train(
       data='data.yaml',
       epochs=150,
       imgsz=640,
       batch=16,
       patience=20,  # early stopping
       save=True,
       device=0,  # GPU
       workers=8,
       augment=True,
       mosaic=1.0,
       mixup=0.1,
       copy_paste=0.1,
       hsv_h=0.015,
       hsv_s=0.7,
       hsv_v=0.4,
       degrees=0.0,
       translate=0.1,
       scale=0.5,
       shear=0.0,
       perspective=0.0,
       flipud=0.0,
       fliplr=0.5,
       bgr=0.0,
   )
   ```

3. **Training Execution:**
   - Monitor mAP@0.5 and mAP@0.5:0.95
   - Track per-class metrics (person vs ok_sign)
   - Use TensorBoard or Weights & Biases for logging
   - Expected training time: 2-4 hours on Ryzen 7 GPU

4. **Model Evaluation:**
   - Run inference on test set
   - Calculate precision/recall for each class
   - Focus on OK sign detection at different distances
   - Analyze false positives (other hand gestures mistaken for OK)
   - Test inference speed (should be <40ms per image)

5. **Export to ONNX (for edge deployment):**
   - `model.export(format='onnx', imgsz=640, half=True)`
   - Verify ONNX model loads and runs correctly

**Deliverables:**
- [ ] Trained YOLOv8n model (.pt file)
- [ ] Validation metrics: mAP@0.5 > 0.75 for OK sign
- [ ] Test set evaluation report
- [ ] ONNX exported model
- [ ] Inference speed benchmark: <40ms per frame

**Estimated Time:** 4-6 hours (including training time)

**Success Criteria:**
- OK sign detection accuracy >80% at 3-5m range
- Person detection accuracy >90% (should be easy with COCO pre-training)
- Inference speed sufficient for 24+ FPS

---

## Phase 4: Person Tracking Integration (ByteTrack)

**Goal:** Implement ByteTrack to maintain person ID across frames

**Tasks:**

1. **Install ByteTrack:**
   ```bash
   pip install boxmot  # or install ByteTrack from source
   ```

2. **Integrate with YOLO Detection:**
   - Create tracking wrapper class:
   ```python
   class PersonTracker:
       def __init__(self):
           self.tracker = BYTETracker(track_thresh=0.5, 
                                      match_thresh=0.8,
                                      track_buffer=30)
       
       def update(self, detections, frame):
           # detections: list of [x1, y1, x2, y2, conf, class_id]
           # Filter for person class only
           person_dets = [d for d in detections if d[5] == 0]  # class 0 = person
           tracks = self.tracker.update(np.array(person_dets), frame.shape)
           return tracks  # Each track has: track_id, bbox, conf
   ```

3. **Associate OK Sign with Person:**
   - For each OK sign detection, find overlapping person bbox
   - Assign OK sign to the person whose bbox contains it
   - If multiple OK signs, assign to closest person by bbox center distance
   - Store mapping: `track_id → {person_bbox, ok_sign_bbox, ok_confidence}`

4. **Handle Edge Cases:**
   - OK sign detected but no person bbox (ignore)
   - Multiple OK signs on same person (possible, choose highest confidence)
   - Person ID switches (ByteTrack should minimize this)
   - Person occluded but OK sign visible (maintain track with prediction)

5. **Tracking Performance:**
   - Test ID consistency across 100+ frames
   - Measure ID switches per minute (should be <2)
   - Test with people crossing paths

**Deliverables:**
- [ ] ByteTrack integrated with YOLO detection
- [ ] Person ID maintained across frames
- [ ] OK sign correctly associated with person
- [ ] ID switch rate < 2 per minute in test scenarios
- [ ] Tracking latency < 5ms per frame

**Estimated Time:** 3-4 hours

---

## Phase 5: Ground Plane Calibration System

**Goal:** Create 4-point calibration to map image coordinates to ground plane (x,y in meters)

**Tasks:**

1. **Calibration UI Tool:**
   - Create calibration script that displays webcam feed
   - User clicks 4 points on ground plane (forming rectangle or quadrilateral)
   - User inputs real-world distances between points
   - Example: 4 corners of a 2m x 3m rectangle on floor

2. **Compute Homography Matrix:**
   ```python
   import cv2
   import numpy as np
   
   # Image points (clicked by user)
   image_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
   
   # Real-world points (input by user)
   # Example: 2m x 3m rectangle
   world_points = np.array([[0, 0], [2, 0], [2, 3], [0, 3]], dtype=np.float32)
   
   # Compute homography
   H, _ = cv2.findHomography(image_points, world_points)
   
   # Save H matrix
   np.save('calibration/homography_matrix.npy', H)
   ```

3. **Ground Position Calculation:**
   ```python
   def get_ground_position(person_bbox, H):
       # Person bottom-center (feet position)
       x1, y1, x2, y2 = person_bbox
       feet_x = (x1 + x2) / 2
       feet_y = y2  # bottom of bbox
       
       # Project to ground plane
       point = np.array([[feet_x, feet_y]], dtype=np.float32)
       world_point = cv2.perspectiveTransform(point.reshape(-1, 1, 2), H)
       
       ground_x, ground_y = world_point[0][0]
       return ground_x, ground_y
   ```

4. **Calibration Validation:**
   - Place objects at known positions
   - Verify projected coordinates match real positions
   - Calculate error: should be ±10-30cm in center of frame
   - Error increases near edges (perspective distortion)

5. **Save/Load Calibration:**
   - Save H matrix and world coordinate metadata
   - Load on startup if calibration exists
   - Allow recalibration via flag

6. **Visual Feedback:**
   - Display ground plane grid overlay during calibration
   - Show real-time coordinate readout
   - Draw floor plane boundaries

**Deliverables:**
- [ ] Calibration tool script (`calibrate.py`)
- [ ] Homography matrix computation
- [ ] Ground position projection function
- [ ] Calibration accuracy validation: ±30cm in center region
- [ ] Persistent calibration storage

**Estimated Time:** 4-5 hours

---

## Phase 6: SAHI Integration for Small Objects

**Goal:** Implement SAHI to detect small OK signs at 5-7m range

**Tasks:**

1. **Install SAHI:**
   ```bash
   pip install sahi
   ```

2. **Configure SAHI Parameters:**
   ```python
   from sahi import AutoDetectionModel
   from sahi.predict import get_sliced_prediction
   
   # Detection model
   detection_model = AutoDetectionModel.from_pretrained(
       model_type='yolov8',
       model_path='models/trained/best.pt',
       confidence_threshold=0.3,
       device='cuda:0'
   )
   
   # SAHI slicing config
   slice_height = 640
   slice_width = 640
   overlap_height_ratio = 0.25  # 25% overlap
   overlap_width_ratio = 0.25
   ```

3. **SAHI Inference Wrapper:**
   ```python
   def detect_with_sahi(frame, detection_model):
       result = get_sliced_prediction(
           image=frame,
           detection_model=detection_model,
           slice_height=640,
           slice_width=640,
           overlap_height_ratio=0.25,
           overlap_width_ratio=0.25,
           postprocess_type='NMS',  # or 'GREEDYNMM'
           postprocess_match_threshold=0.5,
           verbose=0
       )
       
       # Convert to standard format
       detections = []
       for pred in result.object_prediction_list:
           bbox = pred.bbox.to_xyxy()  # [x1, y1, x2, y2]
           conf = pred.score.value
           class_id = pred.category.id
           detections.append([*bbox, conf, class_id])
       
       return detections
   ```

4. **Optimize for Speed:**
   - SAHI adds overhead (~20-40ms)
   - Profile different slice sizes: 480, 640, 768
   - Test overlap ratios: 0.2, 0.25, 0.3
   - Find sweet spot: accuracy vs speed
   - Consider adaptive SAHI (only use when person is far)

5. **Distance-Based Strategy (Optional Optimization):**
   - Close range (<3m): Standard YOLO (no SAHI) - faster
   - Far range (3-7m): SAHI enabled - better accuracy
   - Use person bbox size as proxy for distance

6. **Validate Small Object Detection:**
   - Test OK sign detection at 7m distance
   - Compare with/without SAHI
   - Measure recall improvement for small objects

**Deliverables:**
- [ ] SAHI integrated into inference pipeline
- [ ] Optimized slice/overlap parameters
- [ ] Inference speed with SAHI: <50ms per frame (target 24 FPS)
   - Target: 35-40ms (allows 24-28 FPS)
- [ ] Improved detection at 5-7m range: >70% recall

**Estimated Time:** 3-4 hours

---

## Phase 7: Real-time Pipeline Integration

**Goal:** Combine all components into unified real-time pipeline

**Tasks:**

1. **Pipeline Architecture:**
   ```python
   class OKSignPipeline:
       def __init__(self, model_path, calib_file):
           # Load detection model
           self.detection_model = AutoDetectionModel.from_pretrained(...)
           
           # Initialize tracker
           self.tracker = BYTETracker(...)
           
           # Load calibration
           self.H = np.load(calib_file)
           
           # Performance tracking
           self.fps_counter = deque(maxlen=30)
       
       def process_frame(self, frame):
           # 1. Detect with SAHI
           detections = self.detect(frame)
           
           # 2. Track persons
           tracks = self.update_tracks(detections, frame)
           
           # 3. Associate OK signs
           gestures = self.associate_gestures(detections, tracks)
           
           # 4. Get ground positions
           results = []
           for gesture in gestures:
               ground_x, ground_y = self.get_ground_position(gesture['person_bbox'])
               results.append({
                   'person_id': gesture['track_id'],
                   'ground_x': ground_x,
                   'ground_y': ground_y,
                   'timestamp': datetime.now().isoformat(),
                   'confidence': gesture['ok_confidence']
               })
           
           return results
   ```

2. **Main Loop:**
   ```python
   def main():
       # Initialize webcam
       cap = cv2.VideoCapture(0)
       cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
       cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
       cap.set(cv2.CAP_PROP_FPS, 30)
       
       # Initialize pipeline
       pipeline = OKSignPipeline('models/best.pt', 'calibration/homography_matrix.npy')
       
       while True:
           ret, frame = cap.read()
           if not ret:
               break
           
           # Process frame
           results = pipeline.process_frame(frame)
           
           # Visualize
           display_frame = pipeline.visualize(frame, results)
           
           # Show FPS
           fps = pipeline.get_fps()
           cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), ...)
           
           # Print results
           for r in results:
               print(f"[ID {r['person_id']}] OK sign at ({r['ground_x']:.2f}, {r['ground_y']:.2f})m")
           
           cv2.imshow('OK Sign Detection', display_frame)
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
   ```

3. **Visualization:**
   - Draw person bounding boxes with ID
   - Highlight OK sign detection
   - Display ground coordinates as text overlay
   - Draw ground plane grid (if calibrated)
   - Show FPS counter
   - Color-coded status: Green (detected), Yellow (tracking), Red (lost)

4. **Output Logging:**
   - Print detections to console
   - Optional: Log to JSONL file
   - Optional: Stream via WebSocket for future robot integration

5. **Error Handling:**
   - Handle camera disconnections
   - Handle model loading failures
   - Graceful degradation if calibration missing
   - Keyboard interrupt handling

**Deliverables:**
- [ ] Unified pipeline script (`demo.py`)
- [ ] Real-time visualization with all components
- [ ] Console output: Person ID + coordinates
- [ ] FPS display (target: 24+)
- [ ] Smooth operation without crashes

**Estimated Time:** 4-5 hours

---

## Phase 8: Performance Optimization & N100 Testing

**Goal:** Optimize for target hardware (Intel N100) and validate performance

**Tasks:**

1. **Profile Current Performance (Ryzen 7):**
   - Measure breakdown:
     - Frame capture: ~5ms
     - SAHI detection: ~30-40ms
     - Tracking: ~3ms
     - Calibration/projection: ~1ms
     - Visualization: ~5ms
   - Total target: <42ms per frame (for 24 FPS)

2. **Optimization Strategies:**
   - **ONNX Runtime:** Use ONNX model with optimized runtime
   - **OpenVINO:** Intel-optimized inference (crucial for N100)
   - **Reduce SAHI overhead:** Skip SAHI for close-range detections
   - **Frame skipping:** Process every Nth frame if needed
   - **Half precision (FP16):** Faster inference, minimal accuracy loss
   - **Batch inference:** If multiple slices, batch them

3. **OpenVINO Conversion (for N100):**
   ```bash
   # Convert to OpenVINO format
   yolo export model=best.pt format=openvino half=True
   ```
   - Test inference with OpenVINO runtime
   - Compare speed vs PyTorch

4. **Intel N100 Testing:**
   - Deploy to N100 device
   - Benchmark FPS with different configurations
   - Identify bottlenecks
   - Iterate on optimizations

5. **Adaptive Quality:**
   - If FPS drops, reduce input resolution
   - If FPS drops, reduce SAHI overlap
   - If FPS drops, reduce model size (nano vs small)
   - Maintain minimum 24 FPS

**Deliverables:**
- [ ] Performance profiling report
- [ ] Optimized model (ONNX or OpenVINO)
- [ ] N100 benchmark results: 24+ FPS confirmed
- [ ] Fallback strategies documented (what to reduce if slow)

**Estimated Time:** 4-6 hours (including N100 testing)

---

## Phase 9: Final Validation & Demo

**Goal:** Validate system meets all requirements and create demo

**Tasks:**

1. **Validation Testing:**
   - Test at distances: 1m, 2m, 3m, 5m, 7m
   - Test with 3-5 different people
   - Test different lighting conditions
   - Test continuous operation (5 minutes)
   - Measure accuracy metrics:
     - True positives (correct OK detections)
     - False positives (wrong gestures detected)
     - False negatives (missed OK signs)
     - Coordinate error (ground truth vs measured)

2. **Create Test Report:**
   | Distance | OK Detection Rate | Coordinate Error | FPS |
   |----------|-------------------|------------------|-----|
   | 1m       | 95%              | ±5cm            | 28  |
   | 3m       | 90%              | ±15cm           | 26  |
   | 5m       | 85%              | ±25cm           | 25  |
   | 7m       | 75%              | ±40cm           | 24  |

3. **Demo Video:**
   - Record 2-3 minute demo
   - Show detection at multiple distances
   - Show coordinate output
   - Narrate or add captions

4. **Documentation:**
   - Update README with setup instructions
   - Document calibration procedure
   - Document API/output format
   - Add troubleshooting section

5. **Code Cleanup:**
   - Remove debug prints
   - Add docstrings
   - Organize imports
   - Add type hints (optional)

**Deliverables:**
- [ ] Validation test report
- [ ] Demo video
- [ ] Complete README documentation
- [ ] Clean, documented codebase
- [ ] Requirements.txt verified on clean environment

**Estimated Time:** 3-4 hours

---

## Success Criteria Summary

| Metric | Target | Phase |
|--------|--------|-------|
| OK Sign Detection (1-3m) | >90% accuracy | Phase 9 |
| OK Sign Detection (5-7m) | >70% accuracy | Phase 9 |
| Ground Position Error | ±30cm (center) | Phase 9 |
| FPS (Ryzen 7) | >24 | Phase 7 |
| FPS (Intel N100) | >24 | Phase 8 |
| Tracking ID Persistence | <2 switches/min | Phase 4 |
| Latency (end-to-end) | <50ms | Phase 7 |

---

## Total Estimated Time

- Phase 1 (Setup): 2-3 hours
- Phase 2 (Data): 6-8 hours
- Phase 3 (Training): 4-6 hours
- Phase 4 (Tracking): 3-4 hours
- Phase 5 (Calibration): 4-5 hours
- Phase 6 (SAHI): 3-4 hours
- Phase 7 (Pipeline): 4-5 hours
- Phase 8 (Optimization): 4-6 hours
- Phase 9 (Validation): 3-4 hours

**Total: 33-45 hours** (spread over 2-3 weeks depending on availability)

**Critical Path:** Phases 2-3 (data + training) are the longest and most important.

---

## Next Steps

When ready to start:

1. **Confirm hardware setup:**
   - Ryzen 7 laptop with webcam ready
   - Intel N100 device available for Phase 8
   - Sufficient disk space (10GB+ for datasets)

2. **Prioritize data collection:**
   - Search for public OK sign datasets first
   - If insufficient, plan recording sessions
   - Data is the biggest blocker

3. **Start with Phase 1:**
   - Environment setup is straightforward
   - Unblocks all other phases

Ready to begin? Say "start Phase X" or "begin implementation" and I'll dive into the first task!

---

**Last Updated:** 2026-03-04  
**Status:** Work plan complete, awaiting execution signal
