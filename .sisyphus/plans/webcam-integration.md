# Webcam Integration for YOLO Thumbs Up Model

## TL;DR

> **Quick Summary**: Add webcam support to existing inference.py with mirrored display, FPS counter, detection overlay, and video recording at 25% confidence threshold.
> 
> **Deliverables**:
> - Modified `output/inference.py` with webcam mode
> - VideoWriter integration for saving output
> - FPS display on frames
> - Horizontal mirroring for selfie-style display
> - Confidence threshold set to 0.25
> 
> **Estimated Effort**: Quick (~30 min)
> **Parallel Execution**: NO - single sequential task
> **Critical Path**: Task 1 (webcam integration) → Task 2 (QA verification)

---

## Context

### Original Request
User wants to run the trained YOLO thumbs up detection model on their webcam. The model was trained on Google Colab using HaGRID dataset and saved to `output/` folder.

### Interview Summary
**Key Discussions**:
- Webcam: Index 0 (default camera, only one camera)
- Display: Mirror horizontally (selfie-style), show FPS, show detection confidence
- Save output: Yes, save video with detections overlaid
- Confidence threshold: 0.25 (25%)

### Metis Review
**Identified Gaps** (addressed in plan):
- Buffer buildup: Set `cv2.CAP_PROP_BUFFERSIZE, 1` to prevent lag
- Codec compatibility: Use fallback chain (mp4v → XVID → MJPG) for Windows
- FPS calculation: Use moving average over 30 frames, not instantaneous
- Resource cleanup: Implement try/finally pattern for camera release
- Frame dimensions: Get from `frame.shape`, not camera properties
- Mirror timing: Apply AFTER inference to preserve detection accuracy

---

## Work Objectives

### Core Objective
Modify the existing `output/inference.py` to support webcam input (camera index 0) with mirrored display, FPS overlay, and video output saving at 25% confidence threshold.

### Concrete Deliverables
- Modified `output/inference.py` with `predict_webcam()` function
- Updated CLI arguments to support `--source 0` for webcam mode
- VideoWriter integration with proper codec fallback
- FPS counter displayed on frame
- Horizontal mirroring for selfie-style display
- Confidence threshold default changed to 0.25

### Definition of Done
- [ ] Webcam opens successfully (index 0)
- [ ] Mirrored display shows live feed with FPS counter
- [ ] Detections displayed with bounding boxes and confidence scores
- [ ] Video saved to specified output path
- [ ] Press 'q' gracefully exits and releases resources
- [ ] Video file mode continues to work unchanged

### Must Have
- Webcam capture using OpenCV VideoCapture(0)
- Buffer size set to 1 to prevent lag
- FPS display using moving average (30-frame window)
- Horizontal mirroring applied to display only (after inference)
- VideoWriter with codec fallback (mp4v → XVID → MJPG)
- Confidence threshold 0.25
- Resource cleanup in try/finally block

### Must NOT Have (Guardrails)
- No changes to model training or weights
- No changes to image inference functionality
- No changes to existing video file inference (must remain backward compatible)
- No mirroring before inference (preserve detection accuracy)
- No hard dependency on specific codecs

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES - existing `output/inference.py`
- **Automated tests**: NO - manual verification via QA scenarios
- **Framework**: N/A - verification via direct script execution

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **CLI/Script**: Use `interactive_bash` (tmux) — Run command, validate output, check files created
- **Video Validation**: Use Bash to check output file exists and has non-zero size

---

## Execution Strategy

### Parallel Execution Waves

Since this is a single-file modification with clear dependencies, execution is sequential:

```
Wave 1 (Single Task - Webcam Integration):
└── Task 1: Modify inference.py for webcam support [quick]

Wave 2 (QA Verification):
└── Task 2: Verify all modes work correctly [quick]
```

### Dependency Matrix

- **Task 1**: No dependencies → Blocks Task 2
- **Task 2**: Depends on Task 1

### Agent Dispatch Summary

- **Wave 1**: 1 task → `quick` agent
- **Wave 2**: 1 task → `quick` agent

---

## TODOs

- [ ] 1. Add Webcam Support to inference.py

  **What to do**:
  - Modify `predict_video()` function to accept both file paths (str) and webcam indices (int)
  - Add buffer size control: `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` for webcam mode
  - Implement FPS calculation using moving average over 30 frames
  - Add horizontal mirroring applied AFTER inference for display only
  - Integrate VideoWriter with codec fallback chain (mp4v → XVID → MJPG)
  - Update CLI arguments to support `--source 0` syntax for webcam
  - Set default confidence to 0.25
  - Implement try/finally resource cleanup pattern

  **Must NOT do**:
  - Do not mirror frame BEFORE inference (would affect detection accuracy)
  - Do not break existing video file mode functionality
  - Do not remove existing benchmark or image inference functions

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file modification, well-defined scope, straightforward OpenCV operations
  - **Skills**: []
    - No special skills needed - standard Python/OpenCV work

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 2
  - **Blocked By**: None (can start immediately)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `output/inference.py:36-58` - Current predict_video() function structure
  - `output/inference.py:84-106` - CLI argument parsing pattern

  **External References** (libraries and frameworks):
  - OpenCV VideoCapture: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
  - OpenCV VideoWriter_fourcc: https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html#a5f6f770004e2873f6a8c677daac82f32

  **WHY Each Reference Matters**:
  - `output/inference.py:36-58` - Shows current video processing loop, detection display, and exit handling
  - `output/inference.py:84-106` - Shows argparse setup and main execution flow
  - OpenCV docs - Reference for CAP_PROP_BUFFERSIZE and fourcc codec codes

  **Acceptance Criteria**:

  **Code Changes**:
  - [ ] Function signature updated: `predict_video(model_path, source, conf=0.25, output_path=None, mirror=True, show_fps=True)`
  - [ ] Webcam detection: `is_webcam = isinstance(source, int)`
  - [ ] Buffer control: `if is_webcam: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)`
  - [ ] FPS calculation with 30-frame moving average implemented
  - [ ] Mirroring applied after inference: `display_frame = cv2.flip(annotated, 1)`
  - [ ] VideoWriter with codec fallback chain (mp4v → XVID → MJPG)
  - [ ] CLI updated to parse `--source` as int if numeric, str otherwise
  - [ ] Default confidence changed from 0.5 to 0.25
  - [ ] Try/finally block ensures `cap.release()`, `out.release()`, `cv2.destroyAllWindows()`

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Webcam mode opens and displays correctly
    Tool: interactive_bash (tmux)
    Preconditions: Webcam connected and available at index 0
    Steps:
      1. cd output && python inference.py --model best.pt --source 0 --conf 0.25 --output webcam_test.mp4
      2. Wait 5 seconds for window to appear
      3. Press 'q' to exit
      4. Check output file exists: ls -lh webcam_test.mp4
    Expected Result: 
      - Window opens showing mirrored webcam feed
      - FPS counter visible in top-left corner
      - Output file created with size > 0 bytes
    Failure Indicators: 
      - "Could not open camera" error
      - Window doesn't appear
      - Output file is 0 bytes
    Evidence: .sisyphus/evidence/task-1-webcam-basic.png (screenshot if possible)

  Scenario: Video file mode still works
    Tool: Bash
    Preconditions: Any test video file exists (e.g., test_video.mp4)
    Steps:
      1. cd output && python inference.py --model best.pt --source test_video.mp4 --output video_out.mp4
      2. Wait for processing to complete (or press 'q' after few seconds)
      3. Check output file: ls -lh video_out.mp4
    Expected Result:
      - Video processes without errors
      - Output file created with size > 0 bytes
    Failure Indicators:
      - Errors about source type
      - Video doesn't play
    Evidence: .sisyphus/evidence/task-1-video-file.txt (command output)

  Scenario: Confidence threshold works (0.25)
    Tool: Bash
    Preconditions: Webcam available
    Steps:
      1. cd output && python inference.py --model best.pt --source 0 --conf 0.25 --conf-test
      2. Show help to verify default is 0.25: python inference.py --help | grep conf
    Expected Result:
      - Default confidence shown as 0.25 in help
      - Low-confidence detections (< 0.25) not displayed
    Failure Indicators:
      - Default still shows 0.5
      - Very low confidence detections appearing
    Evidence: .sisyphus/evidence/task-1-confidence.txt
  ```

  **Evidence to Capture**:
  - [ ] Screenshot of webcam window showing FPS and detections
  - [ ] Output video file size verification
  - [ ] Help text showing updated default confidence

  **Commit**: YES
  - Message: `feat(inference): add webcam support with FPS display and video recording`
  - Files: `output/inference.py`
  - Pre-commit: `python -c "import inference"` (syntax check)

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> Single verification task to confirm all requirements met.

- [ ] F1. **Comprehensive QA Check** — `quick` agent
  Run the complete test suite:
  1. Webcam mode: Open camera, verify mirrored display, FPS counter, detection overlay
  2. Video file mode: Process a video file, verify backward compatibility
  3. Image mode: Test single image inference still works
  4. Output validation: Confirm video files are created and playable
  5. Resource cleanup: Verify camera releases properly on exit
  
  Output: `Webcam [PASS/FAIL] | VideoFile [PASS/FAIL] | Image [PASS/FAIL] | OutputFiles [PASS/FAIL] | Cleanup [PASS/FAIL] | VERDICT: APPROVE/REJECT`

---

## Commit Strategy

- **1**: `feat(inference): add webcam support with FPS display and video recording` — output/inference.py, python -c "import inference"

---

## Success Criteria

### Verification Commands
```bash
# Test webcam mode
cd output && python inference.py --model best.pt --source 0 --conf 0.25 --output test_webcam.mp4
# Expected: Opens window, shows mirrored feed with FPS, saves video, exits on 'q'

# Test video file mode (backward compatibility)
cd output && python inference.py --model best.pt --source test.mp4 --output test_out.mp4
# Expected: Processes video, saves with detections

# Test image mode (backward compatibility)
cd output && python inference.py --model best.pt --source test.jpg
# Expected: Processes image, shows/saves results
```

### Final Checklist
- [ ] All "Must Have" present in code
- [ ] All "Must NOT Have" absent from changes
- [ ] Webcam opens and displays correctly
- [ ] FPS counter visible on frames
- [ ] Horizontal mirroring applied to display
- [ ] Video output saved successfully
- [ ] Confidence threshold defaults to 0.25
- [ ] Existing video file mode still works
- [ ] Existing image mode still works
- [ ] Resources cleanup on exit (no "camera in use" errors on next run)
