#!/usr/bin/env python3
"""
Inference script for YOLO26n HaGRID thumbs up detection with ground plane mapping

Supports:
- Image inference
- Video file inference
- Webcam inference with FPS display and video recording
- Ground plane calibration and coordinate mapping
- Bird's eye view visualization

Usage:
    # Image
    python inference.py --model best.pt --source image.jpg

    # Video file
    python inference.py --model best.pt --source video.mp4 --output result.mp4

    # Webcam (index 0)
    python inference.py --model best.pt --source 0 --output webcam_result.mp4

    # With ground calibration
    python inference.py --model best.pt --source 0 --calibration calibration.yml --output result.mp4
"""

from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import sys
import time
from pathlib import Path


def load_calibration(calibration_file):
    """Load homography matrix from calibration file"""
    try:
        fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
        homography = fs.getNode("homography").mat()
        fs.release()

        if homography is None or homography.size == 0:
            print(f"Error: Could not load homography from {calibration_file}")
            return None

        print(f"Loaded calibration from: {calibration_file}")
        return homography

    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None


def predict_image(model_path, image_path, conf=0.25, save=True):
    """
    Run inference on a single image
    """
    # Load model
    model = YOLO(model_path)

    # Run prediction
    results = model.predict(image_path, conf=conf, save=save)

    # Extract detections
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            print(f"Found {len(boxes)} detection(s)")
            for box in boxes:
                cls = int(box.cls)
                conf_val = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0]
                print(
                    f"  Class {cls}: {conf_val:.2f} at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
                )

    return results


def predict_video(
    model_path,
    source,
    conf=0.25,
    output_path=None,
    calibration_file=None,
    mirror=True,
    show_fps=True,
    show_birdseye=True,
):
    """
    Run inference on video file or webcam with optional ground plane mapping

    Args:
        model_path: Path to YOLO model
        source: Video file path (str) or webcam index (int)
        conf: Confidence threshold
        output_path: Path to save output video (optional)
        calibration_file: Path to calibration.yml (optional)
        mirror: Mirror display horizontally (for webcam selfie view)
        show_fps: Display FPS counter
        show_birdseye: Show bird's eye view window (requires calibration)
    """
    # Load model
    model = YOLO(model_path)

    # Load calibration if provided
    homography = None
    if calibration_file:
        homography = load_calibration(calibration_file)
        if homography is None:
            print("Warning: Calibration loading failed. Ground mapping disabled.")

    # Determine if source is webcam or file
    is_webcam = isinstance(source, int)

    # Open video source
    cap = cv2.VideoCapture(source if is_webcam else str(source))
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        if is_webcam:
            print("Is the camera connected and not in use by another application?")
        return

    # For webcam: set buffer size to prevent lag
    if is_webcam:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"Webcam opened (index: {source})")
    else:
        print(f"Video file opened: {source}")

    # Get video properties
    fps_input = cap.get(cv2.CAP_PROP_FPS) if not is_webcam else 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup VideoWriter if output path specified
    video_writer = None
    if output_path:
        # Try multiple codecs for compatibility
        codecs = ["mp4v", "XVID", "MJPG"]
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]
                vw = cv2.VideoWriter(
                    output_path, fourcc, fps_input, (frame_width, frame_height)
                )
                if vw.isOpened():
                    video_writer = vw
                    print(f"Output video: {output_path} (codec: {codec})")
                    break
            except Exception:
                continue

        if video_writer is None:
            print(f"Error: Could not create video writer for {output_path}")

    # FPS calculation variables
    fps_history = []
    fps_window = 30  # Moving average window
    last_time = time.time()
    frame_count = 0

    # Window names
    main_window = "Thumbs Up Detection"
    birdseye_window = "Bird's Eye View"

    # Create windows
    cv2.namedWindow(main_window)
    if show_birdseye and homography is not None:
        cv2.namedWindow(birdseye_window)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    print("Error: Could not read frame from camera")
                break

            # Calculate FPS
            current_time = time.time()
            fps = (
                1.0 / (current_time - last_time)
                if (current_time - last_time) > 0
                else 0
            )
            last_time = current_time

            # Moving average FPS
            fps_history.append(fps)
            if len(fps_history) > fps_window:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

            # Run inference
            results = model.predict(frame, conf=conf, verbose=False)

            # Get annotated frame
            annotated = results[0].plot()

            # Process detections for ground mapping
            if homography is not None and results[0].boxes is not None:
                for box in results[0].boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Calculate bottom-center (contact point with ground)
                    bottom_center_x = (x1 + x2) / 2
                    bottom_center_y = y2

                    # Map to ground coordinates
                    point = np.array(
                        [[[bottom_center_x, bottom_center_y]]], dtype=np.float32
                    )
                    ground_point = cv2.perspectiveTransform(point, homography)
                    ground_x, ground_y = ground_point[0][0]

                    # Display ground coordinates on frame
                    coord_text = f"Ground: ({ground_x:.1f}, {ground_y:.1f})"
                    cv2.putText(
                        annotated,
                        coord_text,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )

            # Add FPS counter
            if show_fps:
                fps_text = f"FPS: {avg_fps:.1f}"
                cv2.putText(
                    annotated,
                    fps_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Add instructions
            instructions = "Press 'q' to quit"
            if homography is not None:
                instructions += " | Ground mapping enabled"
            cv2.putText(
                annotated,
                instructions,
                (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Mirror display for webcam (selfie style)
            display_frame = annotated
            if is_webcam and mirror:
                display_frame = cv2.flip(annotated, 1)

            # Show main window
            cv2.imshow(main_window, display_frame)

            # Create bird's eye view
            if show_birdseye and homography is not None:
                birds_eye = cv2.warpPerspective(
                    frame, homography, (frame_width, frame_height)
                )

                # Draw hand positions on bird's eye view
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bottom_center_x = (x1 + x2) / 2
                        bottom_center_y = y2

                        point = np.array(
                            [[[bottom_center_x, bottom_center_y]]], dtype=np.float32
                        )
                        ground_point = cv2.perspectiveTransform(point, homography)
                        ground_x, ground_y = ground_point[0][0]

                        # Draw circle at ground position
                        cv2.circle(
                            birds_eye,
                            (int(ground_x), int(ground_y)),
                            10,
                            (0, 0, 255),
                            -1,
                        )

                # Add label
                cv2.putText(
                    birds_eye,
                    "Bird's Eye View",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(birdseye_window, birds_eye)

            # Write to output video (save original annotated, not mirrored)
            if video_writer is not None:
                video_writer.write(annotated)

            frame_count += 1

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopping...")
                break

    finally:
        # Cleanup
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"Output saved to: {output_path}")
        cv2.destroyAllWindows()

        print(f"\nProcessed {frame_count} frames")


def benchmark(model_path, n=100):
    """
    Benchmark inference speed
    """
    import time

    model = YOLO(model_path)
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    times = []
    for _ in range(n):
        start = time.time()
        model.predict(test_img, imgsz=640, verbose=False)
        times.append(time.time() - start)

    mean_time = np.mean(times)
    fps = 1.0 / mean_time

    print(f"Benchmark Results ({n} iterations):")
    print(f"  Mean time: {mean_time * 1000:.2f} ms")
    print(f"  FPS: {fps:.2f}")
    print(f"  Std: {np.std(times) * 1000:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thumbs Up Detection with Ground Mapping"
    )
    parser.add_argument("--model", default="best.pt", help="Model path")
    parser.add_argument(
        "--source",
        required=True,
        help="Image path, video path, or webcam index (0, 1, ...)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument("--output", type=str, help="Output video path (optional)")
    parser.add_argument(
        "--calibration",
        type=str,
        help="Calibration file path (optional, enables ground mapping)",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirroring for webcam",
    )
    parser.add_argument(
        "--no-fps", action="store_true", help="Disable FPS counter display"
    )
    parser.add_argument(
        "--no-birdseye", action="store_true", help="Disable bird's eye view window"
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")

    try:
        args = parser.parse_args()
    except SystemExit:
        print("Running in notebook mode: Skipping argparse execution.")
        args = None

    if args:
        if args.benchmark:
            benchmark(args.model)
        else:
            # Determine if source is image, video file, or webcam
            source = args.source

            # Try to parse as integer (webcam index)
            try:
                source = int(source)
                is_webcam = True
            except ValueError:
                is_webcam = False
                source_path = Path(source)

            if is_webcam:
                # Webcam mode
                predict_video(
                    args.model,
                    source,
                    conf=args.conf,
                    output_path=args.output,
                    calibration_file=args.calibration,
                    mirror=False,
                    show_fps=not args.no_fps,
                    show_birdseye=not args.no_birdseye,
                )
            elif str(source).lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                # Video file mode
                predict_video(
                    args.model,
                    source,
                    conf=args.conf,
                    output_path=args.output,
                    calibration_file=args.calibration,
                    mirror=False,  # Don't mirror video files
                    show_fps=not args.no_fps,
                    show_birdseye=not args.no_birdseye,
                )
            else:
                # Image mode
                predict_image(args.model, source, conf=args.conf)
