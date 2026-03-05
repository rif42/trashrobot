#!/usr/bin/env python3
"""
Inference script for YOLO26n HaGRID thumbs up detection - SIMPLIFIED

No calibration, no ground mapping. Just thumb detection.

Usage:
    # Webcam (index 0)
    python inference_simple.py --model best.pt --source 0 --output result.mp4
    
    # Video file
    python inference_simple.py --model best.pt --source video.mp4 --output result.mp4
    
    # Image
    python inference_simple.py --model best.pt --source image.jpg
"""

from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import sys
import time

def predict_image(model_path, image_path, conf=0.25, save=True):
    """Run inference on a single image"""
    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf, save=save)

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            print(f"Found {len(boxes)} detection(s)")
            for box in boxes:
                cls = int(box.cls)
                conf_val = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0]
                print(f"  Class {cls}: {conf_val:.2f} at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    return results

def predict_video(model_path, source, conf=0.25, output_path=None, show_fps=True):
    """Run inference on video file or webcam"""
    model = YOLO(model_path)
    
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
        codecs = ['mp4v', 'XVID', 'MJPG']
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                vw = cv2.VideoWriter(output_path, fourcc, fps_input, (frame_width, frame_height))
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
    fps_window = 30
    last_time = time.time()
    frame_count = 0
    
    # Window name
    window_name = 'Thumbs Up Detection'
    cv2.namedWindow(window_name)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    print("Error: Could not read frame from camera")
                break
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
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
            
            # Add FPS counter
            if show_fps:
                fps_text = f"FPS: {avg_fps:.1f}"
                cv2.putText(annotated, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add instructions
            instructions = "Press 'q' to quit"
            cv2.putText(annotated, instructions, (10, frame_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show window
            cv2.imshow(window_name, annotated)
            
            # Write to output video
            if video_writer is not None:
                video_writer.write(annotated)
            
            frame_count += 1
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
    """Benchmark inference speed"""
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
    print(f"  Mean time: {mean_time*1000:.2f} ms")
    print(f"  FPS: {fps:.2f}")
    print(f"  Std: {np.std(times)*1000:.2f} ms")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Thumbs Up Detection - Simple')
    parser.add_argument('--model', default='best.pt', help='Model path')
    parser.add_argument('--source', required=True, 
                       help='Image path, video path, or webcam index (0, 1, ...)')
    parser.add_argument('--conf', type=float, default=0.5, 
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--output', type=str, 
                       help='Output video path (optional)')
    parser.add_argument('--no-fps', action='store_true',
                       help='Disable FPS counter display')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run benchmark')

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
            
            if is_webcam:
                # Webcam mode
                predict_video(
                    args.model, 
                    source, 
                    conf=args.conf,
                    output_path=args.output,
                    show_fps=not args.no_fps
                )
            elif str(source).lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Video file mode
                predict_video(
                    args.model,
                    source,
                    conf=args.conf,
                    output_path=args.output,
                    show_fps=not args.no_fps
                )
            else:
                # Image mode
                predict_image(args.model, source, conf=args.conf)
