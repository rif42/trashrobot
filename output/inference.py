
#!/usr/bin/env python3
"""
Inference script for YOLO26n HaGRID thumbs up detection
"""

from ultralytics import YOLO
import cv2
import argparse
import sys
from pathlib import Path

def predict_image(model_path, image_path, conf=0.5, save=True):
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
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0]
                print(f"  Class {cls}: {conf:.2f} at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    return results

def predict_video(model_path, video_path, conf=0.5):
    """
    Run inference on video
    """
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, verbose=False)

        # Display
        annotated = results[0].plot()
        cv2.imshow('Detection', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def benchmark(model_path, n=100):
    """
    Benchmark inference speed
    """
    import time
    import numpy as np

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
    parser = argparse.ArgumentParser(description='Thumbs Up Detection')
    parser.add_argument('--model', default='best.pt', help='Model path')
    parser.add_argument('--source', required=True, help='Image or video path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')

    # Use sys.argv[1:] to avoid Colab kernel args if run as script,
    # but we handle the exec() case by checking if we are in an interactive shell
    try:
        args = parser.parse_args()
    except SystemExit:
        # If running via exec() in notebook, don't crash the kernel
        print("Running in notebook mode: Skipping argparse execution.")
        args = None

    if args:
        if args.benchmark:
            benchmark(args.model)
        elif args.source.endswith(('.mp4', '.avi', '.mov')):
            predict_video(args.model, args.source, args.conf)
        else:
            predict_image(args.model, args.source, args.conf)
