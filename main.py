import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
import time


def main():
    # Load model
    model_path = Path("result/best.pt")
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLO model from {model_path}...")
    print(f"Using device: {device}")

    try:
        model = YOLO(str(model_path))
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam (ID 0)")
        return

    print("Webcam opened successfully. Press 'q' to quit, 's' to save screenshot")

    # FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Run inference
            results = model(frame, verbose=False, conf=0.5)
            annotated_frame = results[0].plot()

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # Get detection count
            detection_count = len(results[0].boxes)

            # Display FPS and detection count
            fps_text = f"FPS: {fps:.2f}"
            count_text = f"Detections: {detection_count}"
            device_text = f"Device: {device.upper()}"

            cv2.putText(
                annotated_frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated_frame,
                count_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated_frame,
                device_text,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Show frame
            cv2.imshow("YOLO Webcam Inference", annotated_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit requested")
                break
            elif key == ord("s"):
                screenshot_path = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"Screenshot saved: {screenshot_path}")

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released. Exiting.")


if __name__ == "__main__":
    main()
