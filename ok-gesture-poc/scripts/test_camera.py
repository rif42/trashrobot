#!/usr/bin/env python3
"""
Webcam Test Script
Verify camera is working before starting full pipeline.
"""

import cv2
import time


def test_webcam(source=0, width=1280, height=720, fps=30, duration=10):
    """Test webcam capture and display FPS."""
    print(f"Testing webcam (source={source})...")
    print(f"Target: {width}x{height} @ {fps}fps")

    cap = cv2.VideoCapture(source)

    # Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Get actual settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"\nActual settings: {actual_width}x{actual_height} @ {actual_fps}fps")

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return False

    frame_count = 0
    start_time = time.time()

    print("\nCapturing... Press 'q' to stop early")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame")
                break

            frame_count += 1
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0

            # Display info on frame
            info_text = f"Frame: {frame_count} | FPS: {current_fps:.1f}"
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Show actual resolution
            res_text = f"{actual_width}x{actual_height}"
            cv2.putText(
                frame,
                res_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Webcam Test", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped by user")
                break

            # Auto-stop after duration
            if elapsed >= duration:
                print(f"\nCompleted {duration}s test")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Report results (outside finally block)
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print("\nResults:")
    print(f"  Frames captured: {frame_count}")
    print(f"  Duration: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")

    if avg_fps >= 24:
        print("  ✓ FPS target met (24+ fps)")
    else:
        print(f"  ✗ FPS below target (got {avg_fps:.1f}, need 24+)")

    return avg_fps >= 24


if __name__ == "__main__":
    import sys

    # Allow command line args for camera source
    source = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    success = test_webcam(source=source)
    sys.exit(0 if success else 1)
