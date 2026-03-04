#!/usr/bin/env python3
"""
OK Sign Gesture Detection Pipeline
Main entry point for real-time gesture detection and ground position mapping.
"""

import cv2
import yaml
import time
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import List, Dict, Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from detection.sahi_detector import SAHIDetector
    from tracking.tracker import PersonTracker
    from calibration.calibrator import GroundCalibrator

    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    MODULES_AVAILABLE = False


class OKSignPipeline:
    """Main pipeline for detecting OK signs and mapping to ground positions."""

    def __init__(self, config_path="config/config.yaml"):
        """Initialize the pipeline with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.fps_history = deque(maxlen=30)
        self.frame_count = 0
        self.cap = None

        # Initialize components
        self.detector = None
        self.tracker = None
        self.calibrator = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize detection, tracking, and calibration components."""
        if not MODULES_AVAILABLE:
            print("Warning: Running in limited mode (modules unavailable)")
            return

        # Initialize detector
        model_path = self.config["model"].get("path", "yolov8n.pt")
        sahi_config = self.config.get("sahi", {})

        print("Initializing detector...")
        self.detector = SAHIDetector(
            model_path=model_path,
            slice_height=sahi_config.get("slice_height", 640),
            slice_width=sahi_config.get("slice_width", 640),
            overlap_height_ratio=sahi_config.get("overlap_height_ratio", 0.25),
            overlap_width_ratio=sahi_config.get("overlap_width_ratio", 0.25),
            confidence_threshold=self.config["inference"].get(
                "confidence_threshold", 0.3
            ),
        )

        # Initialize tracker
        tracking_config = self.config.get("tracking", {})
        print("Initializing tracker...")
        self.tracker = PersonTracker(
            track_thresh=tracking_config.get("track_thresh", 0.5),
            match_thresh=tracking_config.get("match_thresh", 0.8),
            track_buffer=tracking_config.get("track_buffer", 30),
        )

        # Initialize calibrator
        calib_config = self.config.get("calibration", {})
        print("Initializing calibrator...")
        self.calibrator = GroundCalibrator(
            calibration_dir=calib_config.get("calibration_dir", "calibration")
        )

        if not self.calibrator.is_calibrated:
            print("Warning: Calibrator not calibrated. Run calibration first!")
            print("  python -m src.calibration.calibrator --interactive")

    def initialize_camera(self):
        """Initialize webcam with configured settings."""
        cam_config = self.config["camera"]
        self.cap = cv2.VideoCapture(cam_config["source"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config["height"])
        self.cap.set(cv2.CAP_PROP_FPS, cam_config["fps"])

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
        return self.cap.isOpened()

    def process_frame(self, frame) -> List[Dict]:
        """
        Process a single frame through the pipeline.

        Returns:
            List of detection results with person_id, ground position, etc.
        """
        self.frame_count += 1
        start_time = time.time()

        results = []

        if not MODULES_AVAILABLE or self.detector is None:
            # Return empty results if modules not available
            processing_time = time.time() - start_time
            self.fps_history.append(
                1.0 / processing_time if processing_time > 0 else 30
            )
            return results

        try:
            # Step 1: Detect objects
            frame_height, frame_width = frame.shape[:2]
            detections = self.detector.detect(frame)

            # Convert to format for tracker: [x1, y1, x2, y2, conf, class_id]
            det_array = []
            for det in detections:
                det_array.append(
                    [
                        det["bbox"][0],
                        det["bbox"][1],
                        det["bbox"][2],
                        det["bbox"][3],
                        det["confidence"],
                        det["class_id"],
                    ]
                )

            # Step 2: Track persons
            tracks = self.tracker.update(det_array, (frame_height, frame_width))

            # Step 3: Find OK signs and associate with persons
            ok_signs = [
                d for d in detections if d["class_id"] == 1
            ]  # class 1 = ok_sign

            for ok_det in ok_signs:
                # Associate with person track
                track_id = self.tracker.associate_gesture(ok_det["bbox"], tracks)

                if track_id is not None:
                    # Find the track
                    track = next((t for t in tracks if t.id == track_id), None)

                    if track and self.calibrator.is_calibrated:
                        # Get ground position
                        ground_x, ground_y = self.calibrator.bbox_to_ground(track.bbox)

                        result = {
                            "person_id": track_id,
                            "ground_x": ground_x,
                            "ground_y": ground_y,
                            "timestamp": datetime.now().isoformat(),
                            "confidence": ok_det["confidence"],
                            "person_bbox": track.bbox,
                            "gesture_bbox": ok_det["bbox"],
                        }
                        results.append(result)

        except Exception as e:
            print(f"Error processing frame: {e}")

        processing_time = time.time() - start_time
        self.fps_history.append(1.0 / processing_time if processing_time > 0 else 30)

        return results

    def visualize(self, frame, results: List[Dict]) -> np.ndarray:
        """Draw detection results on frame."""
        display = frame.copy()
        height, width = display.shape[:2]

        # Draw FPS
        fps = self.get_fps()
        cv2.putText(
            display,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Draw calibration status
        if self.calibrator and self.calibrator.is_calibrated:
            calib_status = "Calibrated"
            calib_color = (0, 255, 0)
        else:
            calib_status = "Not Calibrated"
            calib_color = (0, 0, 255)

        cv2.putText(
            display,
            calib_status,
            (width - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            calib_color,
            2,
        )

        # Draw detection results
        y_offset = 60
        for result in results:
            # Draw person bbox
            if "person_bbox" in result:
                x1, y1, x2, y2 = result["person_bbox"]
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    display,
                    f"ID: {result['person_id']}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

            # Draw gesture bbox
            if "gesture_bbox" in result:
                x1, y1, x2, y2 = result["gesture_bbox"]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    display,
                    "OK",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            # Draw ground coordinates
            if "ground_x" in result and "ground_y" in result:
                coord_text = f"({result['ground_x']:.2f}, {result['ground_y']:.2f})m"
                cv2.putText(
                    display,
                    coord_text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                y_offset += 25

        # Draw instructions
        cv2.putText(
            display,
            "Press 'q' to quit",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        return display

    def get_fps(self) -> float:
        """Calculate average FPS from history."""
        if len(self.fps_history) == 0:
            return 0
        return sum(self.fps_history) / len(self.fps_history)

    def run(self):
        """Main processing loop."""
        if not self.initialize_camera():
            print("Failed to initialize camera")
            return

        print("\nStarting pipeline...")
        print("Press 'q' to quit")
        print("-" * 50)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                # Process frame
                results = self.process_frame(frame)

                # Print results
                for r in results:
                    print(
                        f"[ID {r['person_id']}] OK at ({r['ground_x']:.2f}, {r['ground_y']:.2f})m "
                        f"(conf: {r['confidence']:.2f})"
                    )

                # Visualize
                display = self.visualize(frame, results)

                # Show results
                cv2.imshow("OK Sign Detection", display)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            avg_fps = self.get_fps()
            print(f"\nPipeline stopped.")
            print(f"Processed {self.frame_count} frames")
            print(f"Average FPS: {avg_fps:.1f}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="OK Sign Detection Pipeline")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--calibrate", action="store_true", help="Run calibration first"
    )

    args = parser.parse_args()

    if args.calibrate:
        print("Running calibration...")
        calibrator = GroundCalibrator()
        success = calibrator.create_calibration_tool()
        if not success:
            print("Calibration failed or cancelled")
            return

    pipeline = OKSignPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
