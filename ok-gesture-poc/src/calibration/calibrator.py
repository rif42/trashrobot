#!/usr/bin/env python3
"""
Ground Plane Calibration Module
Maps image coordinates to real-world ground coordinates using homography.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Optional


class GroundCalibrator:
    """Calibrates camera and maps image to ground coordinates."""

    def __init__(self, calibration_dir="calibration"):
        self.calibration_dir = Path(calibration_dir)
        self.calibration_dir.mkdir(exist_ok=True)

        self.H = None  # Homography matrix
        self.world_points = None
        self.image_points = None
        self.is_calibrated = False

        # Try to load existing calibration
        self._load_calibration()

    def _load_calibration(self):
        """Load existing calibration if available."""
        h_file = self.calibration_dir / "homography_matrix.npy"
        world_file = self.calibration_dir / "world_points.json"

        if h_file.exists() and world_file.exists():
            self.H = np.load(h_file)
            with open(world_file, "r") as f:
                data = json.load(f)
                self.world_points = np.array(data["world_points"], dtype=np.float32)
                self.image_points = np.array(data["image_points"], dtype=np.float32)
            self.is_calibrated = True
            print(f"Loaded calibration from {self.calibration_dir}")

    def calibrate(
        self,
        image_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]],
        image_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Calibrate using 4+ corresponding points.

        Args:
            image_points: List of (x, y) in image coordinates
            world_points: List of (X, Y) in world coordinates (meters)
            image_shape: (height, width) for visualization
        """
        if len(image_points) < 4 or len(world_points) < 4:
            raise ValueError("Need at least 4 points for calibration")

        if len(image_points) != len(world_points):
            raise ValueError("Image and world points must have same count")

        # Convert to numpy arrays
        self.image_points = np.array(image_points, dtype=np.float32)
        self.world_points = np.array(world_points, dtype=np.float32)

        # Compute homography
        self.H, mask = cv2.findHomography(self.image_points, self.world_points)

        if self.H is None:
            raise ValueError("Failed to compute homography")

        self.is_calibrated = True

        # Save calibration
        self._save_calibration()

        print(f"Calibration complete!")
        print(f"Homography matrix:\n{self.H}")

        # Validate
        if len(image_points) >= 4:
            errors = self._validate_calibration()
            print(
                f"Reprojection error: {np.mean(errors):.3f}m (max: {np.max(errors):.3f}m)"
            )

    def _save_calibration(self):
        """Save calibration to files."""
        h_file = self.calibration_dir / "homography_matrix.npy"
        world_file = self.calibration_dir / "world_points.json"

        np.save(h_file, self.H)

        with open(world_file, "w") as f:
            json.dump(
                {
                    "world_points": self.world_points.tolist(),
                    "image_points": self.image_points.tolist(),
                },
                f,
                indent=2,
            )

        print(f"Saved calibration to {self.calibration_dir}")

    def _validate_calibration(self) -> List[float]:
        """Validate by reprojecting calibration points."""
        errors = []
        for img_pt, world_pt in zip(self.image_points, self.world_points):
            projected = self.image_to_ground((img_pt[0], img_pt[1]))
            error = np.sqrt(
                (projected[0] - world_pt[0]) ** 2 + (projected[1] - world_pt[1]) ** 2
            )
            errors.append(error)
        return errors

    def image_to_ground(self, image_point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert image point to ground coordinates.

        Args:
            image_point: (x, y) in image coordinates

        Returns:
            (X, Y) in world coordinates (meters)
        """
        if not self.is_calibrated:
            raise RuntimeError("Calibrator not calibrated. Run calibrate() first.")

        # Convert to homogeneous coordinates
        point = np.array([[image_point[0], image_point[1]]], dtype=np.float32)
        point = np.array([point])

        # Apply homography
        world_point = cv2.perspectiveTransform(point, self.H)

        return (float(world_point[0][0][0]), float(world_point[0][0][1]))

    def bbox_to_ground(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Get ground position from person bounding box.
        Uses bottom-center of bbox (feet position).

        Args:
            bbox: (x1, y1, x2, y2) bounding box

        Returns:
            (X, Y) ground coordinates
        """
        x1, y1, x2, y2 = bbox
        # Bottom center of bbox
        feet_x = (x1 + x2) / 2
        feet_y = y2

        return self.image_to_ground((feet_x, feet_y))

    def create_calibration_tool(self, source=0):
        """
        Interactive calibration tool using webcam.

        Args:
            source: Camera source (0 for default webcam)
        """
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        clicked_points = []
        world_points_input = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
                clicked_points.append((x, y))
                print(f"Point {len(clicked_points)}: image=({x}, {y})")

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_callback)

        print("\n" + "=" * 60)
        print("Ground Plane Calibration Tool")
        print("=" * 60)
        print("\nInstructions:")
        print("1. Place 4 markers on the ground in a rectangle pattern")
        print("2. Measure the real-world distances between markers")
        print("3. Click on each marker in the camera view")
        print("4. Enter the real-world coordinates when prompted")
        print("\nPress 'q' to quit, 'r' to reset points")
        print("=" * 60 + "\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                display = frame.copy()
                height, width = display.shape[:2]

                # Draw clicked points
                for i, (x, y) in enumerate(clicked_points):
                    cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
                    cv2.putText(
                        display,
                        str(i + 1),
                        (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                # Draw instructions
                status = f"Points: {len(clicked_points)}/4"
                cv2.putText(
                    display,
                    status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

                if len(clicked_points) == 4:
                    cv2.putText(
                        display,
                        "Press 'c' to calibrate",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

                cv2.putText(
                    display,
                    "Click 4 ground points",
                    (10, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )

                cv2.imshow("Calibration", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    clicked_points.clear()
                    world_points_input.clear()
                    print("Reset points")
                elif key == ord("c") and len(clicked_points) == 4:
                    # Get world coordinates
                    print("\nEnter world coordinates (in meters):")
                    print("Format: x,y for each point")
                    print("Example: 0,0  2,0  2,3  0,3  (for 2m x 3m rectangle)")

                    cap.release()
                    cv2.destroyAllWindows()

                    coords_input = input("Enter 4 coordinates: ").strip()
                    try:
                        world_points_input = []
                        for coord in coords_input.split():
                            x, y = map(float, coord.split(","))
                            world_points_input.append((x, y))

                        if len(world_points_input) == 4:
                            self.calibrate(
                                clicked_points, world_points_input, (height, width)
                            )
                            print("\nCalibration successful!")
                            return True
                        else:
                            print("Error: Need exactly 4 coordinates")
                            return False
                    except ValueError as e:
                        print(f"Error parsing coordinates: {e}")
                        return False

        finally:
            cap.release()
            cv2.destroyAllWindows()

        return False


def test_calibrator():
    """Test calibration with dummy data."""
    calibrator = GroundCalibrator()

    # Example: 4 points of a 2x3m rectangle
    image_points = [
        (320, 240),  # Top-left in image
        (960, 240),  # Top-right
        (960, 480),  # Bottom-right
        (320, 480),  # Bottom-left
    ]

    world_points = [
        (0, 0),  # Top-left in world
        (2, 0),  # Top-right (2m wide)
        (2, 3),  # Bottom-right (3m deep)
        (0, 3),  # Bottom-left
    ]

    print("Testing calibration...")
    calibrator.calibrate(image_points, world_points)

    # Test projection
    test_point = (640, 360)  # Center of image
    ground = calibrator.image_to_ground(test_point)
    print(f"\nTest: image{test_point} -> ground{ground}")

    # Test bbox to ground
    bbox = (300, 200, 400, 500)  # Person bbox
    ground_pos = calibrator.bbox_to_ground(bbox)
    print(f"Person bbox {bbox} -> ground position {ground_pos}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        calibrator = GroundCalibrator()
        calibrator.create_calibration_tool()
    else:
        test_calibrator()
