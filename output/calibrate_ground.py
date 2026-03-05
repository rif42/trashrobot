#!/usr/bin/env python3
"""
Ground Plane Calibration Tool

Calibrates the ground plane by clicking 4 corners of a rectangle on the floor.
Saves homography matrix for mapping image coordinates to ground coordinates.

Usage:
    python calibrate_ground.py [--camera 0] [--output calibration.yml]

Controls:
    Click 4 corners of a rectangle on the floor (clockwise from top-left)
    'r' - Reset points
    's' - Save calibration (after 4 points selected)
    'q' - Quit without saving
"""

import cv2
import numpy as np
import argparse
import sys


class CalibrationTool:
    def __init__(self, camera_index=0, output_file="calibration.yml"):
        self.camera_index = camera_index
        self.output_file = output_file
        self.points = []
        self.frame = None
        self.homography = None
        self.birds_eye_view = None

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select calibration points"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                print(f"Point {len(self.points)}: ({x}, {y})")

                if len(self.points) == 4:
                    print("\n4 points selected! Enter dimensions when prompted.")
                    self.compute_homography()

    def draw_calibration_ui(self, frame):
        """Draw UI elements on the frame"""
        display = frame.copy()

        # Draw instructions
        instructions = [
            "Ground Plane Calibration",
            "Click 4 corners of floor rectangle (clockwise from top-left)",
            f"Points selected: {len(self.points)}/4",
            "'r' = Reset | 's' = Save | 'q' = Quit",
        ]

        y_offset = 30
        for i, text in enumerate(instructions):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(
                display,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Draw clicked points and lines
        for i, point in enumerate(self.points):
            # Draw point
            cv2.circle(display, point, 5, (0, 0, 255), -1)
            # Draw point number
            cv2.putText(
                display,
                str(i + 1),
                (point[0] + 10, point[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # Draw line to next point
            if i > 0:
                cv2.line(display, self.points[i - 1], point, (0, 255, 0), 2)

        # Close the rectangle if 4 points selected
        if len(self.points) == 4:
            cv2.line(display, self.points[3], self.points[0], (0, 255, 0), 2)
            # Fill rectangle with transparency
            overlay = display.copy()
            pts = np.array(self.points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)

        return display

    def compute_homography(self):
        """Compute homography matrix from selected points"""
        if len(self.points) != 4:
            print("Error: Need exactly 4 points")
            return False

        # Get real-world dimensions from user
        print("\n" + "=" * 50)
        print("Enter real-world dimensions of the rectangle:")
        try:
            width = float(
                input("Width (in your preferred units, e.g., cm or meters): ")
            )
            height = float(input("Height (in your preferred units): "))
        except ValueError:
            print("Error: Invalid input. Please enter numbers.")
            return False

        # Define source (image) and destination (world) points
        # Points are in order: TL, TR, BR, BL
        src_points = np.array(self.points, dtype=np.float32)
        dst_points = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        )

        # Compute homography
        self.homography, status = cv2.findHomography(src_points, dst_points)

        if self.homography is None or status is None:
            print("Error: Could not compute homography. Points may be collinear.")
            return False

        print(f"\nHomography matrix computed successfully!")
        print(f"Matrix:\n{self.homography}")

        # Generate preview
        self.show_preview()

        return True

    def show_preview(self):
        """Show bird's eye view preview"""
        if self.homography is None or self.frame is None:
            return

        # Warp frame to bird's eye view
        h, w = self.frame.shape[:2]
        self.birds_eye_view = cv2.warpPerspective(self.frame, self.homography, (w, h))

        # Add label
        cv2.putText(
            self.birds_eye_view,
            "Bird's Eye View Preview (Press 's' to save)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    def save_calibration(self):
        """Save homography matrix to file"""
        if self.homography is None:
            print("Error: No homography computed. Select 4 points first.")
            return False

        try:
            # Use OpenCV FileStorage for YAML format
            fs = cv2.FileStorage(self.output_file, cv2.FILE_STORAGE_WRITE)
            fs.write("homography", self.homography)
            fs.write("width", float(self.homography.shape[1]))
            fs.write("height", float(self.homography.shape[0]))
            fs.release()

            print(f"\nCalibration saved to: {self.output_file}")
            print("You can now run inference with --calibration parameter")
            return True

        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False

    def reset(self):
        """Reset calibration points"""
        self.points = []
        self.homography = None
        self.birds_eye_view = None
        print("\nPoints reset. Click 4 new corners.")

    def run(self):
        """Main calibration loop"""
        # Open camera
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False

        # Set buffer size to 1 for real-time response
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Create window and set mouse callback
        cv2.namedWindow("Ground Calibration")
        cv2.setMouseCallback("Ground Calibration", self.mouse_callback)

        print("=" * 50)
        print("Ground Plane Calibration Tool")
        print("=" * 50)
        print("\nInstructions:")
        print("1. Click 4 corners of a rectangle on the floor")
        print("   Order: Top-Left → Top-Right → Bottom-Right → Bottom-Left")
        print("2. Enter real-world dimensions when prompted")
        print("3. Press 's' to save calibration")
        print("4. Press 'q' to quit")
        print("5. Press 'r' to reset points")
        print("=" * 50 + "\n")

        try:
            while True:
                ret, self.frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break

                # Draw UI
                display = self.draw_calibration_ui(self.frame)
                cv2.imshow("Ground Calibration", display)

                # Show bird's eye preview if available
                if self.birds_eye_view is not None:
                    cv2.imshow("Bird's Eye Preview", self.birds_eye_view)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("\nQuitting without saving...")
                    break
                elif key == ord("r"):
                    self.reset()
                elif key == ord("s"):
                    if len(self.points) == 4:
                        if self.save_calibration():
                            print("\nCalibration complete! Press 'q' to exit.")
                    else:
                        print("\nError: Select 4 points before saving")

        finally:
            cap.release()
            cv2.destroyAllWindows()

        return True


def main():
    parser = argparse.ArgumentParser(description="Ground Plane Calibration Tool")
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calibration.yml",
        help="Output calibration file (default: calibration.yml)",
    )

    args = parser.parse_args()

    tool = CalibrationTool(camera_index=args.camera, output_file=args.output)
    success = tool.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
