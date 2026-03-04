#!/usr/bin/env python3
"""
Data Collection Script for OK Sign Detection
Records webcam footage at various distances and saves frames for annotation.
"""

import cv2
import os
import time
import argparse
from datetime import datetime
from pathlib import Path


class DataCollector:
    """Collect training data from webcam at specified distances."""

    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different conditions
        self.conditions = {
            "close": self.output_dir / "1m_2m",
            "medium": self.output_dir / "3m_4m",
            "far": self.output_dir / "5m_7m",
            "various": self.output_dir / "various_distances",
        }

        for path in self.conditions.values():
            path.mkdir(exist_ok=True)

    def collect_frames(self, source=0, duration=30, interval=0.5, condition="various"):
        """
        Collect frames from webcam.

        Args:
            source: Camera source (0 for default webcam)
            duration: Recording duration in seconds
            interval: Time between saved frames
            condition: Subfolder name ('close', 'medium', 'far', or 'various')
        """
        output_path = self.conditions.get(condition, self.output_dir / condition)
        output_path.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return 0

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Get actual resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n{'=' * 50}")
        print(f"Data Collection - {condition.upper()}")
        print(f"{'=' * 50}")
        print(f"Resolution: {width}x{height}")
        print(f"Duration: {duration} seconds")
        print(f"Interval: {interval} seconds")
        print(f"Expected frames: ~{int(duration / interval)}")
        print(f"\nInstructions:")
        print("- Stand at the specified distance from camera")
        print("- Make OK sign (👌) with different hand positions")
        print("- Vary hand angles and orientations")
        print("- Press 'q' to stop early")
        print(f"{'=' * 50}\n")

        frame_count = 0
        saved_count = 0
        start_time = time.time()
        last_save_time = 0

        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                frame_count += 1
                current_time = time.time() - start_time

                # Save frame at intervals
                if current_time - last_save_time >= interval:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{condition}_{timestamp}.jpg"
                    filepath = output_path / filename
                    cv2.imwrite(str(filepath), frame)
                    saved_count += 1
                    last_save_time = current_time

                    # Visual feedback
                    cv2.putText(
                        frame,
                        f"Saved: {saved_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                # Display timer
                remaining = int(duration - current_time)
                cv2.putText(
                    frame,
                    f"Time: {remaining}s",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Distance: {condition}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )

                # Instructions overlay
                cv2.putText(
                    frame,
                    "Make OK sign (👌)",
                    (10, height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Press 'q' to stop",
                    (10, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )

                cv2.imshow("Data Collection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        print(f"\n{'=' * 50}")
        print(f"Collection Complete!")
        print(f"Frames captured: {frame_count}")
        print(f"Frames saved: {saved_count}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Saved to: {output_path}")
        print(f"{'=' * 50}\n")

        return saved_count

    def collect_all_distances(self, source=0, duration_per_distance=30):
        """Collect data at multiple distances."""
        print("\n" + "=" * 50)
        print("MULTI-DISTANCE DATA COLLECTION")
        print("=" * 50)
        print("\nThis will collect data at 3 distances:")
        print("1. CLOSE (1-2m) - Large hand in frame")
        print("2. MEDIUM (3-4m) - Medium hand size")
        print("3. FAR (5-7m) - Small hand at distance")
        print("\nPress ENTER to start, or Ctrl+C to cancel")
        input()

        total_frames = 0

        # Close distance
        print("\n[1/3] Position yourself 1-2 meters from camera")
        print("Press ENTER when ready...")
        input()
        total_frames += self.collect_frames(
            source, duration_per_distance, condition="close"
        )

        # Medium distance
        print("\n[2/3] Position yourself 3-4 meters from camera")
        print("Press ENTER when ready...")
        input()
        total_frames += self.collect_frames(
            source, duration_per_distance, condition="medium"
        )

        # Far distance
        print("\n[3/3] Position yourself 5-7 meters from camera")
        print("Press ENTER when ready...")
        input()
        total_frames += self.collect_frames(
            source, duration_per_distance, condition="far"
        )

        print("\n" + "=" * 50)
        print("ALL COLLECTIONS COMPLETE!")
        print(f"Total frames saved: {total_frames}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 50)

        return total_frames


def main():
    parser = argparse.ArgumentParser(description="Collect OK sign training data")
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="Collection mode: single distance or all distances",
    )
    parser.add_argument(
        "--distance",
        choices=["close", "medium", "far", "various"],
        default="various",
        help="Distance category for single mode",
    )
    parser.add_argument(
        "--duration", type=int, default=30, help="Duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between saved frames (default: 0.5)",
    )
    parser.add_argument("--output", default="data/raw", help="Output directory")

    args = parser.parse_args()

    collector = DataCollector(args.output)

    if args.mode == "multi":
        collector.collect_all_distances(duration_per_distance=args.duration)
    else:
        collector.collect_frames(
            duration=args.duration, interval=args.interval, condition=args.distance
        )


if __name__ == "__main__":
    main()
