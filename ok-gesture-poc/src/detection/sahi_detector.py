#!/usr/bin/env python3
"""
SAHI Integration Module
Handles small object detection using slicing-aided inference.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class SAHIDetector:
    """Wrapper for SAHI small object detection."""

    def __init__(
        self,
        model_path: str,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.25,
        overlap_width_ratio: float = 0.25,
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize SAHI detector.

        Args:
            model_path: Path to YOLO model
            slice_height: Height of image slices
            slice_width: Width of image slices
            overlap_height_ratio: Vertical overlap between slices
            overlap_width_ratio: Horizontal overlap between slices
            confidence_threshold: Detection confidence threshold
        """
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.confidence_threshold = confidence_threshold

        # Try to import SAHI
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction

            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device="cuda:0",  # Will fall back to CPU if CUDA unavailable
            )
            self.get_sliced_prediction = get_sliced_prediction
            self.has_sahi = True
            print("SAHI initialized successfully")

        except ImportError:
            print("Warning: SAHI not installed, using standard YOLO")
            self.has_sahi = False
            self.detection_model = None

            # Try to load YOLO directly as fallback
            try:
                from ultralytics import YOLO

                self.yolo_model = YOLO(model_path)
                print("Using standard YOLO as fallback")
            except ImportError:
                self.yolo_model = None
                print("Warning: Neither SAHI nor YOLO available")

    def detect(self, frame: np.ndarray, use_sahi: Optional[bool] = None) -> List[Dict]:
        """
        Run detection on frame.

        Args:
            frame: Input image (BGR format)
            use_sahi: Force SAHI on/off (None = auto)

        Returns:
            List of detections: [{bbox, confidence, class_id, class_name}]
        """
        if use_sahi is None:
            use_sahi = self.has_sahi

        if use_sahi and self.has_sahi:
            return self._detect_sahi(frame)
        else:
            return self._detect_standard(frame)

    def _detect_sahi(self, frame: np.ndarray) -> List[Dict]:
        """Detect using SAHI slicing."""
        result = self.get_sliced_prediction(
            image=frame,
            detection_model=self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            postprocess_type="NMS",
            postprocess_match_threshold=0.5,
            verbose=0,
        )

        detections = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()  # [x1, y1, x2, y2]
            detection = {
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "confidence": float(pred.score.value),
                "class_id": int(pred.category.id),
                "class_name": pred.category.name,
            }
            detections.append(detection)

        return detections

    def _detect_standard(self, frame: np.ndarray) -> List[Dict]:
        """Detect using standard YOLO without slicing."""
        if self.yolo_model is None:
            return []

        results = self.yolo_model(frame, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                detection = {
                    "bbox": bbox.tolist(),
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": result.names[cls_id],
                }
                detections.append(detection)

        return detections

    def detect_with_adaptive_sahi(
        self, frame: np.ndarray, person_bboxes: List[List[int]]
    ) -> List[Dict]:
        """
        Adaptive SAHI: Only use SAHI for small/distant objects.

        Args:
            frame: Input image
            person_bboxes: List of person bounding boxes for distance estimation

        Returns:
            Detections
        """
        # Estimate if we need SAHI based on person sizes
        needs_sahi = False
        for bbox in person_bboxes:
            height = bbox[3] - bbox[1]
            # If person is small in frame, likely far away
            if height < frame.shape[0] * 0.3:  # Less than 30% of frame height
                needs_sahi = True
                break

        return self.detect(frame, use_sahi=needs_sahi)


def benchmark_sahi(model_path: str, test_image: str):
    """Benchmark SAHI vs standard detection."""
    import cv2
    import time

    frame = cv2.imread(test_image)
    if frame is None:
        print(f"Cannot load image: {test_image}")
        return

    print("=" * 60)
    print("SAHI Benchmark")
    print("=" * 60)
    print(f"Image size: {frame.shape}")

    # Standard detection
    print("\n1. Standard YOLO...")
    detector_std = SAHIDetector(model_path)

    start = time.time()
    for _ in range(10):
        dets_std = detector_std.detect(frame, use_sahi=False)
    time_std = (time.time() - start) / 10

    print(f"   Time: {time_std * 1000:.1f}ms")
    print(f"   Detections: {len(dets_std)}")

    # SAHI detection
    if detector_std.has_sahi:
        print("\n2. SAHI Detection...")

        start = time.time()
        for _ in range(10):
            dets_sahi = detector_std.detect(frame, use_sahi=True)
        time_sahi = (time.time() - start) / 10

        print(f"   Time: {time_sahi * 1000:.1f}ms")
        print(f"   Detections: {len(dets_sahi)}")
        print(f"   Overhead: {((time_sahi / time_std - 1) * 100):.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        benchmark_sahi(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python sahi_detector.py <model_path> <test_image>")
