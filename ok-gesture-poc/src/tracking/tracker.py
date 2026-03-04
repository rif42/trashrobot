#!/usr/bin/env python3
"""
Person Tracking Module
Integrates ByteTrack for person ID persistence across frames.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Track:
    """Represents a tracked person."""

    id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    age: int  # Frames since last detection
    history: List[Tuple[int, int, int, int]]  # Recent bounding boxes


class PersonTracker:
    """Wrapper for ByteTrack person tracking."""

    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=30):
        """
        Initialize tracker.

        Args:
            track_thresh: Detection confidence threshold
            match_thresh: Matching threshold for track association
            track_buffer: Frames to keep lost tracks
        """
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer

        # Try to import ByteTrack
        try:
            from boxmot import BYTETracker

            self.tracker = BYTETracker(
                track_thresh=track_thresh,
                match_thresh=match_thresh,
                track_buffer=track_buffer,
            )
            self.has_tracker = True
            print("ByteTrack initialized successfully")
        except ImportError:
            print("Warning: boxmot not installed, using simple tracking")
            self.has_tracker = False
            self.tracks = {}
            self.next_id = 1

    def update(
        self, detections: List[List[float]], frame_shape: Tuple[int, int]
    ) -> List[Track]:
        """
        Update tracks with new detections.

        Args:
            detections: List of [x1, y1, x2, y2, confidence, class_id]
            frame_shape: (height, width) of frame

        Returns:
            List of Track objects
        """
        if not self.has_tracker:
            return self._simple_update(detections)

        # Filter for person class only (class_id = 0)
        person_dets = [d for d in detections if int(d[5]) == 0]

        if len(person_dets) == 0:
            # Update tracker with empty detections
            tracks = self.tracker.update(np.array([]), frame_shape)
        else:
            # Format for ByteTrack: [x1, y1, x2, y2, confidence, class_id]
            dets_array = np.array(person_dets)
            tracks = self.tracker.update(dets_array, frame_shape)

        # Convert to Track objects
        track_objects = []
        for t in tracks:
            # ByteTrack format: [x1, y1, x2, y2, track_id, confidence, class, ...]
            track_id = int(t[4])
            bbox = (int(t[0]), int(t[1]), int(t[2]), int(t[3]))
            confidence = float(t[5])

            track = Track(
                id=track_id, bbox=bbox, confidence=confidence, age=0, history=[bbox]
            )
            track_objects.append(track)

        return track_objects

    def _simple_update(self, detections: List[List[float]]) -> List[Track]:
        """Simple IOU-based tracking when ByteTrack unavailable."""
        person_dets = [d for d in detections if int(d[5]) == 0]

        # Simple greedy matching based on IOU
        new_tracks = {}
        used_detections = set()

        # Match existing tracks to new detections
        for track_id, track in self.tracks.items():
            best_iou = 0
            best_det_idx = -1

            for i, det in enumerate(person_dets):
                if i in used_detections:
                    continue

                det_bbox = (int(det[0]), int(det[1]), int(det[2]), int(det[3]))
                iou = self._compute_iou(track.bbox, det_bbox)

                if iou > best_iou and iou > 0.3:  # IOU threshold
                    best_iou = iou
                    best_det_idx = i

            if best_det_idx >= 0:
                # Update track
                det = person_dets[best_det_idx]
                new_bbox = (int(det[0]), int(det[1]), int(det[2]), int(det[3]))
                new_tracks[track_id] = Track(
                    id=track_id,
                    bbox=new_bbox,
                    confidence=float(det[4]),
                    age=0,
                    history=track.history[-10:] + [new_bbox],
                )
                used_detections.add(best_det_idx)
            elif track.age < self.track_buffer:
                # Keep lost track
                new_tracks[track_id] = Track(
                    id=track_id,
                    bbox=track.bbox,
                    confidence=track.confidence,
                    age=track.age + 1,
                    history=track.history,
                )

        # Create new tracks for unmatched detections
        for i, det in enumerate(person_dets):
            if i not in used_detections:
                new_bbox = (int(det[0]), int(det[1]), int(det[2]), int(det[3]))
                new_tracks[self.next_id] = Track(
                    id=self.next_id,
                    bbox=new_bbox,
                    confidence=float(det[4]),
                    age=0,
                    history=[new_bbox],
                )
                self.next_id += 1

        self.tracks = new_tracks
        return list(new_tracks.values())

    def _compute_iou(self, box1: Tuple[int, ...], box2: Tuple[int, ...]) -> float:
        """Compute IOU between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def associate_gesture(
        self, gesture_bbox: Tuple[int, ...], tracks: List[Track]
    ) -> Optional[int]:
        """
        Associate a gesture detection with a person track.

        Args:
            gesture_bbox: [x1, y1, x2, y2] of gesture
            tracks: List of active tracks

        Returns:
            Track ID or None if no match
        """
        best_track_id = None
        best_iou = 0.0

        for track in tracks:
            # Check if gesture is inside person bbox
            if self._is_inside(gesture_bbox, track.bbox):
                # Or use IOU if partially overlapping
                iou = self._compute_iou(gesture_bbox, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track.id

        return best_track_id

    def _is_inside(self, inner: Tuple[int, ...], outer: Tuple[int, ...]) -> bool:
        """Check if inner box is inside outer box."""
        ix1, iy1, ix2, iy2 = inner
        ox1, oy1, ox2, oy2 = outer

        center_x = (ix1 + ix2) / 2
        center_y = (iy1 + iy2) / 2

        return (ox1 <= center_x <= ox2) and (oy1 <= center_y <= oy2)


def test_tracker():
    """Test the tracker with dummy data."""
    tracker = PersonTracker()

    # Simulate detections over 3 frames
    frames = [
        # Frame 1: 2 people
        [[100, 100, 200, 300, 0.9, 0], [300, 100, 400, 300, 0.85, 0]],
        # Frame 2: Same 2 people, slightly moved
        [[105, 105, 205, 305, 0.9, 0], [305, 105, 405, 305, 0.85, 0]],
        # Frame 3: 1 person left
        [[110, 110, 210, 310, 0.9, 0]],
    ]

    print("Testing tracker...")
    for i, detections in enumerate(frames):
        tracks = tracker.update(detections, (480, 640))
        print(f"Frame {i + 1}: {len(tracks)} tracks")
        for t in tracks:
            print(f"  ID {t.id}: bbox={t.bbox}, conf={t.confidence:.2f}")

    print("\nTest complete!")


if __name__ == "__main__":
    test_tracker()
