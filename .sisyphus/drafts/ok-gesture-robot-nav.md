# Draft: OK Sign Gesture Detection + Ground Robot Navigation

## Confirmed Requirements

### Core Objective
Build a proof-of-concept system that detects an "OK" hand gesture from a webcam feed, identifies the person's ground position, and outputs coordinates for a ground robot to navigate to.

### Technical Specifications

**Hardware:**
- Current: Ryzen 7 8845HS (laptop)
- Target: Intel N100 (edge device)
- Camera: Laptop webcam (720p @ 30fps)
- Minimum FPS: 24

**Gesture:**
- OK sign (👌)
- Single gesture detection (not multi-class)
- Detection range: 1-7 meters from camera

**Architecture:** Approach 2 (End-to-End YOLO)
- SAHI tiling for small object detection
- Single YOLO model detecting person + OK gesture
- ByteTrack for person tracking
- 4-point calibration for ground plane mapping

**Output Format:**
```json
{
  "person_id": "track_001",
  "ground_x": 2.5,
  "ground_y": 3.2,
  "timestamp": "2026-03-04T10:30:00Z",
  "confidence": 0.94
}
```

### Future Vision (Not in POC scope)

**Multi-Robot Coordination:**
- Camera sends coordinates to ground robot(s)
- Robot navigates to person location
- Potential multi-robot scenarios

**Obstacle Detection:**
- Mark obstructions at ground level
- Help robots avoid obstacles
- Camera as central perception hub

**Architecture Considerations for Future:**
- Real-time coordinate streaming (WebSocket/MQTT?)
- Multiple robot coordination
- Dynamic obstacle mapping
- Path planning integration
- Multi-camera setup for larger areas

### Open Questions for Future Planning

1. **Camera Mounting:**
   - Static position or movable?
   - Overhead view or eye-level?
   - Height from ground?

2. **Ground Plane:**
   - Flat floor or uneven terrain?
   - Indoor (office/home) or outdoor?
   - Size of operating area?

3. **Robot Integration:**
   - Communication protocol (HTTP, MQTT, ROS?)
   - Coordinate frame (meters from camera? absolute GPS?)
   - Real-time requirements (latency tolerance?)

4. **Obstacle Detection:**
   - Static obstacles (furniture) or dynamic (people walking)?
   - Just detection or need classification (chair vs table)?
   - Persistent map or frame-by-frame?

5. **Training Data:**
   - Public datasets for OK sign?
   - Need to collect own footage?
   - Synthetic data generation needed?

### POC Success Criteria

- [ ] Detect OK sign at 1-7m range with >80% accuracy
- [ ] Output ground coordinates within ±30cm error
- [ ] Maintain 24+ FPS on Ryzen 7
- [ ] Track person ID across frames
- [ ] Calibrate with 4-point method
- [ ] Export coordinates + ID + timestamp

### Notes

- Webcam (720p) is lower res than CCTV - may need higher SAHI overlap
- Wider FOV than CCTV cameras - different distortion profile
- Laptop webcam easier for rapid prototyping
- Future obstacle detection may need separate model or depth estimation
- Consider ROS2 integration for robot communication standard

---

**Status:** Requirements gathering phase. Work plan to be created when details finalized.
**Last Updated:** 2026-03-04
