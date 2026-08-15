"""
Tests for Stage 4d — Phone & Screen Replay Attack Detector.
Validates nominal feeds, phone bezel presentation attacks, moiré grid patterns,
glare/light-spot rejection, and fusion escalation.
"""
import numpy as np
import cv2
import pytest

from pipeline.branches.phone_replay import PhoneReplayDetector, PhoneDetectionResult
from pipeline.branches.branch_runner import BranchRunner
from pipeline.types import AlignedFace, BranchScores, TemporalResult
from pipeline.fusion import DecisionEngine, FusionConfig


class TestPhoneReplayDetector:
    def test_nominal_frame_returns_no_phone(self):
        """Standard human face frame without phone should return phone_detected=False."""
        detector = PhoneReplayDetector()
        img = np.full((480, 640, 3), 40, dtype=np.uint8)
        # Add smooth face circle
        cv2.circle(img, (320, 240), 80, (180, 160, 140), -1)
        
        result = detector.detect(img, face_bbox=(240, 160, 160, 160))
        assert not result.phone_detected
        assert result.confidence < 0.45
        assert result.detection_source == "CLEAR"

    def test_bright_light_spot_and_glare_not_detected_as_phone(self):
        """Bright light sources, lamps, ceiling spots, and glare flares must NOT be flagged as phones."""
        detector = PhoneReplayDetector()
        img = np.full((480, 640, 3), 45, dtype=np.uint8)
        
        # Add normal human face
        cv2.circle(img, (320, 240), 75, (175, 150, 130), -1)
        
        # Add bright light source (ceiling lamp / overhead bulb)
        cv2.circle(img, (120, 80), 45, (255, 255, 255), -1)
        cv2.circle(img, (120, 80), 70, (220, 230, 255), 2)
        
        # Add bright specular glare patch on background wall
        cv2.rectangle(img, (480, 60), (560, 220), (250, 252, 255), -1)

        result = detector.detect(img, face_bbox=(245, 165, 150, 150))
        assert not result.phone_detected, f"Light source was falsely flagged: {result.details}"
        assert result.confidence < 0.50

    def test_background_window_and_door_not_detected_as_phone(self):
        """Background rectangular structures (doors, windows) not enclosing the face are ignored."""
        detector = PhoneReplayDetector()
        img = np.full((480, 640, 3), 35, dtype=np.uint8)
        
        # Background door/window with smartphone-like aspect ratio 2.0 (80x160)
        cv2.rectangle(img, (50, 50), (130, 210), (210, 210, 210), 3)
        cv2.rectangle(img, (53, 53), (127, 207), (60, 60, 60), -1)

        # Human face positioned at center (outside the background window)
        cv2.circle(img, (340, 240), 70, (180, 160, 140), -1)

        result = detector.detect(img, face_bbox=(270, 170, 140, 140))
        assert not result.phone_detected, f"Background door was falsely flagged: {result.details}"

    def test_phone_bezel_enclosing_face_detected(self):
        """High-contrast rectangular smartphone bezel enclosing face triggers detection."""
        detector = PhoneReplayDetector()
        img = np.full((480, 640, 3), 30, dtype=np.uint8)
        
        # Draw smartphone body (aspect ratio ~2.0, e.g. 200x400)
        px, py, pw, ph = 220, 40, 200, 400
        # Dark outer phone
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (10, 10, 10), -1)
        # Bright high-contrast screen bezel border
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (220, 220, 220), 4)
        # Inner screen content (natural display brightness ~80-140)
        cv2.rectangle(img, (px + 6, py + 6), (px + pw - 6, py + ph - 6), (80, 80, 80), -1)
        # Face inside phone screen
        cv2.circle(img, (320, 240), 50, (190, 170, 150), -1)

        result = detector.detect(img, face_bbox=(270, 190, 100, 100))
        assert result.phone_detected
        assert result.confidence >= 0.50
        assert result.face_enclosed
        assert "BEZEL" in result.detection_source or "HYBRID" in result.detection_source

    def test_moire_pattern_detection(self):
        """High-frequency periodic moiré grid spikes trigger spectral detection."""
        detector = PhoneReplayDetector()
        img = np.full((480, 640, 3), 50, dtype=np.uint8)
        
        # Generate synthetic moiré grid pattern in face region
        for y in range(160, 320, 4):
            img[y, 240:400] = 210
        for x in range(240, 400, 4):
            img[160:320, x] = 210

        result = detector.detect(img, face_bbox=(240, 160, 160, 160))
        assert result.phone_detected
        assert result.confidence >= 0.45

    def test_branch_runner_integration(self):
        """BranchRunner executes phone detector and returns flags in BranchScores."""
        runner = BranchRunner()
        aligned = AlignedFace(
            participant_id="alice",
            frame_idx=1,
            timestamp=0.0,
            face_crop=np.zeros((224, 224, 3), dtype=np.float32),
            mouth_crop=np.zeros((96, 96, 3), dtype=np.float32),
            landmarks={},
            pose_confidence=1.0,
        )
        
        raw_img = np.full((480, 640, 3), 30, dtype=np.uint8)
        # Smartphone bezel (200x400)
        cv2.rectangle(raw_img, (220, 40), (420, 440), (220, 220, 220), 4)
        cv2.rectangle(raw_img, (226, 46), (414, 434), (75, 75, 75), -1)

        scores = runner.run(aligned, raw_image_bgr=raw_img, face_bbox=(270, 190, 100, 100))
        assert isinstance(scores, BranchScores)
        assert scores.phone_detected
        assert scores.phone_confidence >= 0.50

    def test_fusion_escalation_on_phone_detected(self):
        """When phone_detected=True, DecisionEngine immediately escalates verdict to high threat."""
        engine = DecisionEngine(FusionConfig(review_threshold=0.55, block_threshold=0.80))
        
        scores = BranchScores(
            p_spatial=0.10,
            p_freq=0.10,
            embedding=np.zeros(512, dtype=np.float32),
            p_sync=0.5,
            phone_detected=True,
            phone_confidence=0.88,
        )
        temporal = TemporalResult(p_temporal=0.05, jitter_score=0.02, p_liveness=0.10)

        decision = engine.decide(
            participant_id="bob",
            frame_idx=10,
            timestamp=1.0,
            branch_scores=scores,
            temporal_result=temporal,
            pose_confidence=1.0,
        )
        
        assert decision.phone_detected
        assert decision.p_frame >= 0.90
        assert decision.smoothed_score >= 0.60
        assert decision.review_flag
