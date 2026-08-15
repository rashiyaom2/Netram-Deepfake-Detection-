"""
Tests for Stage 4e — Social Media & AR Beauty Filter Detector (Snapchat, IG, TikTok).
Validates natural faces, bilateral skin smoothing, anthropometric landmark warping,
digital makeup overlays, and fusion escalation.
"""
import numpy as np
import cv2
import pytest

from pipeline.branches.ar_filter_detector import ARFilterDetector, ARFilterDetectionResult
from pipeline.branches.branch_runner import BranchRunner
from pipeline.types import AlignedFace, BranchScores, TemporalResult
from pipeline.fusion import DecisionEngine, FusionConfig


class TestARFilterDetector:
    def test_natural_face_no_filter(self):
        """Natural human face with healthy skin grain and canonical landmarks returns no filter."""
        detector = ARFilterDetector()
        
        # Create synthetic natural skin crop with grain noise (high variance)
        img = np.full((224, 224, 3), 160, dtype=np.uint8)
        # Skin color in BGR: (140, 160, 200)
        img[:, :] = (140, 160, 200)
        # Add realistic skin pore noise
        noise = np.random.normal(0, 15, (224, 224, 3)).astype(np.int16)
        noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Standard canonical human landmarks
        landmarks = {
            "left_eye": (70.0, 80.0),
            "right_eye": (154.0, 80.0),
            "upper_lip": (112.0, 175.0),
            "lower_lip": (112.0, 195.0),
        }

        res = detector.detect(noisy_img, landmarks)
        assert not res.filter_detected
        assert res.filter_type == "NATURAL"
        assert res.smoothness_score < 0.40
        assert res.warp_score < 0.40

    def test_airbrush_skin_smoothing_detected(self):
        """Ultra-smooth bilateral blur / porcelain skin filter triggers beauty airbrush detection."""
        detector = ARFilterDetector()
        
        # Completely uniform porcelain skin (zero texture variance) with high-contrast eyes
        img = np.full((224, 224, 3), (145, 170, 215), dtype=np.uint8)
        # High contrast eyes
        cv2.circle(img, (70, 80), 10, (10, 10, 10), -1)
        cv2.circle(img, (154, 80), 10, (10, 10, 10), -1)

        res = detector.detect(img)
        assert res.filter_detected
        assert res.smoothness_score >= 0.55
        assert "BEAUTY" in res.filter_type or "AIRBRUSH" in res.filter_type

    def test_eye_enlargement_warping_detected(self):
        """Snapchat Anime/Bambi eye enlargement (>0.44 norm distance) triggers warping detection."""
        detector = ARFilterDetector()
        
        img = np.full((224, 224, 3), 150, dtype=np.uint8)
        # Abnormally wide enlarged eye distance (dx = 120px / 224 = 0.53)
        landmarks = {
            "left_eye": (50.0, 80.0),
            "right_eye": (174.0, 80.0),
            "upper_lip": (112.0, 175.0),
        }

        res = detector.detect(img, landmarks)
        assert res.filter_detected
        assert res.warp_score >= 0.50
        assert "WARPING" in res.filter_type or "HYBRID" in res.filter_type

    def test_synthetic_lipstick_and_blush_detected(self):
        """Heavy digital saturated lipstick & blush overlay triggers makeup detection."""
        detector = ARFilterDetector()
        
        # Skin base
        img = np.full((224, 224, 3), (140, 160, 200), dtype=np.uint8)
        
        # Add hyper-saturated digital lipstick in lip region (BGR: (30, 20, 240) = pure red)
        img[150:185, 75:150] = (20, 10, 245)
        # Add hyper-saturated cheek blush (BGR: (60, 40, 235))
        img[110:140, 30:65] = (60, 40, 235)
        img[110:140, 160:195] = (60, 40, 235)

        res = detector.detect(img)
        assert res.filter_detected
        assert res.synthetic_makeup_score >= 0.50
        assert "MAKEUP" in res.filter_type or "HYBRID" in res.filter_type

    def test_branch_runner_and_fusion_with_ar_filter(self):
        """BranchRunner aggregates AR filter results and DecisionEngine escalates to review."""
        runner = BranchRunner()
        
        # Synthetic porcelain airbrushed face
        smooth_crop = np.full((224, 224, 3), (145, 170, 215), dtype=np.uint8)
        aligned = AlignedFace(
            participant_id="charlie",
            frame_idx=1,
            timestamp=0.0,
            face_crop=smooth_crop,
            mouth_crop=np.zeros((96, 96, 3), dtype=np.float32),
            landmarks={},
            pose_confidence=1.0,
        )

        scores = runner.run(aligned)
        assert isinstance(scores, BranchScores)
        assert scores.ar_filter_detected
        assert scores.ar_filter_confidence >= 0.50

        # Test Fusion Decision Engine
        engine = DecisionEngine(FusionConfig(review_threshold=0.55))
        temporal = TemporalResult(p_temporal=0.10, jitter_score=0.05, p_liveness=0.10)

        decision = engine.decide(
            participant_id="charlie",
            frame_idx=5,
            timestamp=0.5,
            branch_scores=scores,
            temporal_result=temporal,
            pose_confidence=1.0,
        )

        assert decision.ar_filter_detected
        assert decision.p_frame >= 0.70
        assert decision.review_flag
