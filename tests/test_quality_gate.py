"""
Tests for Stage 2 quality gating. Uses a fake face detector (dependency
injected) so these run without MediaPipe installed -- only the geometry/
image-processing math is under test here, not MediaPipe itself.
Run with: pytest tests/test_quality_gate.py -v
"""
import numpy as np
import pytest

from pipeline.config import QualityGateConfig
from pipeline.types import RawFrame
from pipeline.quality_gate import (
    QualityGate, FaceDetection, laplacian_variance, mean_brightness,
    pose_confidence_weight, estimate_pose_angles, _MODEL_POINTS_3D,
)


def _frontal_keypoints_for_shape(w, h):
    """
    Synthetic 2D projections of _MODEL_POINTS_3D under a straight-on,
    centered view -- i.e. what a perfectly frontal face's keypoints would
    look like, for testing pose estimation returns ~0deg.
    """
    cx, cy = w / 2, h / 2
    scale = 3.0
    pts = []
    for X, Y, _Z in _MODEL_POINTS_3D:
        pts.append((cx + X * scale, cy - Y * scale))
    return pts


def make_raw_frame(image_bgr):
    return RawFrame(participant_id="p1", frame_idx=1, timestamp=0.0, image_bgr=image_bgr)


# --------------------------------------------------------------------------- #
# Pure function tests
# --------------------------------------------------------------------------- #
def test_laplacian_variance_flat_image_is_near_zero():
    flat = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert laplacian_variance(flat) < 1.0


def test_laplacian_variance_sharp_edges_is_high():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, 50:] = 255  # hard vertical edge
    assert laplacian_variance(img) > 100.0


def test_mean_brightness_matches_expected():
    img = np.full((10, 10, 3), 200, dtype=np.uint8)
    assert abs(mean_brightness(img) - 200.0) < 1.0


def test_pose_confidence_full_weight_when_frontal():
    cfg = QualityGateConfig(frontal_deg_for_full_weight=20.0, max_yaw_pitch_deg=45.0)
    assert pose_confidence_weight(5.0, -3.0, cfg) == 1.0


def test_pose_confidence_zero_at_hard_cutoff():
    cfg = QualityGateConfig(frontal_deg_for_full_weight=20.0, max_yaw_pitch_deg=45.0)
    assert pose_confidence_weight(45.0, 0.0, cfg) == 0.0


def test_pose_confidence_ramps_linearly_between_bounds():
    cfg = QualityGateConfig(frontal_deg_for_full_weight=20.0, max_yaw_pitch_deg=45.0)
    # Midpoint between 20 and 45 (32.5deg) should give ~0.5 weight
    w = pose_confidence_weight(32.5, 0.0, cfg)
    assert 0.45 <= w <= 0.55


def test_estimate_pose_angles_near_zero_for_frontal_synthetic_face():
    w, h = 640, 480
    kps = _frontal_keypoints_for_shape(w, h)
    yaw, pitch = estimate_pose_angles(kps, (h, w))
    assert abs(yaw) < 10.0
    assert abs(pitch) < 10.0


# --------------------------------------------------------------------------- #
# QualityGate integration tests (fake detector injected)
# --------------------------------------------------------------------------- #
def _sharp_face_image(size=300):
    """A synthetic image with real edges so it passes the blur test."""
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return img


def test_rejects_when_no_face_detected():
    gate = QualityGate(face_detector=lambda img: [])
    result = gate.run(make_raw_frame(_sharp_face_image()))
    assert not result.passed
    assert result.reject_reason == "no_face_detected"


def test_rejects_face_too_small():
    def fake_detector(img):
        return [FaceDetection(bbox=(10, 10, 40, 40), keypoints=[(0, 0)] * 6, detection_confidence=0.9)]

    cfg = QualityGateConfig(min_face_size_px=80)
    gate = QualityGate(cfg=cfg, face_detector=fake_detector)
    result = gate.run(make_raw_frame(_sharp_face_image()))
    assert not result.passed
    assert result.reject_reason == "face_too_small"


def test_rejects_blurry_face():
    def fake_detector(img):
        return [FaceDetection(bbox=(0, 0, 200, 200), keypoints=[(0, 0)] * 6, detection_confidence=0.9)]

    flat_image = np.full((200, 200, 3), 128, dtype=np.uint8)  # no texture -> low laplacian var
    cfg = QualityGateConfig(min_face_size_px=80, laplacian_var_threshold=100.0)
    gate = QualityGate(cfg=cfg, face_detector=fake_detector)
    result = gate.run(make_raw_frame(flat_image))
    assert not result.passed
    assert result.reject_reason == "blurry"


def test_rejects_bad_lighting():
    def fake_detector(img):
        return [FaceDetection(bbox=(0, 0, 200, 200), keypoints=[(0, 0)] * 6, detection_confidence=0.9)]

    # Very dark but with enough texture to pass blur, so lighting gate is what fires.
    dark_image = (np.random.randint(0, 15, (200, 200, 3)).astype(np.uint8))
    cfg = QualityGateConfig(min_face_size_px=80, laplacian_var_threshold=1.0, min_mean_brightness=40.0)
    gate = QualityGate(cfg=cfg, face_detector=fake_detector)
    result = gate.run(make_raw_frame(dark_image))
    assert not result.passed
    assert result.reject_reason == "bad_lighting"


def test_passes_and_returns_pose_confidence_for_good_frontal_frame():
    w, h = 300, 300
    kps = _frontal_keypoints_for_shape(w, h)

    def fake_detector(img):
        return [FaceDetection(bbox=(0, 0, w, h), keypoints=kps, detection_confidence=0.95)]

    cfg = QualityGateConfig(min_face_size_px=80, laplacian_var_threshold=1.0,
                             min_mean_brightness=0.0, max_mean_brightness=255.0)
    gate = QualityGate(cfg=cfg, face_detector=fake_detector)
    result = gate.run(make_raw_frame(_sharp_face_image(w)))
    assert result.passed
    assert result.pose_confidence == 1.0  # frontal synthetic keypoints
    assert result.face_bbox == (0, 0, w, h)
