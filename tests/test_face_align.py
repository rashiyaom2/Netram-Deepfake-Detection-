"""
Tests for Stage 3 face alignment. Uses a fake landmarker (dependency
injected) with synthetic 468-point geometry so these run without
MediaPipe installed -- only the alignment math is under test here.
Run with: pytest tests/test_face_align.py -v
"""
import numpy as np
import pytest

from pipeline.config import AlignmentConfig
from pipeline.types import RawFrame, QualityResult
from pipeline.face_align import (
    FaceAligner, eye_centers, rotation_angle_deg, rotate_image_and_points,
    bbox_with_margin, crop_resize_normalize, project_points_to_output_space,
    LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX,
    MOUTH_TOP_IDX, MOUTH_BOTTOM_IDX, FACE_OVAL_IDX,
)


def make_synthetic_landmarks(w, h, tilt_deg=0.0, face_radius=100):
    """
    468 synthetic points: face-oval indices placed on an ellipse (radius
    scaled by tilt so bbox tests have real geometry), eyes/mouth placed at
    known exact positions (optionally tilted) so alignment math is checkable.
    """
    n_points = 468
    cx, cy = w / 2, h / 2
    pts = np.zeros((n_points, 2), dtype=np.float32)

    # fill everything with plausible points on an ellipse around the center
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts[:, 0] = cx + face_radius * np.cos(angles)
    pts[:, 1] = cy + face_radius * np.sin(angles)

    # overwrite face oval indices with a clean ellipse too (already close enough)
    for idx in FACE_OVAL_IDX:
        a = 2 * np.pi * idx / n_points
        pts[idx] = [cx + face_radius * np.cos(a), cy + face_radius * np.sin(a)]

    # place eyes at a known baseline separation, then apply tilt
    theta = np.radians(tilt_deg)
    eye_half_sep = 40.0
    base_left = np.array([-eye_half_sep, 0.0])
    base_right = np.array([eye_half_sep, 0.0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    left_eye = rot @ base_left + [cx, cy]
    right_eye = rot @ base_right + [cx, cy]
    for idx in LEFT_EYE_IDX:
        pts[idx] = left_eye
    for idx in RIGHT_EYE_IDX:
        pts[idx] = right_eye

    # mouth below eyes, also tilted consistently
    base_mouth_left = np.array([-20.0, 50.0])
    base_mouth_right = np.array([20.0, 50.0])
    base_mouth_top = np.array([0.0, 40.0])
    base_mouth_bottom = np.array([0.0, 60.0])
    pts[MOUTH_LEFT_IDX] = rot @ base_mouth_left + [cx, cy]
    pts[MOUTH_RIGHT_IDX] = rot @ base_mouth_right + [cx, cy]
    pts[MOUTH_TOP_IDX] = rot @ base_mouth_top + [cx, cy]
    pts[MOUTH_BOTTOM_IDX] = rot @ base_mouth_bottom + [cx, cy]

    return [tuple(p) for p in pts]


def make_raw_frame(w=400, h=400):
    image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return RawFrame(participant_id="p1", frame_idx=1, timestamp=0.0, image_bgr=image,
                     audio_window=np.zeros(10, dtype=np.float32))


# --------------------------------------------------------------------------- #
# Pure geometry tests
# --------------------------------------------------------------------------- #
def test_eye_centers_matches_known_positions():
    w, h = 400, 400
    lm = make_synthetic_landmarks(w, h, tilt_deg=0.0)
    left, right = eye_centers(lm)
    assert abs(left[1] - right[1]) < 1e-3  # untitled: same height
    assert right[0] > left[0]


def test_rotation_levels_a_tilted_eye_line():
    w, h = 400, 400
    lm = make_synthetic_landmarks(w, h, tilt_deg=15.0)
    left, right = eye_centers(lm)
    angle = rotation_angle_deg(left, right)
    center = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)

    dummy_image = np.zeros((h, w, 3), dtype=np.uint8)
    _rot_img, rotated_pts = rotate_image_and_points(dummy_image, lm, angle, center)

    rot_left, rot_right = eye_centers([tuple(p) for p in rotated_pts])
    assert abs(rot_left[1] - rot_right[1]) < 0.5  # eyes now level (within rounding)


def test_bbox_with_margin_expands_correctly():
    pts = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float32)  # 10x10 box at (10,10)
    x0, y0, x1, y1 = bbox_with_margin(pts, margin_pct=0.5, image_shape=(200, 200))
    # 10x10 box, 50% margin -> 5px on each side -> box from (5,5) to (25,25)
    assert (x0, y0, x1, y1) == (5, 5, 25, 25)


def test_bbox_with_margin_clips_to_image_bounds():
    pts = np.array([[2, 2], [8, 2], [8, 8], [2, 8]], dtype=np.float32)
    x0, y0, x1, y1 = bbox_with_margin(pts, margin_pct=2.0, image_shape=(20, 20))  # huge margin
    assert x0 == 0 and y0 == 0
    assert x1 <= 20 and y1 <= 20


def test_crop_resize_normalize_output_shape_and_range():
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    crop, sx, sy = crop_resize_normalize(image, (10, 10, 60, 60), (32, 32))
    assert crop.shape == (32, 32, 3)
    assert crop.dtype == np.float32
    assert 0.0 <= crop.min() and crop.max() <= 1.0
    assert abs(sx - 32 / 50) < 1e-6
    assert abs(sy - 32 / 50) < 1e-6


def test_project_points_to_output_space():
    pts = np.array([[15.0, 15.0], [60.0, 60.0]], dtype=np.float32)
    projected = project_points_to_output_space(pts, bbox=(10, 10, 60, 60), scale_x=2.0, scale_y=2.0)
    # (15,15) is 5px into the bbox -> *2 scale -> (10,10)
    assert np.allclose(projected[0], [10.0, 10.0])


# --------------------------------------------------------------------------- #
# FaceAligner integration tests (fake landmarker injected)
# --------------------------------------------------------------------------- #
def test_returns_none_when_quality_not_passed():
    aligner = FaceAligner(landmarker=lambda img: make_synthetic_landmarks(400, 400))
    raw = make_raw_frame()
    quality = QualityResult(passed=False, pose_confidence=0.0, blur_score=0.0, reject_reason="blurry")
    assert aligner.align(raw, quality) is None


def test_returns_none_when_no_landmarks_found():
    aligner = FaceAligner(landmarker=lambda img: None)
    raw = make_raw_frame()
    quality = QualityResult(passed=True, pose_confidence=1.0, blur_score=500.0)
    assert aligner.align(raw, quality) is None


def test_aligned_face_has_expected_shapes_and_passthrough_fields():
    w, h = 400, 400
    cfg = AlignmentConfig(margin_pct=0.25, output_size=(128, 128), mouth_crop_size=(48, 48))
    aligner = FaceAligner(cfg=cfg, landmarker=lambda img: make_synthetic_landmarks(w, h, tilt_deg=10.0))
    raw = make_raw_frame(w, h)
    quality = QualityResult(passed=True, pose_confidence=0.8, blur_score=300.0)

    result = aligner.align(raw, quality)
    assert result is not None
    assert result.face_crop.shape == (128, 128, 3)
    assert result.mouth_crop.shape == (48, 48, 3)
    assert result.pose_confidence == 0.8
    assert result.participant_id == "p1"
    assert result.audio_window is not None
    assert set(result.landmarks.keys()) == {
        "left_eye_outer", "left_eye_inner", "left_eye_top", "left_eye_bottom",
        "right_eye_outer", "right_eye_inner", "right_eye_top", "right_eye_bottom",
        "mouth_left", "mouth_right", "mouth_top", "mouth_bottom", "nose_tip", "chin",
    }


def test_aligned_landmarks_land_within_output_crop_bounds():
    w, h = 400, 400
    cfg = AlignmentConfig(margin_pct=0.3, output_size=(224, 224))
    aligner = FaceAligner(cfg=cfg, landmarker=lambda img: make_synthetic_landmarks(w, h, tilt_deg=0.0))
    raw = make_raw_frame(w, h)
    quality = QualityResult(passed=True, pose_confidence=1.0, blur_score=300.0)

    result = aligner.align(raw, quality)
    for name, (x, y) in result.landmarks.items():
        # eyes/mouth/nose/chin should land inside (or very near) the 224x224 output crop
        assert -5 <= x <= 229, f"{name} x={x} out of expected bounds"
        assert -5 <= y <= 229, f"{name} y={y} out of expected bounds"
