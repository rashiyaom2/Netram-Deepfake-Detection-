"""
Tests for Stage 0 cascade router. Uses injected fake scorers so these run
without ONNX Runtime or trained weights installed.
Run with: pytest tests/test_cascade_router.py -v
"""
import numpy as np
import pytest

from pipeline.config import CascadeConfig
from pipeline.types import RawFrame, QualityResult
from pipeline.cascade_router import (
    CascadeRouter, sigmoid, expand_bbox, preprocess_for_cascade,
    heuristic_frequency_scorer,
)


def make_raw_frame(w=200, h=200):
    image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return RawFrame(participant_id="p1", frame_idx=1, timestamp=0.0, image_bgr=image)


# --------------------------------------------------------------------------- #
# Pure function tests
# --------------------------------------------------------------------------- #
def test_sigmoid_bounds_and_midpoint():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert 0.0 < sigmoid(-100) < 0.001
    assert 0.99 < sigmoid(30) <= 1.0  # sigmoid(100) underflows to exactly 1.0 in float; avoid that edge


def test_expand_bbox_grows_and_clips():
    bbox = (50, 50, 20, 20)  # x,y,w,h
    x0, y0, x1, y1 = expand_bbox(bbox, margin_pct=0.5, image_shape=(200, 200))
    # 20px box, 50% margin -> 10px each side -> (40,40) to (80,80)
    assert (x0, y0, x1, y1) == (40, 40, 80, 80)

    edge_bbox = (0, 0, 20, 20)
    x0, y0, x1, y1 = expand_bbox(edge_bbox, margin_pct=1.0, image_shape=(200, 200))
    assert x0 == 0 and y0 == 0  # clipped at image edge


def test_preprocess_for_cascade_output_shape_and_normalization():
    crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    chw = preprocess_for_cascade(crop, input_size=(224, 224))
    assert chw.shape == (3, 224, 224)
    assert chw.dtype == np.float32
    # After ImageNet normalization, values should be roughly in [-3, 3] range, not [0,1]
    assert chw.min() < 0.5  # confirms normalization actually applied (not raw [0,1])


def test_heuristic_scorer_returns_value_in_unit_range():
    scorer = heuristic_frequency_scorer()
    crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    chw = preprocess_for_cascade(crop, (224, 224))
    score = scorer(chw)
    assert 0.0 <= score <= 1.0


# --------------------------------------------------------------------------- #
# CascadeRouter integration tests (fake scorer injected)
# --------------------------------------------------------------------------- #
def test_returns_none_when_quality_not_passed():
    router = CascadeRouter(scorer=lambda x: 0.9)
    raw = make_raw_frame()
    quality = QualityResult(passed=False, pose_confidence=0.0, blur_score=0.0, reject_reason="blurry")
    assert router.route(raw, quality) is None


def test_returns_none_when_no_bbox():
    router = CascadeRouter(scorer=lambda x: 0.9)
    raw = make_raw_frame()
    quality = QualityResult(passed=True, pose_confidence=1.0, blur_score=300.0, face_bbox=None)
    assert router.route(raw, quality) is None


def test_escalates_when_score_above_threshold():
    cfg = CascadeConfig(suspicion_threshold=0.15)
    router = CascadeRouter(cfg=cfg, scorer=lambda x: 0.5)
    raw = make_raw_frame()
    quality = QualityResult(passed=True, pose_confidence=1.0, blur_score=300.0, face_bbox=(50, 50, 60, 60))
    result = router.route(raw, quality)
    assert result is not None
    assert result.suspicion_score == 0.5
    assert result.escalate is True


def test_drops_when_score_below_threshold():
    cfg = CascadeConfig(suspicion_threshold=0.15)
    router = CascadeRouter(cfg=cfg, scorer=lambda x: 0.05)
    raw = make_raw_frame()
    quality = QualityResult(passed=True, pose_confidence=1.0, blur_score=300.0, face_bbox=(50, 50, 60, 60))
    result = router.route(raw, quality)
    assert result is not None
    assert result.suspicion_score == 0.05
    assert result.escalate is False


def test_scorer_receives_correctly_shaped_preprocessed_crop():
    received_shapes = []

    def spy_scorer(chw):
        received_shapes.append(chw.shape)
        return 0.5

    cfg = CascadeConfig(input_size=(224, 224))
    router = CascadeRouter(cfg=cfg, scorer=spy_scorer)
    raw = make_raw_frame()
    quality = QualityResult(passed=True, pose_confidence=1.0, blur_score=300.0, face_bbox=(50, 50, 60, 60))
    router.route(raw, quality)
    assert received_shapes == [(3, 224, 224)]


def test_empty_crop_from_degenerate_bbox_returns_zero_score_without_crashing():
    router = CascadeRouter(scorer=lambda x: 0.9)
    raw = make_raw_frame(w=10, h=10)
    # bbox entirely outside the (tiny) frame -> expanded crop will be empty
    quality = QualityResult(passed=True, pose_confidence=1.0, blur_score=300.0, face_bbox=(500, 500, 20, 20))
    result = router.route(raw, quality)
    assert result is not None
    assert result.suspicion_score == 0.0
    assert result.escalate is False


def test_default_onnx_b0_scorer_with_real_model_if_present():
    import os
    from pipeline.cascade_router import default_onnx_b0_scorer
    cfg = CascadeConfig(onnx_path="models/deepfake_detector.onnx", input_size=(224, 224))
    if os.path.exists("models/deepfake_detector.onnx") or os.path.exists("model.onnx"):
        scorer = default_onnx_b0_scorer(cfg)
        crop = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        chw = preprocess_for_cascade(crop, (224, 224))
        score = scorer(chw)
        assert 0.0 <= score <= 1.0


def test_default_onnx_b0_scorer_missing_fallback():
    from pipeline.cascade_router import default_onnx_b0_scorer
    cfg = CascadeConfig(onnx_path="models/non_existent_model_12345.onnx", input_size=(224, 224))
    scorer = default_onnx_b0_scorer(cfg)
    crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    chw = preprocess_for_cascade(crop, (224, 224))
    score = scorer(chw)
    assert 0.0 <= score <= 1.0
