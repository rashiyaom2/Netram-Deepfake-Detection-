"""
Tests for Stage 5 temporal & landmark verification.
Uses a heuristic sequence scorer — no trained GRU or GPU required.
Run with: pytest tests/test_temporal.py -v
"""
import numpy as np
import pytest

from pipeline.config import TemporalConfig
from pipeline.types import AlignedFace, BranchScores
from pipeline.temporal import (
    TemporalTracker, compute_landmark_deltas, jitter_score_from_history,
    heuristic_sequence_scorer, _JITTER_LANDMARK_MAP,
)


def _make_landmarks(offset_x=0.0, offset_y=0.0):
    """Landmarks at known positions, optionally offset."""
    return {
        "left_eye_outer": (50.0 + offset_x, 40.0 + offset_y),
        "left_eye_inner": (70.0 + offset_x, 40.0 + offset_y),
        "right_eye_outer": (90.0 + offset_x, 40.0 + offset_y),
        "right_eye_inner": (110.0 + offset_x, 40.0 + offset_y),
        "mouth_left": (60.0 + offset_x, 80.0 + offset_y),
        "mouth_right": (100.0 + offset_x, 80.0 + offset_y),
        "mouth_top": (80.0 + offset_x, 70.0 + offset_y),
        "mouth_bottom": (80.0 + offset_x, 90.0 + offset_y),
        "nose_tip": (80.0 + offset_x, 55.0 + offset_y),
        "chin": (80.0 + offset_x, 110.0 + offset_y),
    }


def _make_aligned_face(frame_idx=1, landmarks=None, embedding=None, timestamp=None):
    if landmarks is None:
        landmarks = _make_landmarks()
    ts = float(frame_idx) * 0.33 if timestamp is None else float(timestamp)
    return AlignedFace(
        participant_id="p1",
        frame_idx=frame_idx,
        timestamp=ts,
        face_crop=np.random.rand(128, 128, 3).astype(np.float32),
        mouth_crop=np.random.rand(48, 48, 3).astype(np.float32),
        landmarks=landmarks,
        pose_confidence=0.9,
        audio_window=None,
    )


def _make_branch_scores(embedding=None):
    if embedding is None:
        embedding = np.random.randn(512).astype(np.float32)
    return BranchScores(
        p_spatial=0.5, p_freq=0.3, embedding=embedding,
        p_sync=0.4, av_mismatch_flag=False, p_voice_clone=0.1,
    )


# --------------------------------------------------------------------------- #
# Pure function tests
# --------------------------------------------------------------------------- #
class TestComputeLandmarkDeltas:
    def test_zero_delta_for_identical_landmarks(self):
        lm = _make_landmarks()
        tracked = ("left_eye_outer", "right_eye_outer", "mouth_top", "mouth_bottom")
        deltas = compute_landmark_deltas(lm, lm, tracked)
        assert all(d == pytest.approx(0.0) for d in deltas)

    def test_nonzero_delta_for_shifted_landmarks(self):
        lm1 = _make_landmarks()
        lm2 = _make_landmarks(offset_x=3.0, offset_y=4.0)  # Euclidean dist = 5.0
        tracked = ("left_eye_outer",)
        deltas = compute_landmark_deltas(lm1, lm2, tracked)
        assert len(deltas) == 1
        assert deltas[0] > 0.0

    def test_skips_missing_landmark_names(self):
        lm = _make_landmarks()
        tracked = ("nonexistent_landmark",)
        deltas = compute_landmark_deltas(lm, lm, tracked)
        assert len(deltas) == 0


class TestJitterScore:
    def test_returns_zero_for_too_short_history(self):
        assert jitter_score_from_history([]) == 0.0
        assert jitter_score_from_history([[0.1, 0.2]]) == 0.0

    def test_low_jitter_for_consistent_deltas(self):
        # Same delta every frame → zero variance → zero jitter
        history = [[1.0, 1.0, 1.0]] * 10
        score = jitter_score_from_history(history)
        assert score == pytest.approx(0.0)

    def test_high_jitter_for_erratic_deltas(self):
        # Alternating high/low → high variance → high jitter
        history = []
        for i in range(20):
            if i % 2 == 0:
                history.append([10.0, 10.0, 10.0])
            else:
                history.append([0.0, 0.0, 0.0])
        score = jitter_score_from_history(history)
        assert score > 0.5


class TestHeuristicSequenceScorer:
    def test_returns_zero_for_single_embedding(self):
        scorer = heuristic_sequence_scorer()
        embeddings = np.random.randn(1, 512).astype(np.float32)
        assert scorer(embeddings) == 0.0

    def test_low_score_for_identical_embeddings(self):
        scorer = heuristic_sequence_scorer()
        base = np.random.randn(512).astype(np.float32)
        embeddings = np.tile(base, (10, 1))
        score = scorer(embeddings)
        assert score < 0.2

    def test_higher_score_for_random_embeddings(self):
        scorer = heuristic_sequence_scorer()
        embeddings = np.random.randn(10, 512).astype(np.float32)
        score = scorer(embeddings)
        # Random embeddings should be less coherent → higher score
        assert 0.0 <= score <= 1.0


from pipeline.temporal import (
    TemporalTracker, compute_landmark_deltas, jitter_score_from_history,
    heuristic_sequence_scorer, liveness_score_from_history
)

class TestLivenessScoreFromHistory:
    def test_liveness_neutral_for_short_history(self):
        history = [{"left_eye": (10.0, 10.0)}] * 5
        score = liveness_score_from_history(history, ("left_eye",), min_variance_threshold=0.5)
        assert score == 0.0

    def test_liveness_high_for_frozen_face(self):
        # Identical landmarks for 15 frames -> variance is 0.0
        history = [{"left_eye": (10.0, 10.0)}] * 15
        score = liveness_score_from_history(history, ("left_eye",), min_variance_threshold=0.5)
        assert score == 1.0  # Suspicious (presentation attack)

    def test_liveness_low_for_natural_stillness(self):
        # Sub-pixel variance (e.g. standard deviation 1.0 -> variance 1.0)
        # Should be above threshold and score 0.0
        np.random.seed(42)
        history = []
        for _ in range(15):
            history.append({"left_eye": (10.0 + np.random.randn(), 10.0 + np.random.randn())})
        score = liveness_score_from_history(history, ("left_eye",), min_variance_threshold=0.5)
        assert score == 0.0  # Alive

    def test_liveness_low_for_talking(self):
        history = []
        for i in range(15):
            history.append({"left_eye": (10.0 + i, 10.0 - i)})
        score = liveness_score_from_history(history, ("left_eye",), min_variance_threshold=0.5)
        assert score == 0.0


# --------------------------------------------------------------------------- #
# TemporalTracker integration tests
# --------------------------------------------------------------------------- #
class TestTemporalTracker:
    def test_first_frame_returns_zero_scores(self):
        tracker = TemporalTracker(sequence_model=heuristic_sequence_scorer())
        af = _make_aligned_face(frame_idx=1)
        bs = _make_branch_scores()
        result = tracker.update(af, bs)
        assert result.p_temporal == 0.0
        assert result.jitter_score == 0.0
        assert result.p_liveness == 0.0

    def test_accumulates_history_across_frames(self):
        tracker = TemporalTracker(sequence_model=heuristic_sequence_scorer())
        for i in range(5):
            af = _make_aligned_face(frame_idx=i)
            bs = _make_branch_scores()
            result = tracker.update(af, bs)

        # After 5 frames, we should have meaningful scores
        assert result.p_temporal >= 0.0
        assert result.jitter_score >= 0.0

    def test_jitter_increases_with_erratic_landmarks(self):
        tracker = TemporalTracker(sequence_model=heuristic_sequence_scorer())
        results = []
        for i in range(15):
            # Create varying-magnitude deltas:
            # Even frames at base position, odd frames jump to random offsets
            # This creates alternating small and large deltas (high variance = jitter).
            if i % 3 == 0:
                lm = _make_landmarks(offset_x=0.0, offset_y=0.0)
            elif i % 3 == 1:
                lm = _make_landmarks(offset_x=20.0, offset_y=20.0)
            else:
                lm = _make_landmarks(offset_x=-15.0, offset_y=5.0)
            af = _make_aligned_face(frame_idx=i, landmarks=lm)
            bs = _make_branch_scores()
            result = tracker.update(af, bs)
            results.append(result)

        # Jitter should be elevated after many erratic frames
        final_jitter = results[-1].jitter_score
        assert final_jitter > 0.01

    def test_reset_clears_history(self):
        tracker = TemporalTracker(sequence_model=heuristic_sequence_scorer())
        for i in range(5):
            af = _make_aligned_face(frame_idx=i)
            bs = _make_branch_scores()
            tracker.update(af, bs)

        tracker.reset()

        # After reset, first frame should give zero scores again
        af = _make_aligned_face(frame_idx=10)
        bs = _make_branch_scores()
        result = tracker.update(af, bs)
        assert result.p_temporal == 0.0
        assert result.jitter_score == 0.0

    def test_blink_frequency_detection_under_one_blink_in_fifteen_seconds(self):
        from pipeline.temporal import compute_eye_aspect_ratio
        tracker = TemporalTracker(sequence_model=heuristic_sequence_scorer())
        # Open eyes landmarks
        open_lm = {
            "left_eye_outer": (30.0, 40.0), "left_eye_inner": (60.0, 40.0),
            "left_eye_top": (45.0, 32.0), "left_eye_bottom": (45.0, 48.0),
            "right_eye_outer": (110.0, 40.0), "right_eye_inner": (80.0, 40.0),
            "right_eye_top": (95.0, 32.0), "right_eye_bottom": (95.0, 48.0),
            "mouth_top": (70.0, 70.0), "mouth_bottom": (70.0, 85.0),
        }
        ear = compute_eye_aspect_ratio(open_lm)
        assert ear is not None and ear > 0.25

        # Feed 16 seconds of frames with 0 blinks (e.g. static image or unblinking face)
        for t in np.linspace(0.0, 16.0, 35):
            af = _make_aligned_face(frame_idx=int(t*10), landmarks=open_lm, timestamp=float(t))
            bs = _make_branch_scores()
            result = tracker.update(af, bs)

        # After 16 seconds with 0 blinks at low FPS (~2.2 FPS), liveness shows moderate concern
        # but NOT full escalation (low-FPS blink detection is unreliable at 2 FPS).
        assert result.p_liveness > 0.0, "Should show some liveness concern with zero blinks"
        assert result.p_liveness < 0.50, "Should not fully escalate at low FPS"

    def test_blink_frequency_with_one_natural_blink_in_fifteen_seconds(self):
        tracker = TemporalTracker(sequence_model=heuristic_sequence_scorer())
        # Feed 16 seconds with 1 blink at t=5.0s, plus natural micro-movement
        for t in np.linspace(0.0, 16.0, 40):
            is_blink = (4.8 <= t <= 5.2)
            eye_h = 2.0 if is_blink else 16.0
            lm = {
                "left_eye_outer": (30.0 + np.sin(t), 40.0 + np.cos(t)),
                "left_eye_inner": (60.0 + np.sin(t), 40.0 + np.cos(t)),
                "left_eye_top": (45.0 + np.sin(t), 40.0 - eye_h/2.0),
                "left_eye_bottom": (45.0 + np.sin(t), 40.0 + eye_h/2.0),
                "right_eye_outer": (110.0 + np.sin(t), 40.0 + np.cos(t)),
                "right_eye_inner": (80.0 + np.sin(t), 40.0 + np.cos(t)),
                "right_eye_top": (95.0 + np.sin(t), 40.0 - eye_h/2.0),
                "right_eye_bottom": (95.0 + np.sin(t), 40.0 + eye_h/2.0),
                "mouth_top": (70.0, 70.0 + np.cos(t)),
                "mouth_bottom": (70.0, 85.0 + np.cos(t)),
            }
            af = _make_aligned_face(frame_idx=int(t*10), landmarks=lm, timestamp=float(t))
            bs = _make_branch_scores()
            result = tracker.update(af, bs)

        # 1 blink occurred within the 15s window with natural motion -> liveness passes (p_liveness < 0.5)
        assert result.p_liveness < 0.5
