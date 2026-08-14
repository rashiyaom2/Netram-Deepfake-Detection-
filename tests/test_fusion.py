"""
Tests for Stage 6 decision engine / adaptive fusion.
Uses the heuristic fusion model — no trained sklearn model required.
Run with: pytest tests/test_fusion.py -v
"""
import numpy as np
import pytest

from pipeline.config import FusionConfig
from pipeline.types import BranchScores, TemporalResult, FrameDecision, FusionInput
from pipeline.fusion import (
    DecisionEngine, heuristic_fusion, build_fusion_feature_vector,
    exponential_smooth,
)


def _make_branch_scores(p_spatial=0.5, p_freq=0.3, p_sync=0.4):
    return BranchScores(
        p_spatial=p_spatial,
        p_freq=p_freq,
        embedding=np.zeros(512, dtype=np.float32),
        p_sync=p_sync,
        av_mismatch_flag=False,
        p_voice_clone=0.1,
    )


def _make_temporal_result(p_temporal=0.3, jitter=0.2, p_liveness=0.0):
    return TemporalResult(p_temporal=p_temporal, jitter_score=jitter, p_liveness=p_liveness)


# --------------------------------------------------------------------------- #
# Pure function tests
# --------------------------------------------------------------------------- #
class TestBuildFusionFeatureVector:
    def test_produces_7_element_vector(self):
        fi = FusionInput(p_spatial=0.5, p_freq=0.3, p_temporal=0.2,
                          p_sync=0.4, jitter=0.1, pose_confidence=0.9, p_liveness=0.0)
        vec = build_fusion_feature_vector(fi)
        assert vec.shape == (7,)
        assert vec.dtype == np.float32

    def test_none_sync_replaced_with_neutral(self):
        fi = FusionInput(p_spatial=0.5, p_freq=0.3, p_temporal=0.2,
                          p_sync=None, jitter=0.1, pose_confidence=0.9, p_liveness=0.0)
        vec = build_fusion_feature_vector(fi)
        assert vec[3] == 0.5  # neutral prior for missing sync

    def test_preserves_input_values(self):
        fi = FusionInput(p_spatial=0.1, p_freq=0.2, p_temporal=0.3,
                          p_sync=0.4, jitter=0.5, pose_confidence=0.6, p_liveness=0.7)
        vec = build_fusion_feature_vector(fi)
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
        assert np.allclose(vec, expected)


class TestExponentialSmooth:
    def test_alpha_1_gives_current_score(self):
        assert exponential_smooth(0.9, 0.1, alpha=1.0) == pytest.approx(0.9)

    def test_alpha_0_gives_previous_score(self):
        assert exponential_smooth(0.9, 0.1, alpha=0.0) == pytest.approx(0.1)

    def test_alpha_05_gives_average(self):
        assert exponential_smooth(0.8, 0.2, alpha=0.5) == pytest.approx(0.5)


class TestHeuristicFusion:
    def test_returns_value_in_unit_range(self):
        fuse = heuristic_fusion()
        features = np.array([0.5, 0.3, 0.4, 0.6, 0.2, 0.9, 0.0], dtype=np.float32)
        result = fuse(features)
        assert 0.0 <= result <= 1.0

    def test_higher_scores_increase_result(self):
        fuse = heuristic_fusion()
        low = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.0], dtype=np.float32)
        high = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.0], dtype=np.float32)
        assert fuse(high) > fuse(low)

    def test_low_pose_confidence_reduces_score(self):
        fuse = heuristic_fusion()
        high_pose = np.array([0.7, 0.5, 0.6, 0.8, 0.3, 1.0, 0.0], dtype=np.float32)
        low_pose = np.array([0.7, 0.5, 0.6, 0.8, 0.3, 0.0, 0.0], dtype=np.float32)
        assert fuse(high_pose) > fuse(low_pose)


# --------------------------------------------------------------------------- #
# DecisionEngine integration tests
# --------------------------------------------------------------------------- #
class TestDecisionEngine:
    def test_single_frame_produces_decision(self):
        engine = DecisionEngine(fusion_model=heuristic_fusion())
        bs = _make_branch_scores()
        tr = _make_temporal_result()
        decision = engine.decide("p1", 1, 0.0, bs, tr, pose_confidence=0.9)
        assert isinstance(decision, FrameDecision)
        assert decision.participant_id == "p1"
        assert 0.0 <= decision.p_frame <= 1.0
        assert 0.0 <= decision.smoothed_score <= 1.0

    def test_smoothing_moves_toward_current_score(self):
        cfg = FusionConfig(smoothing_alpha=0.5)
        engine = DecisionEngine(cfg=cfg, fusion_model=heuristic_fusion())
        bs = _make_branch_scores(p_spatial=0.9, p_freq=0.9, p_sync=0.9)
        tr = _make_temporal_result(p_temporal=0.9, jitter=0.9)

        decisions = []
        for i in range(10):
            d = engine.decide("p1", i, float(i), bs, tr, pose_confidence=0.9)
            decisions.append(d)

        # Smoothed score should increase over time toward p_frame
        assert decisions[-1].smoothed_score > decisions[0].smoothed_score

    def test_review_flag_requires_sustained_threshold(self):
        cfg = FusionConfig(
            review_threshold=0.1,  # very low so heuristic scores trigger it
            sustained_seconds=2.0,
        )
        # Use a fusion model that always returns a high score
        engine = DecisionEngine(cfg=cfg, fusion_model=lambda f: 0.95)

        # First few frames: above threshold but not sustained long enough
        d1 = engine.decide("p1", 1, 0.0, _make_branch_scores(),
                            _make_temporal_result(), 0.9)
        assert d1.review_flag is False  # 0s elapsed, need 2s

        d2 = engine.decide("p1", 2, 1.0, _make_branch_scores(),
                            _make_temporal_result(), 0.9)
        assert d2.review_flag is False  # 1s elapsed, need 2s

        d3 = engine.decide("p1", 3, 2.5, _make_branch_scores(),
                            _make_temporal_result(), 0.9)
        assert d3.review_flag is True  # 2.5s elapsed > 2s sustained

    def test_block_threshold_higher_than_review(self):
        cfg = FusionConfig(
            review_threshold=0.1,
            block_threshold=0.99,  # very high, never reached
            sustained_seconds=0.0,
        )
        engine = DecisionEngine(cfg=cfg, fusion_model=lambda f: 0.5)
        d = engine.decide("p1", 1, 0.0, _make_branch_scores(),
                           _make_temporal_result(), 0.9)
        assert d.review_flag is True
        assert d.block_flag is False

    def test_threshold_resets_when_score_drops(self):
        cfg = FusionConfig(
            review_threshold=0.1,
            sustained_seconds=2.0,
            smoothing_alpha=1.0,  # no smoothing, instant response
        )

        call_count = 0
        def varying_fuse(f):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return 0.8  # above threshold
            elif call_count == 3:
                return 0.01  # below threshold, reset
            else:
                return 0.8  # above again

        engine = DecisionEngine(cfg=cfg, fusion_model=varying_fuse)

        engine.decide("p1", 1, 0.0, _make_branch_scores(),
                       _make_temporal_result(), 0.9)
        engine.decide("p1", 2, 1.5, _make_branch_scores(),
                       _make_temporal_result(), 0.9)
        # Score drops below threshold
        d3 = engine.decide("p1", 3, 2.0, _make_branch_scores(),
                            _make_temporal_result(), 0.9)
        assert d3.review_flag is False  # timer reset

        # Score back up but timer restarted
        d4 = engine.decide("p1", 4, 2.5, _make_branch_scores(),
                            _make_temporal_result(), 0.9)
        assert d4.review_flag is False  # only 0.5s since restart

    def test_multiple_participants_independent(self):
        engine = DecisionEngine(fusion_model=heuristic_fusion())
        bs = _make_branch_scores()
        tr = _make_temporal_result()

        d_alice = engine.decide("alice", 1, 0.0, bs, tr, 0.9)
        d_bob = engine.decide("bob", 1, 0.0, bs, tr, 0.9)

        # Both should get the same p_frame but have independent smoothed scores
        assert d_alice.participant_id == "alice"
        assert d_bob.participant_id == "bob"
        assert d_alice.p_frame == pytest.approx(d_bob.p_frame)

    def test_reset_participant_clears_state(self):
        cfg = FusionConfig(smoothing_alpha=0.3)
        engine = DecisionEngine(cfg=cfg, fusion_model=lambda f: 0.7)
        bs = _make_branch_scores()
        tr = _make_temporal_result()

        # Build up smoothed score
        for i in range(10):
            engine.decide("p1", i, float(i), bs, tr, 0.9)

        engine.reset_participant("p1")

        # After reset, smoothed score should start from 0 again
        d = engine.decide("p1", 100, 100.0, bs, tr, 0.9)
        # First frame after reset: smoothed = alpha * p_frame + (1-alpha) * 0
        assert d.smoothed_score < 0.5  # should be near alpha * p_frame
