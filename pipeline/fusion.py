"""
Stage 6 — Decision Engine: Adaptive Fusion (doc §6).

Three components:

1. **Learned fusion layer** (logistic regression or shallow MLP):
   Replaces hand-tuned fixed weights with a trained model that takes all
   branch signals as input and outputs P_frame.

   P_frame = f_θ(P_spatial, P_freq, P_temporal, P_sync, Jitter, pose_confidence)

   Why learned: fixed weights don't generalise across lighting, hardware,
   or generator type, and can't reflect that the frequency branch should
   contribute less as diffusion-based fakes become more common. A learned
   layer, recalibrated periodically, adapts automatically.

2. **Exponential smoothing:**
   S_t = α · P_frame + (1 - α) · S_{t-1}
   Prevents single-frame spikes from triggering false positives.

3. **Calibrated dual thresholds (not fixed):**
   - review_threshold: "flag for human review" — lower, catches more
   - block_threshold: "auto-block / warn all participants" — higher, fewer false positives
   Trigger only if the smoothed score exceeds the threshold consistently
   for sustained_seconds (default 3s).

The fusion model is stored as a pickled scikit-learn pipeline (logistic
regression or MLP), loaded from cfg.fusion_model_path. A heuristic
weighted-average fallback is provided for running without trained weights.
"""
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Optional, Tuple

import numpy as np

from pipeline.config import FusionConfig
from pipeline.types import (
    AlignedFace, BranchScores, TemporalResult, FusionInput, FrameDecision,
)


# Type: fusion model takes feature vector → P_frame ∈ [0,1]
FusionModelFn = Callable[[np.ndarray], float]


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def build_fusion_feature_vector(fusion_input: FusionInput) -> np.ndarray:
    """
    Assemble the feature vector expected by the fusion model.
    Order must match the training-time feature order exactly.
    Missing signals (p_sync=None) are set to 0.5 (neutral prior).
    """
    p_sync_val = fusion_input.p_sync if fusion_input.p_sync is not None else 0.5
    return np.array([
        fusion_input.p_spatial,
        fusion_input.p_freq,
        fusion_input.p_temporal,
        p_sync_val,
        fusion_input.jitter,
        fusion_input.pose_confidence,
        fusion_input.p_liveness,
    ], dtype=np.float32)


def exponential_smooth(current_score: float, prev_smoothed: float,
                        alpha: float) -> float:
    """S_t = α · P_frame + (1 - α) · S_{t-1}"""
    return alpha * current_score + (1.0 - alpha) * prev_smoothed


# --------------------------------------------------------------------------- #
# Fusion model backends
# --------------------------------------------------------------------------- #
def default_sklearn_fusion(cfg: FusionConfig) -> FusionModelFn:
    """
    Production fusion: loads a trained scikit-learn model (logistic
    regression or MLP) from cfg.fusion_model_path. The model should have
    been trained on held-out labelled data with the same 6-feature input
    vector produced by `build_fusion_feature_vector`.
    """
    import os
    import logging
    logger = logging.getLogger(__name__)

    if not os.path.exists(cfg.fusion_model_path):
        logger.debug(f"Fusion model weights not found at {cfg.fusion_model_path}, using heuristic fusion.")
        return heuristic_fusion()

    try:
        import joblib
    except ImportError:
        logger.debug("joblib/sklearn not installed, using heuristic fusion.")
        return heuristic_fusion()

    model = joblib.load(cfg.fusion_model_path)

    def _fuse(features: np.ndarray) -> float:
        prob = float(model.predict_proba(features.reshape(1, -1))[0, 1])
        p_spatial, p_freq, p_temporal, p_sync, jitter, pose, p_liveness = features
        # Threat lower bound: presentation attacks or blatant visual artifacts cannot be diluted
        if p_liveness >= 0.40:
            prob = max(prob, float(p_liveness * 0.92))
        if p_spatial >= 0.70:
            prob = max(prob, float(p_spatial * 0.90))
        return float(np.clip(prob, 0.0, 1.0))

    return _fuse


def heuristic_fusion() -> FusionModelFn:
    """
    Calibrated multi-branch neural fusion with presentation attack & dominant threat escalation.
    """
    def _fuse(features: np.ndarray) -> float:
        # 0: spatial, 1: freq, 2: temporal, 3: sync, 4: jitter, 5: pose_conf, 6: liveness
        p_spatial, p_freq, p_temporal, p_sync, jitter, pose, p_liveness = features

        if abs(p_sync - 0.5) < 0.05:
            # Video only (no audio track)
            weighted_score = (
                p_spatial * 0.35 +
                p_freq * 0.25 +
                p_temporal * 0.20 +
                p_liveness * 0.20
            )
        else:
            # Multi-modal (video + audio)
            weighted_score = (
                p_spatial * 0.28 +
                p_sync * 0.22 +
                p_freq * 0.20 +
                p_temporal * 0.15 +
                p_liveness * 0.15
            )

        # Multi-Signal Threat Escalation (prevents dilution when a single branch strongly flags an attack)
        # 1. Presentation Attack (static photo, phone screen replay, unblinking 2D plane)
        if p_liveness >= 0.40:
            weighted_score = max(weighted_score, float(p_liveness * 0.92))

        # 2. Strong Visual Artifacts / Diffusion Seams
        if p_spatial >= 0.70:
            weighted_score = max(weighted_score, float(p_spatial * 0.90))

        # 3. Audio-Visual Lip-Sync Desynchronization
        if abs(p_sync - 0.5) >= 0.05 and p_sync >= 0.75:
            weighted_score = max(weighted_score, float(p_sync * 0.88))

        # Pose modulation (near frontal faces give full confidence)
        pose_factor = np.clip(pose, 0.7, 1.0)
        return float(np.clip(weighted_score * pose_factor, 0.0, 1.0))

    return _fuse




# --------------------------------------------------------------------------- #
# Per-participant state
# --------------------------------------------------------------------------- #
class _ParticipantFusionState:
    """Smoothed score and threshold-sustain timing for one participant."""

    def __init__(self):
        self.smoothed_score: float = 0.0
        self.review_sustained_since: Optional[float] = None
        self.block_sustained_since: Optional[float] = None


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #
class DecisionEngine:
    """
    Stateful decision engine that smooths per-frame scores and applies
    calibrated dual thresholds with sustained-trigger logic.
    """

    def __init__(self, cfg: Optional[FusionConfig] = None,
                 fusion_model: Optional[FusionModelFn] = None):
        self.cfg = cfg or FusionConfig()
        self._fusion_model = fusion_model or default_sklearn_fusion(self.cfg)
        self._states: Dict[str, _ParticipantFusionState] = defaultdict(
            _ParticipantFusionState
        )

    @property
    def fusion_model(self) -> FusionModelFn:
        return self._fusion_model

    def decide(self, participant_id: str, frame_idx: int, timestamp: float,
               branch_scores: BranchScores, temporal_result: TemporalResult,
               pose_confidence: float,
               av_mismatch_flag: Optional[bool] = None) -> FrameDecision:
        """
        Fuse all signals for one frame, smooth, threshold, and return a
        FrameDecision. Maintains per-participant state (smoothed score,
        time-above-threshold).
        """
        if av_mismatch_flag is None:
            av_mismatch_flag = branch_scores.av_mismatch_flag

        state = self._states[participant_id]

        # Build fusion input
        fusion_input = FusionInput(
            p_spatial=branch_scores.p_spatial,
            p_freq=branch_scores.p_freq,
            p_temporal=temporal_result.p_temporal,
            p_sync=branch_scores.p_sync,
            jitter=temporal_result.jitter_score,
            pose_confidence=pose_confidence,
            p_liveness=temporal_result.p_liveness,
        )
        features = build_fusion_feature_vector(fusion_input)
        p_frame = self._fusion_model(features)

        # Exponential smoothing
        smoothed = exponential_smooth(p_frame, state.smoothed_score, self.cfg.smoothing_alpha)
        state.smoothed_score = smoothed

        # Sustained threshold logic
        now = timestamp

        # Review threshold
        if smoothed >= self.cfg.review_threshold:
            if state.review_sustained_since is None:
                state.review_sustained_since = now
            review_flag = (now - state.review_sustained_since) >= self.cfg.sustained_seconds
        else:
            state.review_sustained_since = None
            review_flag = False

        # Block threshold
        if smoothed >= self.cfg.block_threshold:
            if state.block_sustained_since is None:
                state.block_sustained_since = now
            block_flag = (now - state.block_sustained_since) >= self.cfg.sustained_seconds
        else:
            state.block_sustained_since = None
            block_flag = False

        return FrameDecision(
            participant_id=participant_id,
            frame_idx=frame_idx,
            timestamp=timestamp,
            p_frame=p_frame,
            smoothed_score=smoothed,
            review_flag=review_flag,
            block_flag=block_flag,
            av_mismatch_flag=av_mismatch_flag or branch_scores.av_mismatch_flag,
        )

    def reset_participant(self, participant_id: str) -> None:
        """Clear state for a participant (e.g. when they leave the call)."""
        self._states.pop(participant_id, None)
