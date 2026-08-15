"""
Shared message types passed between pipeline stages.
Keeping these as plain dataclasses (not framework-specific objects) makes
every stage independently testable with synthetic data.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import numpy as np


@dataclass
class RawFrame:
    """One sampled video frame + the aligned audio slice, pre-processing."""
    participant_id: str
    frame_idx: int
    timestamp: float                 # seconds, shared clock with audio
    image_bgr: np.ndarray            # raw frame, HxWx3 uint8
    audio_window: Optional[np.ndarray] = None   # mono float32, ~3-5s @ 16kHz


@dataclass
class QualityResult:
    passed: bool
    pose_confidence: float           # 0-1, 1 = frontal, degrades to ~0 at 45deg
    blur_score: float
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    reject_reason: Optional[str] = None


@dataclass
class AlignedFace:
    participant_id: str
    frame_idx: int
    timestamp: float
    face_crop: np.ndarray            # aligned, resized, RGB float32 normalized
    mouth_crop: np.ndarray           # for audio-visual sync branch
    landmarks: Dict[str, Tuple[float, float]]
    pose_confidence: float
    audio_window: Optional[np.ndarray] = None


@dataclass
class CascadeResult:
    suspicion_score: float
    escalate: bool                   # True if suspicion_score > cascade threshold


@dataclass
class BranchScores:
    """Output of Stage 4 (all sub-branches) for a single frame."""
    p_spatial: float
    p_freq: float
    embedding: np.ndarray             # 512-d, from spatial branch
    p_sync: Optional[float] = None
    av_mismatch_flag: Optional[bool] = None   # video faked / audio real (or vice versa)
    p_voice_clone: Optional[float] = None
    phone_detected: bool = False
    phone_confidence: float = 0.0
    ar_filter_detected: bool = False
    ar_filter_confidence: float = 0.0
    filter_type: Optional[str] = None


@dataclass
class TemporalResult:
    p_temporal: float
    jitter_score: float
    p_liveness: float
    blink_detected: bool = False
    recent_blinks: int = 0


@dataclass
class FusionInput:
    p_spatial: float
    p_freq: float
    p_temporal: float
    p_sync: Optional[float]
    jitter: float
    pose_confidence: float
    p_liveness: float
    phone_detected: bool = False
    phone_confidence: float = 0.0
    ar_filter_detected: bool = False
    ar_filter_confidence: float = 0.0
    filter_type: Optional[str] = None
    blink_detected: bool = False
    recent_blinks: int = 0


@dataclass
class FrameDecision:
    participant_id: str
    frame_idx: int
    timestamp: float
    p_frame: float
    smoothed_score: float
    review_flag: bool
    block_flag: bool
    av_mismatch_flag: Optional[bool] = None
    phone_detected: bool = False
    phone_confidence: float = 0.0
    ar_filter_detected: bool = False
    ar_filter_confidence: float = 0.0
    filter_type: Optional[str] = None
    blink_detected: bool = False
    recent_blinks: int = 0
