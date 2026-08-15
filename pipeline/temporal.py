"""
Stage 5 — Advanced Multi-Modal Temporal & Landmark Verification (doc §5).

Includes:
1. Flexible Landmark Alias Resolution (MediaPipe, dlib 68-pt, OpenFace).
2. Scale-Invariant Landmark Micro-Jitter (normalized by inter-ocular distance).
3. Affine-Invariant Internal Non-Rigid Strain Metric (anti-spoofing presentation attack detection).
4. 2-Layer Bidirectional GRU with self-attention sequence pooling (Torch GPU/CPU inference with stride caching).
5. Head-Pose Gated Adaptive Baseline Eye Aspect Ratio (EAR) Blink Kinematics.
6. Automatic Participant Inactivity Timeout & Clean Reset.
"""
import time
import logging
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})
    TORCH_AVAILABLE = False

from pipeline.config import TemporalConfig
from pipeline.types import AlignedFace, BranchScores, TemporalResult

logger = logging.getLogger(__name__)

# Type: takes a sequence of embeddings (N x dim) and returns P_temporal
SequenceModelFn = Callable[[np.ndarray], float]

# --------------------------------------------------------------------------- #
# Landmark Dictionary Mapping & Alias Resolution
# --------------------------------------------------------------------------- #
LANDMARK_ALIASES: Dict[str, Tuple[str, ...]] = {
    "left_eye_top": ("left_eye_top", "left_eyelid_upper", "38", "159", "386"),
    "left_eye_bottom": ("left_eye_bottom", "left_eyelid_lower", "41", "145", "374"),
    "left_eye_outer": ("left_eye_outer", "left_eye_corner", "left_eye_left", "36", "33"),
    "left_eye_inner": ("left_eye_inner", "left_eye_right", "39", "133"),
    "right_eye_top": ("right_eye_top", "right_eyelid_upper", "43", "386"),
    "right_eye_bottom": ("right_eye_bottom", "right_eyelid_lower", "47", "374"),
    "right_eye_outer": ("right_eye_outer", "right_eye_corner", "right_eye_right", "45", "263"),
    "right_eye_inner": ("right_eye_inner", "right_eye_left", "42", "362"),
    "mouth_top": ("mouth_top", "upper_lip", "lip_top", "51", "13"),
    "mouth_bottom": ("mouth_bottom", "lower_lip", "lip_bottom", "57", "14"),
    "mouth_left": ("mouth_left", "lip_left", "48", "61"),
    "mouth_right": ("mouth_right", "lip_right", "54", "291"),
    "nose_tip": ("nose_tip", "nose", "30", "1"),
    "chin": ("chin", "jaw_bottom", "8", "152"),
}


def resolve_landmark(landmarks: Dict[str, Tuple[float, float]], key: str) -> Optional[Tuple[float, float]]:
    """Resolves landmark coordinates across MediaPipe, dlib 68-pt, and custom dictionary keys."""
    if not landmarks:
        return None
    if key in landmarks:
        return landmarks[key]
    aliases = LANDMARK_ALIASES.get(key, ())
    for alias in aliases:
        if alias in landmarks:
            return landmarks[alias]
    return None


# --------------------------------------------------------------------------- #
# Pure, Scale-Invariant Jitter & Liveness Helpers
# --------------------------------------------------------------------------- #
def compute_landmark_deltas(
    prev_landmarks: Dict[str, Tuple[float, float]],
    curr_landmarks: Dict[str, Tuple[float, float]],
    tracked_names: Tuple[str, ...],
) -> List[float]:
    """
    Euclidean distance between each tracked landmark across consecutive frames,
    normalized by inter-ocular distance (IOD) for scale invariance across resolutions.
    """
    p1_le = resolve_landmark(curr_landmarks, "left_eye_outer") or resolve_landmark(curr_landmarks, "left_eye_top")
    p1_re = resolve_landmark(curr_landmarks, "right_eye_outer") or resolve_landmark(curr_landmarks, "right_eye_top")
    if p1_le and p1_re:
        scale = float(np.hypot(p1_re[0] - p1_le[0], p1_re[1] - p1_le[1])) + 1e-6
    else:
        scale = 100.0  # default normalization scale

    deltas = []
    for name in tracked_names:
        prev_pt = resolve_landmark(prev_landmarks, name)
        curr_pt = resolve_landmark(curr_landmarks, name)
        if not prev_pt or not curr_pt:
            continue
        delta = float(np.hypot(curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1]))
        deltas.append(delta / scale)
    return deltas


def jitter_score_from_history(delta_history: List[List[float]], max_val: float = 1.2, scale: float = 0.0002) -> float:
    """
    Aggregate scale-invariant micro-jitter score from landmark delta history.
    Uses a second-order high-pass difference (acceleration) and outlier clipping
    to eliminate macro head rotation and transient re-detection spikes,
    leaving a pure high-frequency micro-jitter estimation.
    """
    if len(delta_history) < 2:
        return 0.0

    arr = np.clip(np.array(delta_history), -max_val, max_val)
    acc = np.diff(arr, axis=0)
    if len(acc) < 1:
        return 0.0

    per_landmark_var = np.var(acc, axis=0)
    mean_var = float(np.mean(per_landmark_var))
    # Calibrated threshold (scale=0.0002 maps micro-jitter to [0,1] suspicion score)
    score = float(np.clip(mean_var / scale, 0.0, 1.0))
    return score


def liveness_score_from_history(
    landmark_history: List[Dict[str, Tuple[float, float]]],
    tracked_names: Tuple[str, ...] = ("left_eye_top", "left_eye_bottom", "right_eye_top", "right_eye_bottom", "mouth_top", "mouth_bottom", "mouth_left", "mouth_right", "nose_tip", "chin"),
    min_variance_threshold: float = 0.5,
    frozen_threshold: float = 4e-5,
) -> float:
    """
    Anti-spoofing detector: detects presentation attacks (static photo printout, phone screen replay).
    Computes Affine-Invariant Internal Non-Rigid Strain Ratios across key facial landmark pairs.
    """
    if len(landmark_history) < 10:
        return 0.0

    strain_vectors = []
    pairs = [
        ("left_eye_top", "left_eye_bottom"),
        ("right_eye_top", "right_eye_bottom"),
        ("mouth_top", "mouth_bottom"),
        ("mouth_left", "mouth_right"),
        ("nose_tip", "mouth_top"),
        ("nose_tip", "chin"),
        ("left_eye_outer", "left_eye_top"),
        ("right_eye_outer", "right_eye_top"),
    ]

    for frame_lm in landmark_history:
        p_le = resolve_landmark(frame_lm, "left_eye_outer") or resolve_landmark(frame_lm, "left_eye_top")
        p_re = resolve_landmark(frame_lm, "right_eye_outer") or resolve_landmark(frame_lm, "right_eye_top")
        if not p_le or not p_re:
            continue
        eye_dist = float(np.hypot(p_re[0] - p_le[0], p_re[1] - p_le[1])) + 1e-6

        ratios = []
        for p1_name, p2_name in pairs:
            p1 = resolve_landmark(frame_lm, p1_name)
            p2 = resolve_landmark(frame_lm, p2_name)
            if p1 and p2:
                d = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
                ratios.append(d / eye_dist)
        if ratios:
            strain_vectors.append(ratios)

    if len(strain_vectors) < 10:
        positions = []
        for frame_lm in landmark_history:
            pts = [resolve_landmark(frame_lm, k) for k in tracked_names]
            valid_pts = [pt for pt in pts if pt is not None]
            if valid_pts:
                positions.append(valid_pts)
        if not positions or len(positions) < 10:
            return 0.0
        arr = np.array(positions)
        mean_var = float(np.mean(np.var(arr, axis=0)))
        if mean_var < min_variance_threshold:
            return float(np.clip(1.0 - (mean_var / min_variance_threshold), 0.0, 1.0))
        return 0.0

    strains_arr = np.array(strain_vectors)  # (N_frames, N_pairs)
    strain_var = float(np.mean(np.var(strains_arr, axis=0)))

    if strain_var >= frozen_threshold:
        return 0.0
    else:
        suspicion = 1.0 - (strain_var / frozen_threshold)
        return float(np.clip(suspicion, 0.0, 1.0))


def compute_eye_aspect_ratio(landmarks: Dict[str, Tuple[float, float]]) -> Optional[float]:
    """
    Computes Eye Aspect Ratio (EAR) from eyelid landmarks using flexible alias resolution.
    """
    le_top = resolve_landmark(landmarks, "left_eye_top")
    le_bot = resolve_landmark(landmarks, "left_eye_bottom")
    le_out = resolve_landmark(landmarks, "left_eye_outer")
    le_in = resolve_landmark(landmarks, "left_eye_inner")

    re_top = resolve_landmark(landmarks, "right_eye_top")
    re_bot = resolve_landmark(landmarks, "right_eye_bottom")
    re_out = resolve_landmark(landmarks, "right_eye_outer")
    re_in = resolve_landmark(landmarks, "right_eye_inner")

    if not (le_top and le_bot and le_out and le_in and re_top and re_bot and re_out and re_in):
        return None

    ear_l = np.hypot(le_top[0] - le_bot[0], le_top[1] - le_bot[1]) / (np.hypot(le_out[0] - le_in[0], le_out[1] - le_in[1]) + 1e-6)
    ear_r = np.hypot(re_top[0] - re_bot[0], re_top[1] - re_bot[1]) / (np.hypot(re_out[0] - re_in[0], re_out[1] - re_in[1]) + 1e-6)
    return float((ear_l + ear_r) / 2.0)


# --------------------------------------------------------------------------- #
# Sequence Model Backends
# --------------------------------------------------------------------------- #
class BidirectionalTemporalGRU(nn.Module if TORCH_AVAILABLE else object):
    """
    2-Layer Bidirectional GRU with recurrent dropout and temporal self-attention pooling
    for detecting inter-frame temporal inconsistencies and generative artifacts.
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.proj = nn.Linear(input_dim, 512) if input_dim != 512 else nn.Identity()
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.att_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: Any) -> Any:
        x = self.proj(x)
        gru_out, _ = self.gru(x)
        att_weights = torch.softmax(self.att_layer(gru_out), dim=1)
        context = torch.sum(gru_out * att_weights, dim=1)
        return self.classifier(context)


def default_torch_gru_model(cfg: TemporalConfig) -> SequenceModelFn:
    """Production path: 2-layer Bidirectional GRU model with self-attention."""
    import os

    if not os.path.exists(cfg.temporal_weights_path) or not TORCH_AVAILABLE:
        logger.debug(f"PyTorch or temporal weights not found at {cfg.temporal_weights_path}, using heuristic sequence scorer.")
        return heuristic_sequence_scorer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BidirectionalTemporalGRU(cfg.embedding_dim, cfg.gru_hidden_dim).to(device)

    try:
        checkpoint = torch.load(cfg.temporal_weights_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        elif isinstance(checkpoint, dict):
            state = checkpoint.get("state_dict", checkpoint)
            cleaned = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(cleaned, strict=False)
        model.eval()
        logger.info(f"Loaded Bidirectional Temporal GRU from {cfg.temporal_weights_path}")
    except Exception as e:
        logger.warning(f"Failed to load temporal weights from {cfg.temporal_weights_path}: {e}, using heuristic.")
        return heuristic_sequence_scorer()

    def _score(embeddings: np.ndarray) -> float:
        if embeddings.shape[0] < 2:
            return 0.0
        try:
            inp = torch.from_numpy(embeddings).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logit = model(inp)
                prob = torch.sigmoid(logit).item()
            return float(np.clip(prob, 0.0, 1.0))
        except Exception as ex:
            logger.debug(f"GRU forward failed: {ex}, falling back to heuristic.")
            return heuristic_sequence_scorer()(embeddings)

    return _score


def heuristic_sequence_scorer() -> SequenceModelFn:
    """Heuristic fallback for embedding sequence temporal variance."""
    def _score(embeddings: np.ndarray) -> float:
        if embeddings.shape[0] < 2:
            return 0.0
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        normalised = embeddings / norms
        cosine_sims = []
        for i in range(1, len(normalised)):
            sim = float(np.dot(normalised[i], normalised[i - 1]))
            cosine_sims.append(sim)
        cosine_sims = np.array(cosine_sims)
        sim_var = float(np.var(cosine_sims))
        mean_sim = float(np.mean(cosine_sims))
        var_score = float(np.clip(sim_var * 20.0, 0.0, 1.0))
        mean_score = float(np.clip(1.0 - mean_sim, 0.0, 1.0))
        return float(np.clip(0.5 * var_score + 0.5 * mean_score, 0.0, 1.0))

    return _score


# --------------------------------------------------------------------------- #
# Per-Participant Stateful Tracker
# --------------------------------------------------------------------------- #
_JITTER_LANDMARK_MAP = {
    "left_eye_corner": "left_eye_outer",
    "right_eye_corner": "right_eye_outer",
    "upper_lip": "mouth_top",
    "lower_lip": "mouth_bottom",
}


class TemporalTracker:
    """
    Stateful per-participant tracker that accumulates landmark and embedding
    history across frames, producing TemporalResult on each call.
    """

    def __init__(self, cfg: Optional[TemporalConfig] = None,
                 sequence_model: Optional[SequenceModelFn] = None):
        self.cfg = cfg or TemporalConfig()
        self._sequence_model = sequence_model

        # Rolling history buffers
        self._landmark_history: Deque[Dict[str, Tuple[float, float]]] = deque(
            maxlen=self.cfg.sequence_len
        )
        self._delta_history: Deque[List[float]] = deque(maxlen=self.cfg.sequence_len)
        self._embedding_history: Deque[np.ndarray] = deque(maxlen=self.cfg.sequence_len)

        # Adaptive Blink & Liveness tracking state
        self._ear_history: Deque[Tuple[float, float]] = deque()
        self._blink_timestamps: Deque[float] = deque()
        self._is_blinking: bool = False
        self._blink_start_time: Optional[float] = None
        self._valid_blinks_count: int = 0
        self._first_seen_time: Optional[float] = None

        # Optimization & Auto-Cleanup State
        self.last_active_time: float = time.time()
        self._frame_count: int = 0
        self._cached_p_temporal: float = 0.0

    @property
    def sequence_model(self) -> SequenceModelFn:
        if self._sequence_model is None:
            self._sequence_model = default_torch_gru_model(self.cfg)
        return self._sequence_model

    def is_inactive(self, timeout_s: Optional[float] = None) -> bool:
        """Returns True if no updates have been received for longer than timeout_s."""
        t_out = timeout_s if timeout_s is not None else self.cfg.inactivity_timeout_s
        return (time.time() - self.last_active_time) > t_out

    def _resolve_landmark_names(self) -> Tuple[str, ...]:
        resolved = []
        for cfg_name in self.cfg.jitter_landmark_ids:
            actual_name = _JITTER_LANDMARK_MAP.get(cfg_name, cfg_name)
            resolved.append(actual_name)
        return tuple(resolved)

    def get_adaptive_ear_threshold(self) -> float:
        if len(self._ear_history) < 5:
            return self.cfg.ear_blink_threshold
        ears = [ear_val for _, ear_val in self._ear_history]
        baseline_open_ear = float(np.percentile(ears, 80))
        dynamic_thresh = baseline_open_ear * (1.0 - self.cfg.ear_relative_drop)
        return float(np.clip(dynamic_thresh, 0.10, 0.26))

    def update(self, aligned_face: AlignedFace, branch_scores: BranchScores) -> TemporalResult:
        now = aligned_face.timestamp if aligned_face.timestamp > 0 else time.time()
        self.last_active_time = now
        self._frame_count += 1

        if self._first_seen_time is None:
            self._first_seen_time = now

        curr_landmarks = aligned_face.landmarks

        # 1. Landmark jitter tracking
        if self._landmark_history:
            prev_landmarks = self._landmark_history[-1]
            deltas = compute_landmark_deltas(prev_landmarks, curr_landmarks, self._resolve_landmark_names())
            if deltas:
                self._delta_history.append(deltas)

        self._landmark_history.append(curr_landmarks)
        jitter = jitter_score_from_history(list(self._delta_history))

        # 2. Sequential feature pooling (Bi-GRU with stride caching for CPU optimization)
        self._embedding_history.append(branch_scores.embedding)

        if self._frame_count % max(1, self.cfg.gru_stride) == 0 or self._cached_p_temporal == 0.0:
            if len(self._embedding_history) >= 2:
                seq_arr = np.array(list(self._embedding_history))
                self._cached_p_temporal = self.sequence_model(seq_arr)
            else:
                self._cached_p_temporal = 0.0

        p_temporal = self._cached_p_temporal

        # 3. Head-Pose Gated Adaptive Baseline Eye Aspect Ratio (EAR) Blink Detection
        head_angle_ok = True
        if hasattr(aligned_face, "pose_angles") and aligned_face.pose_angles:
            pitch, yaw, _ = aligned_face.pose_angles
            if abs(pitch) > self.cfg.max_pose_angle_deg or abs(yaw) > self.cfg.max_pose_angle_deg:
                head_angle_ok = False

        just_blinked = False
        if head_angle_ok:
            ear = compute_eye_aspect_ratio(curr_landmarks)
            if ear is not None:
                self._ear_history.append((now, ear))
                while self._ear_history and (now - self._ear_history[0][0]) > self.cfg.blink_window_seconds:
                    self._ear_history.popleft()

                dyn_thresh = self.get_adaptive_ear_threshold()
                # Relative EAR dip check: eyes closed if below dynamic threshold OR relative 25% dip from 80th percentile
                ear_baseline = float(np.percentile([e for _, e in self._ear_history], 80)) if len(self._ear_history) >= 4 else 0.28
                is_closed = (ear < dyn_thresh) or (ear < ear_baseline * 0.75)

                if is_closed:
                    if not self._is_blinking:
                        self._is_blinking = True
                        self._blink_start_time = now
                else:
                    if self._is_blinking:
                        self._is_blinking = False
                        if self._blink_start_time is not None:
                            duration = now - self._blink_start_time
                            # Natural physiological blink duration (handles 3-30 FPS WebRTC sampling)
                            if (duration <= self.cfg.max_blink_duration_s) or (duration == 0.0):
                                self._blink_timestamps.append(now)
                                self._valid_blinks_count += 1
                                just_blinked = True
                        self._blink_start_time = None

        while self._blink_timestamps and (now - self._blink_timestamps[0]) > self.cfg.blink_window_seconds:
            self._blink_timestamps.popleft()

        # 4. Multi-Signal Physiological Anti-Spoofing / Liveness Verification
        stillness_score = liveness_score_from_history(
            list(self._landmark_history),
            self._resolve_landmark_names(),
            self.cfg.liveness_min_variance,
            self.cfg.frozen_threshold,
        )

        n_blinks = len(self._blink_timestamps)

        p_liveness = 0.0
        if len(self._landmark_history) >= 10:
            if stillness_score > 0.35:
                elapsed = now - self._first_seen_time if self._first_seen_time else 0.0
                n_frames = len(self._landmark_history)

                # Low-FPS guard: at 3 FPS, blink detection is unreliable.
                fps_estimate = n_frames / max(elapsed, 1.0)
                is_low_fps = fps_estimate < 8.0

                if n_blinks == 0:
                    if is_low_fps:
                        if elapsed < 30.0:
                            p_liveness = float(np.clip(stillness_score * 0.15, 0.0, 0.18))
                        else:
                            p_liveness = float(np.clip(stillness_score * 0.40, 0.0, 0.45))
                    else:
                        if elapsed < 10.0:
                            p_liveness = float(np.clip(stillness_score * 0.30, 0.0, 0.35))
                        else:
                            p_liveness = float(np.clip(stillness_score * 0.75, 0.0, 0.80))
                else:
                    p_liveness = 0.0
            else:
                p_liveness = 0.0

        # Biological Proof-of-Life ("Blink = Real"): if blinks are verified, presentation attack probability is strictly 0.0
        if n_blinks > 0 or just_blinked:
            p_liveness = 0.0

        return TemporalResult(
            p_temporal=p_temporal,
            jitter_score=jitter,
            p_liveness=p_liveness,
            blink_detected=just_blinked,
            recent_blinks=n_blinks
        )

    def reset(self) -> None:
        """Clear all history (e.g. when a participant rejoins)."""
        self._landmark_history.clear()
        self._delta_history.clear()
        self._embedding_history.clear()
        self._ear_history.clear()
        self._blink_timestamps.clear()
        self._is_blinking = False
        self._blink_start_time = None
        self._valid_blinks_count = 0
        self._first_seen_time = None
        self.last_active_time = time.time()
        self._frame_count = 0
        self._cached_p_temporal = 0.0
