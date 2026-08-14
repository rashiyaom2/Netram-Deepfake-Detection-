"""
Stage 4a — Audio-Visual Sync Branch (doc §4a).

The single highest-value addition (per the architecture doc). Real-time
face-swap pipelines frequently fail to perfectly time-align mouth movement
to cloned or pass-through audio, and this signal degrades much more slowly
than visual artifacts as generator quality improves.

Two sub-components:

1. **Sync confidence model** (Wav2Lip-style sync network):
   Runs over the aligned mouth crop + parallel audio window.
   Outputs P_sync ∈ [0,1] and an av_mismatch_flag.

2. **Voice-cloning artifact detector**:
   Runs on the audio track independently, looking for spectral
   discontinuities typical of TTS/voice-conversion output.
   Catches the case where a fake audio track is paired with a real
   video feed — which the sync model alone won't detect.
   Outputs P_voice_clone ∈ [0,1].

Two backends are provided for each sub-component:
  - default_torch_sync / default_torch_voice_clone : production, needs trained weights.
  - heuristic_sync / heuristic_voice_clone         : placeholder, no trained model needed.
"""
from typing import Callable, Optional, Tuple

import numpy as np

from pipeline.config import BranchConfig

# Type: sync scorer takes (mouth_crop_rgb01, audio_window_float32) →
#       (p_sync, av_mismatch_flag)
SyncScorerFn = Callable[[np.ndarray, np.ndarray], Tuple[float, bool]]

# Type: voice clone scorer takes audio_window_float32 → p_voice_clone
VoiceCloneScorerFn = Callable[[np.ndarray], float]


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def compute_mel_spectrogram(audio: np.ndarray, sample_rate: int = 16000,
                             n_fft: int = 512, hop_length: int = 160,
                             n_mels: int = 80) -> np.ndarray:
    """
    Compute a mel spectrogram from raw audio (mono float32).
    Uses a simplified approach with NumPy — for production, consider
    using librosa or torchaudio for better filter bank implementation.
    """
    # Pad if too short
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))

    # STFT
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    if n_frames < 1:
        n_frames = 1
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    window = np.hanning(n_fft)
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        if end > len(audio):
            break
        frame = audio[start:end] * window
        stft[:, i] = np.fft.rfft(frame)

    power = np.abs(stft) ** 2

    # Simplified mel filter bank
    mel_low = 0
    mel_high = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft // 2)

    mel_filter = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]
        for k in range(f_left, f_center):
            if f_center != f_left:
                mel_filter[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                mel_filter[m - 1, k] = (f_right - k) / (f_right - f_center)

    mel_spec = mel_filter @ power
    log_mel = np.log1p(mel_spec)
    return log_mel


def spectral_discontinuity_score(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Detect spectral discontinuities typical of TTS/voice-conversion output.
    TTS systems often produce audible artifacts at segment boundaries where
    different phoneme chunks are concatenated — these show up as sudden
    jumps in the spectral centroid or energy trajectory.
    """
    mel = compute_mel_spectrogram(audio, sample_rate)
    if mel.shape[1] < 3:
        return 0.0

    # Frame-to-frame spectral centroid deltas
    centroids = np.sum(mel * np.arange(mel.shape[0])[:, None], axis=0) / (np.sum(mel, axis=0) + 1e-8)
    centroid_deltas = np.abs(np.diff(centroids))

    # Frame-to-frame energy deltas
    energies = np.sum(mel, axis=0)
    energy_deltas = np.abs(np.diff(energies))

    # Large deltas → spectral discontinuities → possible voice cloning artifact
    mean_centroid_delta = float(np.mean(centroid_deltas))
    max_centroid_delta = float(np.max(centroid_deltas)) if len(centroid_deltas) > 0 else 0.0
    mean_energy_delta = float(np.mean(energy_deltas))

    # Heuristic mapping to [0,1] suspicion score
    centroid_score = float(np.clip(max_centroid_delta / (mean_centroid_delta + 1e-8) / 10.0, 0.0, 1.0))
    energy_score = float(np.clip(mean_energy_delta / 50.0, 0.0, 1.0))

    return float(np.clip(0.6 * centroid_score + 0.4 * energy_score, 0.0, 1.0))


def mouth_motion_energy(mouth_crop_rgb01: np.ndarray) -> float:
    """
    Estimate mouth region motion energy via gradient magnitude.
    Used as a proxy for lip movement when comparing against audio activity.
    """
    gray = np.mean(mouth_crop_rgb01, axis=2).astype(np.float32)
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))


# --------------------------------------------------------------------------- #
# Scorer backends — sync
# --------------------------------------------------------------------------- #
def default_torch_sync(cfg: BranchConfig) -> SyncScorerFn:
    """
    Production sync scorer: loads a trained Wav2Lip SyncNet model (SyncNet_color) from
    cfg.sync_weights_path. Computes cosine similarity between 512-d audio and lip embeddings.
    """
    import os
    import logging
    import cv2
    logger = logging.getLogger(__name__)

    candidate_paths = [
        getattr(cfg, "sync_weights_path", "models/sync_net.pt"),
        "models/syncnet_v2.model",
        "models/sync_net.pt",
        "models/syncnet.pth",
    ]
    model_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)

    if not model_path:
        logger.debug(f"SyncNet weights not found, using heuristic scorer.")
        return heuristic_sync()

    try:
        import torch
        from pipeline.branches.wav2lip import SyncNet_color
    except ImportError:
        logger.debug("PyTorch/SyncNet not available, using heuristic scorer.")
        return heuristic_sync()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        else:
            model = SyncNet_color()
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
            # Clean possible DataParallel 'module.' prefix
            cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned, strict=False)
        model.to(device).eval()
        logger.info(f"Loaded SyncNet model weights from {model_path}")
    except Exception as e:
        logger.warning(f"Failed to load SyncNet weights from {model_path}: {e}, falling back to heuristic.")
        return heuristic_sync()

    def _score(mouth_crop_rgb01: np.ndarray, audio_window: np.ndarray) -> Tuple[float, bool]:
        audio_energy = float(np.mean(np.abs(audio_window))) if audio_window is not None and len(audio_window) > 0 else 0.0
        if audio_energy < 0.005:
            return 0.5, False

        # Prepare 5-frame stacked mouth input (48x96, 15 channels: 5 * 3 RGB channels)
        mouth_resized = cv2.resize(mouth_crop_rgb01, (96, 48))
        mouth_5 = np.concatenate([mouth_resized] * 5, axis=2)  # (48, 96, 15)
        mouth_tensor = torch.from_numpy(
            np.transpose(mouth_5, (2, 0, 1))[np.newaxis, ...]
        ).float().to(device)

        # Audio mel-spectrogram: (1, 1, 80, 16)
        mel = compute_mel_spectrogram(audio_window, n_mels=80)
        if mel.shape[1] < 16:
            mel = np.pad(mel, ((0, 0), (0, 16 - mel.shape[1])))
        else:
            mel = mel[:, :16]
        mel_tensor = torch.from_numpy(mel[np.newaxis, np.newaxis, ...]).float().to(device)

        with torch.no_grad():
            a_emb, f_emb = model(mel_tensor, mouth_tensor)
            # Cosine similarity between normalized 512-d embeddings
            cos_sim = float(torch.sum(a_emb * f_emb, dim=1).item())
            # In SyncNet: cos_sim > 0.3 is in-sync (p_sync ~ 0.9), cos_sim < 0.0 is desynced (p_sync ~ 0.1)
            p_sync = float(np.clip((cos_sim + 0.2) / 0.8, 0.0, 1.0))

        m_energy = mouth_motion_energy(mouth_crop_rgb01)
        av_mismatch = (p_sync < 0.3) and (audio_energy > 0.01) and (m_energy > 0.02)
        return p_sync, av_mismatch

    return _score


def heuristic_sync() -> SyncScorerFn:
    """
    Placeholder sync scorer — uses correlation between mouth gradient
    energy and audio energy as a crude proxy. Replace with
    `default_torch_sync` once trained weights exist.
    """
    def _score(mouth_crop_rgb01: np.ndarray, audio_window: np.ndarray) -> Tuple[float, bool]:
        # Audio activity level
        audio_energy = float(np.mean(np.abs(audio_window))) if audio_window is not None and len(audio_window) > 0 else 0.0
        
        # If audio is silent/absent, no sync evaluation can be made
        if audio_energy < 0.005:
            return 0.5, False

        audio_active = audio_energy > 0.01

        # Mouth motion level
        m_energy = mouth_motion_energy(mouth_crop_rgb01)
        mouth_active = m_energy > 0.02

        # Sync heuristic: if both are active or both inactive, likely in sync
        if audio_active == mouth_active:
            p_sync = float(np.clip(0.8 + 0.2 * min(audio_energy * 10, 1.0), 0.0, 1.0))
            av_mismatch = False
        else:
            # Mismatch: audio on but mouth still (or vice versa)
            p_sync = float(np.clip(0.2 + 0.1 * min(audio_energy * 10, 1.0), 0.0, 1.0))
            av_mismatch = True

        return p_sync, av_mismatch

    return _score


# --------------------------------------------------------------------------- #
# Scorer backends — voice clone detection
# --------------------------------------------------------------------------- #
def default_torch_voice_clone(cfg: BranchConfig) -> VoiceCloneScorerFn:
    """
    Production voice-clone / synthetic speech detector: loads AASIST
    (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)
    from cfg.voice_clone_weights_path or assist/weights/AASIST.pth.
    """
    import os
    import logging
    logger = logging.getLogger(__name__)

    candidate_paths = [
        getattr(cfg, "voice_clone_weights_path", "assist/weights/AASIST.pth"),
        "assist/weights/AASIST.pth",
        "assist/weights/AASIST-L.pth",
        "models/voice_clone_detector.pt",
    ]
    model_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)

    if not model_path:
        logger.debug("Voice clone weights not found, using heuristic scorer.")
        return heuristic_voice_clone()

    try:
        import torch
        from assist.AASIST import Model as AASISTModel
    except ImportError:
        logger.debug("PyTorch / AASIST module not available, using heuristic scorer.")
        return heuristic_voice_clone()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        torch.set_num_threads(1)
        d_args = {
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 1.0, 1.0],
        }
        model = AASISTModel(d_args)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        logger.info(f"Loaded AASIST voice clone / synthetic speech detector from {model_path}")

    except Exception as e:
        logger.warning(f"Failed to load AASIST weights from {model_path}: {e}, using heuristic.")
        return heuristic_voice_clone()

    def _score(audio_window: np.ndarray) -> float:
        if audio_window is None or len(audio_window) == 0:
            return 0.0
        audio_energy = float(np.mean(np.abs(audio_window)))
        if audio_energy < 0.005:
            return 0.0

        # AASIST expects 64,600 samples (~4 seconds of 16kHz mono audio)
        target_len = 64600
        raw_audio = audio_window.astype(np.float32)
        if len(raw_audio) < target_len:
            num_repeats = int(np.ceil(target_len / len(raw_audio)))
            padded = np.tile(raw_audio, num_repeats)[:target_len]
        else:
            padded = raw_audio[:target_len]

        padded_contig = np.ascontiguousarray(padded, dtype=np.float32)
        tensor = torch.from_numpy(padded_contig).unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                _, logits = model(tensor)
                probs = torch.softmax(logits, dim=-1)
                p_spoof = float(probs[0, 1].item())
        except Exception as e:
            logger.debug(f"AASIST inference warning: {e}")
            return 0.0

        return float(np.clip(p_spoof, 0.0, 1.0))


    return _score


def heuristic_voice_clone() -> VoiceCloneScorerFn:
    """
    Placeholder voice-clone detector — uses spectral discontinuity analysis
    as a crude proxy. Replace with `default_torch_voice_clone` once
    trained weights exist.
    """
    def _score(audio_window: np.ndarray) -> float:
        return spectral_discontinuity_score(audio_window)

    return _score


# --------------------------------------------------------------------------- #
# Stage 4a entry point
# --------------------------------------------------------------------------- #
class AVSyncBranch:
    """Audio-visual sync + voice-clone detection (doc §4a)."""

    def __init__(self, cfg: Optional[BranchConfig] = None,
                 sync_scorer: Optional[SyncScorerFn] = None,
                 voice_clone_scorer: Optional[VoiceCloneScorerFn] = None):
        self.cfg = cfg or BranchConfig()
        self._sync_scorer = sync_scorer
        self._voice_clone_scorer = voice_clone_scorer

    @property
    def sync_scorer(self) -> SyncScorerFn:
        if self._sync_scorer is None:
            self._sync_scorer = default_torch_sync(self.cfg)
        return self._sync_scorer

    @property
    def voice_clone_scorer(self) -> VoiceCloneScorerFn:
        if self._voice_clone_scorer is None:
            self._voice_clone_scorer = default_torch_voice_clone(self.cfg)
        return self._voice_clone_scorer

    def run(self, mouth_crop_rgb01: np.ndarray,
            audio_window: Optional[np.ndarray] = None
            ) -> Tuple[Optional[float], Optional[bool], Optional[float]]:
        """
        Returns (p_sync, av_mismatch_flag, p_voice_clone).
        All are None if no audio_window or silent audio is available.
        """
        if audio_window is None or len(audio_window) == 0:
            return None, None, None

        audio_energy = float(np.mean(np.abs(audio_window)))
        if audio_energy < 0.005:
            return None, None, None

        p_sync, av_mismatch = self.sync_scorer(mouth_crop_rgb01, audio_window)
        p_voice_clone = self.voice_clone_scorer(audio_window)

        return p_sync, av_mismatch, p_voice_clone
