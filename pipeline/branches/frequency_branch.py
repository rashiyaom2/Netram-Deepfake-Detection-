"""
Stage 4c — Frequency Branch (doc §4c).

Applies 2D FFT/DCT to the aligned face crop to surface power-spectrum
artifacts introduced by deepfake generators.

Known limitation (doc §4c caveat): this branch is strong against GAN-era
fakes (DeepFaceLab, FaceSwap) but weaker against diffusion and
Gaussian-splatting generators which produce smoother spectra. Its output
should be treated as one input to an adaptive fusion step (Stage 6),
not a fixed-weight component.

Output: P_freq ∈ [0, 1].

This branch is intentionally pure NumPy/SciPy — no PyTorch dependency —
since the core computation is a 2D FFT, not a learned model. A learned
frequency-domain classifier could replace the heuristic scoring below if
one were trained on labelled frequency spectra.
"""
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.fft import dctn

from pipeline.config import BranchConfig

FreqScorerFn = Callable[[np.ndarray], float]


# --------------------------------------------------------------------------- #
# Pure, independently-testable helpers
# --------------------------------------------------------------------------- #
def compute_dct_spectrum(face_crop_rgb01: np.ndarray) -> np.ndarray:
    """
    Convert face crop → grayscale → 2D DCT.
    Returns the log-magnitude DCT coefficient matrix (same spatial size as input).
    """
    gray = np.mean(face_crop_rgb01, axis=2).astype(np.float64)
    dct_coeffs = dctn(gray, type=2, norm="ortho")
    log_mag = np.log1p(np.abs(dct_coeffs))
    return log_mag


def compute_fft_spectrum(face_crop_rgb01: np.ndarray) -> np.ndarray:
    """
    Convert face crop → grayscale → 2D FFT → centred log-magnitude spectrum.
    """
    gray = np.mean(face_crop_rgb01, axis=2).astype(np.float64)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    log_mag = np.log1p(np.abs(fshift))
    return log_mag


def high_freq_energy_ratio(spectrum: np.ndarray, radius_frac: float = 0.25) -> float:
    """
    Ratio of energy in the high-frequency ring (outside `radius_frac` of
    the spectrum centre) to total energy. GAN-generated faces often have
    abnormal high-frequency patterns; diffusion fakes do not.
    """
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * radius_frac)
    yy, xx = np.ogrid[:h, :w]
    mask_high = (yy - cy) ** 2 + (xx - cx) ** 2 > radius ** 2
    high_energy = spectrum[mask_high].sum()
    total_energy = spectrum.sum() + 1e-8
    return float(high_energy / total_energy)


def mid_freq_variance(spectrum: np.ndarray, inner_frac: float = 0.15,
                       outer_frac: float = 0.40) -> float:
    """
    Variance of the mid-frequency band. GAN artifacts often show anomalous
    regularity (lower variance) in mid-frequencies compared to natural images.
    """
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    r_inner = int(min(h, w) * inner_frac)
    r_outer = int(min(h, w) * outer_frac)
    yy, xx = np.ogrid[:h, :w]
    dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
    mask_mid = (dist_sq > r_inner ** 2) & (dist_sq <= r_outer ** 2)
    if mask_mid.sum() == 0:
        return 0.0
    return float(np.var(spectrum[mask_mid]))


# --------------------------------------------------------------------------- #
# Scorer
# --------------------------------------------------------------------------- #
def heuristic_frequency_scorer() -> FreqScorerFn:
    """
    Combines FFT and DCT spectral analysis into a single suspicion score.
    This is a heuristic — effective primarily against GAN-era fakes per
    the doc §4c caveat. The fusion layer (Stage 6) should learn to
    down-weight this signal for diffusion-based generators.
    """
    def _score(face_crop_rgb01: np.ndarray) -> float:
        # FFT path: high-frequency energy ratio
        fft_spec = compute_fft_spectrum(face_crop_rgb01)
        hf_ratio = high_freq_energy_ratio(fft_spec)

        # DCT path: mid-frequency variance anomaly
        dct_spec = compute_dct_spectrum(face_crop_rgb01)
        mf_var = mid_freq_variance(dct_spec)

        # Combine: lower high-freq ratio or anomalous mid-freq variance → higher suspicion.
        # Calibrated so that natural webcam faces produce baseline ~0.08-0.15:
        # Natural webcam hf_ratio is typically 0.35-0.55, mf_var is 2.0-8.0.
        # GAN artifacts cause hf_ratio < 0.25 and mf_var < 1.0.
        hf_score = float(np.clip(1.0 - hf_ratio * 2.5, 0.0, 1.0))
        mf_score = float(np.clip(1.0 - mf_var / 2.0, 0.0, 1.0))

        p_freq = float(np.clip(0.4 * hf_score + 0.6 * mf_score, 0.0, 1.0))
        return p_freq

    return _score


def default_torch_freq_scorer(cfg: Optional[BranchConfig] = None) -> FreqScorerFn:
    """
    Production frequency scorer: loads the trained FreqSpectrumCNN from
    models/freq_classifier.pt and runs inference on DCT spectral maps.
    Falls back to the heuristic FFT/DCT scorer if model file is missing.
    """
    import os
    import logging
    logger = logging.getLogger(__name__)

    model_path = "models/freq_classifier.pt"
    if not os.path.exists(model_path):
        logger.debug(f"Frequency classifier weights not found at {model_path}, using FFT/DCT heuristic.")
        return heuristic_frequency_scorer()

    try:
        import torch
        from pipeline.branches.freq_classifier import FreqSpectrumCNN
    except ImportError:
        logger.debug("PyTorch not installed, using FFT/DCT heuristic.")
        return heuristic_frequency_scorer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FreqSpectrumCNN().to(device)

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"Loaded trained FreqSpectrumCNN from {model_path}")
    except Exception as e:
        logger.warning(f"Failed to load frequency classifier from {model_path}: {e}, using heuristic.")
        return heuristic_frequency_scorer()

    def _score(face_crop_rgb01: np.ndarray) -> float:
        # Compute DCT log-magnitude spectrum
        dct_spec = compute_dct_spectrum(face_crop_rgb01)
        # Normalize to [0, 1]
        vmin, vmax = dct_spec.min(), dct_spec.max()
        if vmax - vmin > 1e-8:
            dct_norm = (dct_spec - vmin) / (vmax - vmin)
        else:
            dct_norm = np.zeros_like(dct_spec)
        # Resize to model input size (112x112)
        from scipy.ndimage import zoom as sp_zoom
        h, w = dct_norm.shape
        dct_resized = sp_zoom(dct_norm, (112.0 / h, 112.0 / w), order=1).astype(np.float32)
        # Run inference
        tensor = torch.from_numpy(dct_resized).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 112, 112)
        with torch.no_grad():
            logit = model(tensor)
        prob = float(torch.sigmoid(logit).item())

        # Also blend in the FFT heuristic signal for robustness (70% CNN + 30% FFT)
        fft_spec = compute_fft_spectrum(face_crop_rgb01)
        hf_ratio = high_freq_energy_ratio(fft_spec)
        hf_score = float(np.clip(1.0 - hf_ratio * 1.5, 0.0, 1.0))
        blended = 0.7 * prob + 0.3 * hf_score

        return float(np.clip(blended, 0.0, 1.0))

    return _score


class FrequencyBranch:
    """Stage 4c entry point."""

    def __init__(self, cfg: Optional[BranchConfig] = None,
                 scorer: Optional[FreqScorerFn] = None):
        self.cfg = cfg or BranchConfig()
        # Try trained model first, then fall back to heuristic
        self._scorer = scorer or default_torch_freq_scorer(self.cfg)

    def run(self, face_crop_rgb01: np.ndarray) -> float:
        """Returns P_freq ∈ [0, 1]."""
        return self._scorer(face_crop_rgb01)

