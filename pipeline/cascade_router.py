"""
Stage 0 — Cascade Router (doc §0).

Runs after QC gating (Stage 2) but BEFORE full alignment (Stage 3) and the
heavy branches (Stage 4). It reuses Stage 2's rough face bbox rather than
waiting for Stage 3's precise aligned crop -- the whole point is to avoid
spending alignment + full-model compute on frames that are about to be
dropped as confidently-real.

Only frames with suspicion_score > cfg.suspicion_threshold get escalated.
In a typical multi-participant call this is what keeps >4-5 participants
tractable in real time (doc: "Deployment Notes").

Two scorer backends are provided:
  - default_onnx_b0_scorer   : production path, EfficientNet-B0 via ONNX Runtime.
                                Requires a trained/exported model at cfg.onnx_path.
  - heuristic_frequency_scorer: NOT a trained deepfake detector. A cheap,
                                deterministic image-statistics stand-in so the
                                pipeline is runnable/testable end-to-end before
                                a trained cascade model exists. Swap it out
                                before relying on suspicion scores for anything
                                real.
"""
from typing import Callable, Optional, Tuple

import numpy as np
import cv2

from pipeline.config import CascadeConfig
from pipeline.types import RawFrame, QualityResult, CascadeResult

# EfficientNet-B0 (timm/torchvision pretrained-style) preprocessing constants.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ScorerFn = Callable[[np.ndarray], float]  # takes preprocessed CHW float32 array, returns suspicion score in [0,1]


# --------------------------------------------------------------------------- #
# Pure, independently-testable helpers
# --------------------------------------------------------------------------- #
def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def expand_bbox(bbox: Tuple[int, int, int, int], margin_pct: float,
                 image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    h_img, w_img = image_shape[:2]
    mx, my = w * margin_pct, h * margin_pct
    x0 = int(max(0, x - mx))
    y0 = int(max(0, y - my))
    x1 = int(min(w_img, x + w + mx))
    y1 = int(min(h_img, y + h + my))
    return x0, y0, x1, y1


def preprocess_for_cascade(crop_bgr: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    """Resize -> RGB -> [0,1] -> ImageNet-normalize -> CHW, matching standard EfficientNet-B0 preprocessing."""
    resized = cv2.resize(crop_bgr, input_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return chw.astype(np.float32)


# --------------------------------------------------------------------------- #
# Scorer backends
# --------------------------------------------------------------------------- #
def _get_safe_ort_providers():
    import ctypes
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            for dll in ["cublasLt64_13.dll", "cublasLt64_12.dll", "cublasLt64_11.dll"]:
                try:
                    ctypes.CDLL(dll)
                    return ["CUDAExecutionProvider", "CPUExecutionProvider"]
                except Exception:
                    pass
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def default_onnx_b0_scorer(cfg: CascadeConfig) -> ScorerFn:
    """
    Production path: loads the lightweight ONNX cascade model (e.g. EfficientNet-B0 / MobileNet).
    Imported lazily so the module works without onnxruntime if heuristics are used.
    """
    import os
    import logging
    logger = logging.getLogger(__name__)

    candidate_paths = [
        cfg.onnx_path,
        "models/cascade_b0.onnx",
        # Fallback: use the full spatial ViT at reduced resolution as cascade triage.
        # This is a real neural network (not a heuristic) — just slower than a dedicated
        # lightweight cascade model would be. Eliminates the FFT placeholder entirely.
        "models/deepfake_detector.onnx",
        "model.onnx",
    ]
    model_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)

    if not model_path:
        logger.warning("No ONNX model found for cascade triage — using FFT frequency heuristic as last resort.")
        return heuristic_frequency_scorer()


    try:
        import onnxruntime as ort
    except ImportError:
        logger.debug("onnxruntime not installed, falling back to heuristic.")
        return heuristic_frequency_scorer()

    providers = _get_safe_ort_providers()

    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        logger.warning(f"Failed to load cascade ONNX model at {model_path}: {e}, falling back to heuristic.")
        return heuristic_frequency_scorer()

    input_name = session.get_inputs()[0].name

    def _score(chw_array: np.ndarray) -> float:
        batch = chw_array[np.newaxis, ...]  # add batch dim (1, C, H, W)
        outputs = session.run(None, {input_name: batch})
        raw_out = np.asarray(outputs[0])

        if raw_out.ndim >= 2 and raw_out.shape[-1] >= 2:
            # 2-class logits (e.g. ViT [Real, Fake]): compute softmax over classes
            logits = raw_out.reshape(-1, raw_out.shape[-1])
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-8)
            # Index 1 corresponds to Fake in standard 2-class classifiers (Class 0 = Real, Class 1 = Fake)
            return float(np.clip(probs[0, 1], 0.0, 1.0))
        else:
            # 1-class binary logit
            logit = float(raw_out.reshape(-1)[0])
            return float(np.clip(sigmoid(logit), 0.0, 1.0))

    return _score


def heuristic_frequency_scorer() -> ScorerFn:
    """
    NOT a trained deepfake detector -- a placeholder so the pipeline runs
    end-to-end without ONNX weights. Uses high-frequency energy ratio (crude
    proxy: real webcam faces have natural high-frequency texture; heavily
    smoothed/generated regions often don't). Replace with
    `default_onnx_b0_scorer` once you have a trained cascade model.
    """
    def _score(chw_array: np.ndarray) -> float:
        # chw_array is normalized; undo enough to get a usable grayscale magnitude image.
        gray = chw_array.mean(axis=0)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4
        yy, xx = np.ogrid[:h, :w]
        mask_high = (yy - cy) ** 2 + (xx - cx) ** 2 > radius ** 2
        high_energy = magnitude[mask_high].sum()
        total_energy = magnitude.sum() + 1e-8
        ratio = float(high_energy / total_energy)
        # Map ratio to [0,1] suspicion-ish score; this is a heuristic, not calibrated probability.
        return float(np.clip(1.0 - ratio * 2.0, 0.0, 1.0))

    return _score


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #
class CascadeRouter:
    def __init__(self, cfg: Optional[CascadeConfig] = None, scorer: Optional[ScorerFn] = None):
        self.cfg = cfg or CascadeConfig()
        self._scorer = scorer

    @property
    def scorer(self) -> ScorerFn:
        if self._scorer is None:
            self._scorer = default_onnx_b0_scorer(self.cfg)
        return self._scorer

    def route(self, raw_frame: RawFrame, quality: QualityResult) -> Optional[CascadeResult]:
        """Returns None if the frame was already dropped at QC gating (nothing to route)."""
        if not quality.passed or quality.face_bbox is None:
            return None

        expanded = expand_bbox(quality.face_bbox, self.cfg.crop_margin_pct, raw_frame.image_bgr.shape)
        x0, y0, x1, y1 = expanded
        crop = raw_frame.image_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return CascadeResult(suspicion_score=0.0, escalate=False)

        preprocessed = preprocess_for_cascade(crop, self.cfg.input_size)
        score = self.scorer(preprocessed)
        return CascadeResult(suspicion_score=score, escalate=score > self.cfg.suspicion_threshold)
