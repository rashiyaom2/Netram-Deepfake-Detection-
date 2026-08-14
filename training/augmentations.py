"""
Training — Augmentations (doc Training Strategy table).

Heavy augmentation forces reliance on deep artifact features rather than
resolution/quality cues that streaming compression destroys:

  - JPEG compression  (Q ∈ [30, 90])
  - WebM downscaling  (simulate variable-resolution video tiles)
  - Motion blur       (simulate head movement)
  - Gaussian noise    (simulate sensor / compression noise)

All augmentations operate on uint8 HWC numpy arrays (standard OpenCV format)
and are composable via `AugmentationPipeline`.
"""
import io
from typing import List, Optional, Tuple

import numpy as np
import cv2


# --------------------------------------------------------------------------- #
# Individual augmentations (each is a pure function: np.ndarray → np.ndarray)
# --------------------------------------------------------------------------- #
def jpeg_compress(image: np.ndarray, quality: Optional[int] = None,
                   quality_range: Tuple[int, int] = (30, 90)) -> np.ndarray:
    """
    Re-encode through JPEG at a random quality level to simulate the
    artifacts introduced by video codec compression.
    """
    if quality is None:
        quality = int(np.random.randint(quality_range[0], quality_range[1] + 1))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def webm_downscale(image: np.ndarray,
                    scale_range: Tuple[float, float] = (0.3, 0.8)) -> np.ndarray:
    """
    Simulate variable-resolution WebM video tiles by downscaling then
    upscaling back to original size — introduces the smoothing/blur
    that WebRTC does to low-bandwidth participants.
    """
    h, w = image.shape[:2]
    scale = np.random.uniform(scale_range[0], scale_range[1])
    small_h, small_w = max(1, int(h * scale)), max(1, int(w * scale))
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def motion_blur(image: np.ndarray,
                 kernel_size_range: Tuple[int, int] = (5, 25)) -> np.ndarray:
    """
    Apply directional motion blur at a random angle to simulate head
    movement during a video call.
    """
    ksize = np.random.randint(kernel_size_range[0], kernel_size_range[1] + 1)
    if ksize % 2 == 0:
        ksize += 1
    angle = np.random.uniform(0, 180)

    # Build motion blur kernel
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))
    for i in range(ksize):
        offset = i - center
        x = int(center + offset * cos_a)
        y = int(center + offset * sin_a)
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1.0
    kernel /= kernel.sum() + 1e-8

    return cv2.filter2D(image, -1, kernel)


def gaussian_noise(image: np.ndarray,
                    sigma_range: Tuple[float, float] = (5.0, 25.0)) -> np.ndarray:
    """
    Add Gaussian noise to simulate sensor and compression noise.
    """
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    noise = np.random.randn(*image.shape) * sigma
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def random_brightness_contrast(image: np.ndarray,
                                  brightness_range: Tuple[float, float] = (-30, 30),
                                  contrast_range: Tuple[float, float] = (0.7, 1.3)
                                  ) -> np.ndarray:
    """
    Random brightness and contrast adjustment to simulate varying
    webcam exposure settings.
    """
    brightness = np.random.uniform(brightness_range[0], brightness_range[1])
    contrast = np.random.uniform(contrast_range[0], contrast_range[1])
    img = image.astype(np.float32)
    img = contrast * img + brightness
    return np.clip(img, 0, 255).astype(np.uint8)


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    """Random horizontal flip (50% chance)."""
    if np.random.random() < 0.5:
        return cv2.flip(image, 1)
    return image


# --------------------------------------------------------------------------- #
# Audio augmentations (for A/V sync branch training)
# --------------------------------------------------------------------------- #
def audio_time_shift(audio: np.ndarray,
                      max_shift_samples: int = 800) -> np.ndarray:
    """
    Shift audio forward or backward by a random number of samples.
    Used to create deliberately desynced audio-video pairs for sync
    branch training (doc: "Audio-visual pair augmentation").
    """
    shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
    if shift == 0:
        return audio
    return np.roll(audio, shift)


def audio_gaussian_noise(audio: np.ndarray,
                           snr_db_range: Tuple[float, float] = (10.0, 30.0)
                           ) -> np.ndarray:
    """Add Gaussian noise at a random SNR level."""
    snr_db = np.random.uniform(snr_db_range[0], snr_db_range[1])
    signal_power = np.mean(audio ** 2) + 1e-8
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
    return (audio + noise).astype(np.float32)


# --------------------------------------------------------------------------- #
# Composable pipeline
# --------------------------------------------------------------------------- #
class AugmentationPipeline:
    """
    Applies a random subset of augmentations with configurable probabilities.

    Usage:
        aug = AugmentationPipeline(p_jpeg=0.5, p_motion=0.3, ...)
        augmented = aug(image)
    """

    def __init__(self,
                 p_jpeg: float = 0.5,
                 p_webm: float = 0.3,
                 p_motion: float = 0.2,
                 p_noise: float = 0.3,
                 p_brightness: float = 0.3,
                 p_flip: float = 0.5):
        self.augmentations = [
            (p_jpeg, jpeg_compress),
            (p_webm, webm_downscale),
            (p_motion, motion_blur),
            (p_noise, gaussian_noise),
            (p_brightness, random_brightness_contrast),
            (p_flip, horizontal_flip),
        ]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        for prob, aug_fn in self.augmentations:
            if np.random.random() < prob:
                image = aug_fn(image)
        return image
