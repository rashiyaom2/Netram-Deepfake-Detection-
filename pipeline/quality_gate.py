"""
Stage 2 — Quality Control Gating (doc §2).

Runs a cheap face+pose estimate (MediaPipe's short-range detector, 6
keypoints + solvePnP) — NOT the full landmark mesh used later in Stage 3
for alignment. This stage's job is to filter unusable frames fast, before
any heavy inference runs.

Hard rejects: no face found, face too small, too blurry, too dark/bright,
pose beyond the ±45deg hard cutoff.

Soft signal: for frames that DO pass, `pose_confidence` is carried forward
(not thresholded away) so a 40deg frame that passes still contributes less
to Stage 6 fusion than a near-frontal one, per doc §2's pose-weighting note.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List

import numpy as np
import cv2

from pipeline.config import QualityGateConfig
from pipeline.types import RawFrame, QualityResult


# --------------------------------------------------------------------------- #
# Generic 3D face model (approximate, mm) matching MediaPipe's 6 keypoints:
# right_eye, left_eye, nose_tip, mouth_center, right_ear_tragion, left_ear_tragion.
# Good enough for a coarse yaw/pitch estimate -- not for fine-grained alignment.
# --------------------------------------------------------------------------- #
_MODEL_POINTS_3D = np.array([
    [-30.0,  32.0, -26.0],   # right eye (subject's right)
    [ 30.0,  32.0, -26.0],   # left eye
    [  0.0,   0.0,   0.0],   # nose tip (origin)
    [  0.0, -60.0, -10.0],   # mouth center
    [-75.0,   8.0, -50.0],   # right ear tragion
    [ 75.0,   8.0, -50.0],   # left ear tragion
], dtype=np.float64)


@dataclass
class FaceDetection:
    bbox: Tuple[int, int, int, int]           # x, y, w, h in pixels
    keypoints: List[Tuple[float, float]]      # 6 points, pixel coords, order matches _MODEL_POINTS_3D
    detection_confidence: float


FaceDetectorFn = Callable[[np.ndarray], List[FaceDetection]]


def default_mediapipe_detector(model_path: str = "models/blaze_face_short_range.tflite") -> FaceDetectorFn:
    """
    Builds a detector function backed by MediaPipe's FaceDetector Task.
    """
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
    detector = vision.FaceDetector.create_from_options(options)

    def _detect(image_bgr: np.ndarray) -> List[FaceDetection]:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)
        
        detections: List[FaceDetection] = []
        if not results.detections:
            return detections
            
        for det in results.detections:
            bbox = det.bounding_box
            x = bbox.origin_x
            y = bbox.origin_y
            bw = bbox.width
            bh = bbox.height
            
            kps_list = det.keypoints
            if kps_list and len(kps_list) >= 6:
                keypoints = [(kp.x * w, kp.y * h) for kp in kps_list]
            else:
                keypoints = [(0.0, 0.0)] * 6
                
            detections.append(FaceDetection(
                bbox=(x, y, bw, bh),
                keypoints=keypoints[:6],
                detection_confidence=det.categories[0].score if det.categories else 0.0,
            ))
        return detections

    return _detect


# --------------------------------------------------------------------------- #
# Pure, independently-testable helper functions
# --------------------------------------------------------------------------- #
def laplacian_variance(image_bgr: np.ndarray) -> float:
    """Higher = sharper. Doc §2: 'Laplacian blur test' (OpenCV)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_brightness(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    return float(gray.mean())


def estimate_pose_angles(keypoints: List[Tuple[float, float]], image_shape: Tuple[int, int]) -> Tuple[float, float]:
    """
    Coarse yaw/pitch estimate (degrees) via solvePnP against a generic 3D
    face model. Returns (yaw_deg, pitch_deg). Positive yaw = turned to
    subject's right; positive pitch = looking up. Sign conventions only
    matter for consistency within this pipeline, not absolute accuracy.
    """
    h, w = image_shape[:2]
    focal_length = w  # rough approximation: focal length ~= image width
    camera_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

    image_points = np.array(keypoints, dtype=np.float64)
    if image_points.shape[0] != _MODEL_POINTS_3D.shape[0]:
        raise ValueError(f"Expected {_MODEL_POINTS_3D.shape[0]} keypoints, got {image_points.shape[0]}")

    # Near-frontal, near-symmetric keypoint sets are a classic ambiguous case for
    # solvePnP (it can converge to a ~180deg-flipped solution that fits the 2D
    # points almost as well). Seeding with a frontal initial guess resolves it.
    rvec_init = np.zeros((3, 1))
    tvec_init = np.array([[0.0], [0.0], [focal_length]])
    success, rvec, _tvec = cv2.solvePnP(
        _MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
        rvec_init, tvec_init, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    # Decompose rotation matrix -> Euler angles (yaw, pitch, roll), yaw/pitch only needed here.
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw = 0.0
    return float(yaw), float(pitch)


def pose_confidence_weight(yaw_deg: float, pitch_deg: float, cfg: QualityGateConfig) -> float:
    """
    Doc §2 (new): pose angle carried forward as a confidence weight instead
    of a binary pass/fail. 1.0 within `frontal_deg_for_full_weight`, linearly
    ramping down to 0.0 at `max_yaw_pitch_deg` (the hard-reject boundary).
    """
    worst_angle = max(abs(yaw_deg), abs(pitch_deg))
    if worst_angle <= cfg.frontal_deg_for_full_weight:
        return 1.0
    if worst_angle >= cfg.max_yaw_pitch_deg:
        return 0.0
    span = cfg.max_yaw_pitch_deg - cfg.frontal_deg_for_full_weight
    return float(1.0 - (worst_angle - cfg.frontal_deg_for_full_weight) / span)


def crop_with_bounds(image_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = bbox
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    return image_bgr[y0:y1, x0:x1]


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #
class QualityGate:
    def __init__(self, cfg: Optional[QualityGateConfig] = None, face_detector: Optional[FaceDetectorFn] = None):
        self.cfg = cfg or QualityGateConfig()
        # Dependency-injectable so this stage is unit-testable without mediapipe installed.
        self._face_detector = face_detector

    @property
    def face_detector(self) -> FaceDetectorFn:
        if self._face_detector is None:
            self._face_detector = default_mediapipe_detector(self.cfg.face_detector_model_path)
        return self._face_detector

    def run(self, raw_frame: RawFrame) -> QualityResult:
        image = raw_frame.image_bgr
        detections = self.face_detector(image)
        if not detections:
            return QualityResult(passed=False, pose_confidence=0.0, blur_score=0.0,
                                  reject_reason="no_face_detected")

        # Multiple faces shouldn't occur given per-participant ingestion (Chunk 2),
        # but if they do, take the largest (most likely the actual speaker tile).
        det = max(detections, key=lambda d: d.bbox[2] * d.bbox[3])
        _, _, bw, bh = det.bbox

        if bw < self.cfg.min_face_size_px or bh < self.cfg.min_face_size_px:
            return QualityResult(passed=False, pose_confidence=0.0, blur_score=0.0,
                                  face_bbox=det.bbox, reject_reason="face_too_small")

        face_crop = crop_with_bounds(image, det.bbox)
        if face_crop.size == 0:
            return QualityResult(passed=False, pose_confidence=0.0, blur_score=0.0,
                                  face_bbox=det.bbox, reject_reason="invalid_crop")

        blur = laplacian_variance(face_crop)
        if blur < self.cfg.laplacian_var_threshold:
            return QualityResult(passed=False, pose_confidence=0.0, blur_score=blur,
                                  face_bbox=det.bbox, reject_reason="blurry")

        brightness = mean_brightness(face_crop)
        if not (self.cfg.min_mean_brightness <= brightness <= self.cfg.max_mean_brightness):
            return QualityResult(passed=False, pose_confidence=0.0, blur_score=blur,
                                  face_bbox=det.bbox, reject_reason="bad_lighting")

        try:
            yaw, pitch = estimate_pose_angles(det.keypoints, image.shape)
        except ValueError:
            return QualityResult(passed=False, pose_confidence=0.0, blur_score=blur,
                                  face_bbox=det.bbox, reject_reason="pose_estimation_failed")

        worst_angle = max(abs(yaw), abs(pitch))
        if worst_angle > self.cfg.max_yaw_pitch_deg:
            return QualityResult(passed=False, pose_confidence=0.0, blur_score=blur,
                                  face_bbox=det.bbox, reject_reason="extreme_pose")

        weight = pose_confidence_weight(yaw, pitch, self.cfg)
        return QualityResult(passed=True, pose_confidence=weight, blur_score=blur, face_bbox=det.bbox)
