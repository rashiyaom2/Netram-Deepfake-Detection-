"""
Stage 3 — Face Detection & Alignment (doc §3).

Runs the FULL landmark model (MediaPipe Face Mesh, 468 points) -- a
heavier/more precise model than Stage 2's cheap 6-keypoint QC detector.
This is intentional: Stage 2 just needs to gate frames fast, Stage 3 needs
accurate geometry for pixel-level artifact detection downstream.

Pipeline:
  1. Landmark detection
  2. Eye-line rotation (levels the face so eyes sit on a horizontal plane)
  3. Margin padding (20-30% border -- artifacts cluster at chin/hairline/neck)
  4. Resize to model input size (224x224 or 299x299)

Also produces a separate mouth crop (for the §4a audio-visual sync branch)
and returns all landmarks re-projected into the final aligned/resized crop's
coordinate space, so Stage 5's frame-to-frame jitter tracking is comparing
like-for-like coordinates.
"""
import threading
from dataclasses import dataclass
from math import atan2, degrees
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import cv2

from pipeline.config import AlignmentConfig
from pipeline.types import RawFrame, QualityResult, AlignedFace


# --------------------------------------------------------------------------- #
# MediaPipe Face Mesh landmark indices used by this stage.
# (Standard indices for the 468-point topology; RetinaFace users should map
# their 5-point output to at least the eye/mouth indices below.)
# --------------------------------------------------------------------------- #
LEFT_EYE_IDX = (33, 133)     # outer, inner corner
RIGHT_EYE_IDX = (362, 263)   # outer, inner corner
MOUTH_LEFT_IDX = 61
MOUTH_RIGHT_IDX = 291
MOUTH_TOP_IDX = 13
MOUTH_BOTTOM_IDX = 14
FACE_OVAL_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

NAMED_LANDMARKS_FOR_OUTPUT = {
    "left_eye_outer": 33, "left_eye_inner": 133,
    "left_eye_top": 159, "left_eye_bottom": 145,
    "right_eye_outer": 362, "right_eye_inner": 263,
    "right_eye_top": 386, "right_eye_bottom": 374,
    "mouth_left": 61, "mouth_right": 291,
    "mouth_top": 13, "mouth_bottom": 14,
    "nose_tip": 1, "chin": 152,
}

LandmarkerFn = Callable[[np.ndarray], Optional[List[Tuple[float, float]]]]  # returns pixel coords, len 468, or None


_GLOBAL_LANDMARKER = None
_LANDMARKER_LOCK = threading.Lock()

def default_mediapipe_face_mesh(model_path: str = "models/face_landmarker.task") -> LandmarkerFn:
    """Thread-safe cached MediaPipe FaceLandmarker Task (468 points)."""
    global _GLOBAL_LANDMARKER
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    with _LANDMARKER_LOCK:
        if _GLOBAL_LANDMARKER is None:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3,
                output_face_blendshapes=False,
            )
            _GLOBAL_LANDMARKER = vision.FaceLandmarker.create_from_options(options)

    landmarker = _GLOBAL_LANDMARKER

    def _detect(image_bgr: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with _LANDMARKER_LOCK:
            results = landmarker.detect(mp_image)
        if not results.face_landmarks:
            return None
        lm = results.face_landmarks[0]
        return [(pt.x * w, pt.y * h) for pt in lm]


    return _detect


# --------------------------------------------------------------------------- #
# Pure, independently-testable geometry helpers
# --------------------------------------------------------------------------- #
def eye_centers(landmarks_px: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    lx = np.mean([landmarks_px[i][0] for i in LEFT_EYE_IDX])
    ly = np.mean([landmarks_px[i][1] for i in LEFT_EYE_IDX])
    rx = np.mean([landmarks_px[i][0] for i in RIGHT_EYE_IDX])
    ry = np.mean([landmarks_px[i][1] for i in RIGHT_EYE_IDX])
    return (float(lx), float(ly)), (float(rx), float(ry))


def rotation_angle_deg(left_eye: Tuple[float, float], right_eye: Tuple[float, float]) -> float:
    """Angle (degrees) to feed into cv2.getRotationMatrix2D to level the eye line."""
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    return degrees(atan2(dy, dx))


def rotate_image_and_points(image_bgr: np.ndarray, points_px: List[Tuple[float, float]],
                             angle_deg: float, center: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_bgr.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_image = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_LINEAR)
    pts = np.array([points_px], dtype=np.float32)
    rotated_points = cv2.transform(pts, M)[0]
    return rotated_image, rotated_points


def bbox_with_margin(points: np.ndarray, margin_pct: float, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Tight bbox around `points`, expanded by margin_pct on each side, clipped to image bounds."""
    h, w = image_shape[:2]
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    x_max, y_max = points[:, 0].max(), points[:, 1].max()
    bw, bh = x_max - x_min, y_max - y_min
    mx, my = bw * margin_pct, bh * margin_pct
    x0 = int(max(0, x_min - mx))
    y0 = int(max(0, y_min - my))
    x1 = int(min(w, x_max + mx))
    y1 = int(min(h, y_max + my))
    return x0, y0, x1, y1


def crop_resize_normalize(image_bgr: np.ndarray, bbox: Tuple[int, int, int, int],
                           output_size: Tuple[int, int]) -> Tuple[np.ndarray, float, float]:
    """
    Crops `bbox`, resizes to `output_size`, converts BGR->RGB, normalizes to [0,1] float32.
    Returns (crop, scale_x, scale_y) where scale_* map crop-space coords to output-space coords,
    so landmarks can be re-projected into the final output coordinate system.
    """
    x0, y0, x1, y1 = bbox
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("Empty crop from bbox; upstream bbox computation produced degenerate region")
    ch, cw = crop.shape[:2]
    resized = cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    scale_x = output_size[0] / cw
    scale_y = output_size[1] / ch
    return rgb, scale_x, scale_y


def project_points_to_output_space(points: np.ndarray, bbox: Tuple[int, int, int, int],
                                    scale_x: float, scale_y: float) -> np.ndarray:
    x0, y0, _x1, _y1 = bbox
    out = points.copy()
    out[:, 0] = (out[:, 0] - x0) * scale_x
    out[:, 1] = (out[:, 1] - y0) * scale_y
    return out


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #
class FaceAligner:
    def __init__(self, cfg: Optional[AlignmentConfig] = None, landmarker: Optional[LandmarkerFn] = None):
        self.cfg = cfg or AlignmentConfig()
        self._landmarker = landmarker

    @property
    def landmarker(self) -> LandmarkerFn:
        if self._landmarker is None:
            self._landmarker = default_mediapipe_face_mesh(self.cfg.face_landmarker_model_path)
        return self._landmarker

    def align(self, raw_frame: RawFrame, quality: QualityResult) -> Optional[AlignedFace]:
        if not quality.passed:
            return None

        landmarks_px = self.landmarker(raw_frame.image_bgr)
        if landmarks_px is None:
            return None
        landmarks_arr = np.array(landmarks_px, dtype=np.float32)

        left_eye, right_eye = eye_centers(landmarks_px)
        angle = rotation_angle_deg(left_eye, right_eye)
        center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)

        rotated_image, rotated_landmarks = rotate_image_and_points(
            raw_frame.image_bgr, landmarks_px, angle, center
        )

        # --- Full face crop ---
        oval_pts = rotated_landmarks[FACE_OVAL_IDX]
        face_bbox = bbox_with_margin(oval_pts, self.cfg.margin_pct, rotated_image.shape)
        try:
            face_crop, sx, sy = crop_resize_normalize(rotated_image, face_bbox, self.cfg.output_size)
        except ValueError:
            return None
        face_space_landmarks = project_points_to_output_space(rotated_landmarks, face_bbox, sx, sy)

        # --- Mouth crop (for §4a audio-visual sync branch) ---
        mouth_idx = [MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX, MOUTH_TOP_IDX, MOUTH_BOTTOM_IDX]
        mouth_pts = rotated_landmarks[mouth_idx]
        mouth_bbox = bbox_with_margin(mouth_pts, self.cfg.mouth_margin_pct, rotated_image.shape)
        try:
            mouth_crop, _msx, _msy = crop_resize_normalize(rotated_image, mouth_bbox, self.cfg.mouth_crop_size)
        except ValueError:
            mouth_crop = np.zeros((*self.cfg.mouth_crop_size, 3), dtype=np.float32)

        named_landmarks: Dict[str, Tuple[float, float]] = {
            name: (float(face_space_landmarks[idx][0]), float(face_space_landmarks[idx][1]))
            for name, idx in NAMED_LANDMARKS_FOR_OUTPUT.items()
        }

        return AlignedFace(
            participant_id=raw_frame.participant_id,
            frame_idx=raw_frame.frame_idx,
            timestamp=raw_frame.timestamp,
            face_crop=face_crop,
            mouth_crop=mouth_crop,
            landmarks=named_landmarks,
            pose_confidence=quality.pose_confidence,
            audio_window=raw_frame.audio_window,
        )
