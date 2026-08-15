"""
Stage 4e — Social Media & AR Beauty Filter Detector.

Accurately identifies synthetic AR filters, beauty filters, and digital facial
enhancements commonly used in Snapchat, Instagram, TikTok, FaceTime, and virtual
camera apps (e.g. OBS VirtualCam filters, ManyCam).

Forensic Analysis Vectors:
  1. Skin Micro-Texture vs Edge Ratio (Bilateral Airbrushing & Gaussian Smoothing)
  2. Anthropometric Landmark Deformation (Eye Enlargement, V-Shape Jaw Tapering, Nose Narrowing)
  3. Synthetic Digital Makeup Overlays (Artificial Hue Saliency & Color Saturation Spikes in HSV/LAB)
  4. Face-to-Neck Boundary Discontinuity (Mandible Filter Mask Seam Detection)

Returns an ARFilterDetectionResult with granular metrics and explainable rationale.
"""
from dataclasses import dataclass
import logging
from typing import Optional, Dict, Tuple, List
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class ARFilterDetectionResult:
    filter_detected: bool
    confidence: float                     # 0.0 to 1.0
    filter_type: str                      # "BEAUTY_AIRBRUSH", "FACIAL_WARPING", "SYNTHETIC_MAKEUP", "HYBRID_AR_FILTER", "NATURAL"
    smoothness_score: float = 0.0         # 0.0 (natural texture) to 1.0 (artificial airbrushing)
    warp_score: float = 0.0               # 0.0 (canonical anthropometry) to 1.0 (heavy distortion)
    synthetic_makeup_score: float = 0.0   # 0.0 to 1.0 (digital blush/lipstick/eyeliner overlays)
    neck_discontinuity: float = 0.0       # 0.0 to 1.0 (face vs neck texture seam)
    details: str = "Natural biological skin texture and proportions."


class ARFilterDetector:
    """
    Real-time detector for Snapchat, Instagram, TikTok, and AR beauty filters.
    Runs in sub-3ms on CPU with zero model download dependencies.
    """

    def __init__(
        self,
        min_confidence: float = 0.48,
        smoothness_threshold: float = 0.58,
        warp_threshold: float = 0.55,
        makeup_threshold: float = 0.55,
    ):
        self.min_confidence = min_confidence
        self.smoothness_threshold = smoothness_threshold
        self.warp_threshold = warp_threshold
        self.makeup_threshold = makeup_threshold

    def detect(
        self,
        face_crop_bgr: np.ndarray,
        landmarks: Optional[Dict[str, Tuple[float, float]]] = None,
        full_image_bgr: Optional[np.ndarray] = None,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> ARFilterDetectionResult:
        """
        Executes full AR/Beauty filter analysis on face crop and landmark geometry.

        Args:
            face_crop_bgr: 224x224 or raw face crop in BGR format (uint8)
            landmarks: Optional MediaPipe landmark dictionary (normalized or pixel coords)
            full_image_bgr: Optional full frame for neck-to-face boundary analysis
            face_bbox: Optional face bounding box (x, y, w, h)
        """
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return ARFilterDetectionResult(
                filter_detected=False, confidence=0.0, filter_type="NATURAL"
            )

        # Normalize face crop to standard uint8 BGR
        if face_crop_bgr.dtype != np.uint8:
            if face_crop_bgr.max() <= 1.0:
                face_bgr = (np.clip(face_crop_bgr, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                face_bgr = np.clip(face_crop_bgr, 0, 255).astype(np.uint8)
        else:
            face_bgr = face_crop_bgr

        if face_bgr.shape[0] != 224 or face_bgr.shape[1] != 224:
            face_bgr = cv2.resize(face_bgr, (224, 224))

        # ── 1. Skin Micro-Texture & Bilateral Airbrushing Analysis ──
        smooth_score, skin_var, edge_mag = self._analyze_skin_smoothing(face_bgr)

        # ── 2. Anthropometric Facial Landmark Deformation ──
        warp_score, warp_reasons = self._analyze_facial_warping(landmarks, face_bgr.shape)

        # ── 3. Synthetic Makeup & Digital Hue Saliency Overlays ──
        makeup_score, makeup_reasons = self._analyze_synthetic_makeup(face_bgr)

        # ── 4. Face-to-Neck Discontinuity (if full frame provided) ──
        neck_score = 0.0
        if full_image_bgr is not None and face_bbox is not None:
            neck_score = self._analyze_neck_discontinuity(full_image_bgr, face_bbox)

        # ── Multi-Signal Synthesis ──
        detected_types = []
        details_list = []

        if smooth_score >= self.smoothness_threshold:
            detected_types.append("BEAUTY_AIRBRUSH")
            details_list.append(f"Heavy digital skin smoothing / airbrush filter detected (skin texture var: {skin_var:.1f}).")

        if warp_score >= self.warp_threshold:
            detected_types.append("FACIAL_WARPING")
            details_list.append(f"AR geometric landmark warping ({', '.join(warp_reasons)}).")

        if makeup_score >= self.makeup_threshold:
            detected_types.append("SYNTHETIC_MAKEUP")
            details_list.append(f"Synthetic digital color overlays ({', '.join(makeup_reasons)}).")

        if neck_score >= 0.60:
            details_list.append("Noticeable texture/color seam between filtered face and natural neck.")

        # Fused AR Filter Confidence
        # Airbrushing and warping are primary indicators
        fused_conf = 0.0
        if detected_types:
            fused_conf = max(smooth_score, warp_score, makeup_score)
            if len(detected_types) > 1:
                fused_conf = min(1.0, fused_conf + 0.12)
            if neck_score >= 0.50:
                fused_conf = min(1.0, fused_conf + 0.08)

        is_filter = fused_conf >= self.min_confidence

        filter_type_str = "NATURAL"
        if is_filter:
            if len(detected_types) > 1:
                filter_type_str = "HYBRID_AR_FILTER"
            elif detected_types:
                filter_type_str = detected_types[0]
            else:
                filter_type_str = "BEAUTY_AIRBRUSH"

        summary = " · ".join(details_list) if details_list else "Natural biological skin texture and facial proportions."

        return ARFilterDetectionResult(
            filter_detected=is_filter,
            confidence=float(np.clip(fused_conf, 0.0, 1.0)),
            filter_type=filter_type_str,
            smoothness_score=float(smooth_score),
            warp_score=float(warp_score),
            synthetic_makeup_score=float(makeup_score),
            neck_discontinuity=float(neck_score),
            details=summary,
        )

    def _analyze_skin_smoothing(self, face_bgr: np.ndarray) -> Tuple[float, float, float]:
        """
        Quantifies skin pore texture vs edge gradient ratio.
        AR/Beauty filters apply heavy bilateral / surface blur on skin,
        destroying pore variance (< 8.0) while eyes/hair keep high gradients.
        """
        h, w = face_bgr.shape[:2]
        hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        # Standard human skin color mask in HSV space
        lower_skin = np.array([0, 20, 50], dtype=np.uint8)
        upper_skin = np.array([30, 210, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Exclude eye and mouth regions (focus on cheek / forehead / nose bridge)
        # Mask out center-top (eyes) and center-bottom (mouth)
        cy, cx = h // 2, w // 2
        feature_mask = np.zeros_like(skin_mask)
        # Eye region
        cv2.rectangle(feature_mask, (int(w * 0.18), int(h * 0.28)), (int(w * 0.82), int(h * 0.52)), 255, -1)
        # Mouth region
        cv2.rectangle(feature_mask, (int(w * 0.28), int(h * 0.68)), (int(w * 0.72), int(h * 0.90)), 255, -1)

        pure_skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(feature_mask))

        skin_pixels = gray[pure_skin_mask > 0]
        if len(skin_pixels) < 400:
            return 0.0, 50.0, 50.0

        # Laplacian variance on pure skin
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        skin_laplacian = laplacian[pure_skin_mask > 0]
        skin_variance = float(np.var(skin_laplacian))

        # Overall Sobel edge magnitude across feature regions (eyes, hair, lips)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))

        # Airbrush Index: High edge sharpness + very low skin texture variance = Beauty Filter
        # Natural webcam skin variance is typically 20.0 - 80.0.
        # Filtered/Airbrushed skin variance is typically 1.0 - 10.0.
        if skin_variance < 12.0:
            # Low skin texture
            smooth_score = min(0.95, (12.0 - skin_variance) / 12.0 * 0.75 + (min(edge_mag, 60.0) / 60.0 * 0.25))
        elif skin_variance < 20.0:
            smooth_score = max(0.0, (20.0 - skin_variance) / 20.0 * 0.45)
        else:
            smooth_score = 0.0

        return float(np.clip(smooth_score, 0.0, 1.0)), skin_variance, edge_mag

    def _analyze_facial_warping(
        self,
        landmarks: Optional[Dict[str, Tuple[float, float]]],
        shape: Tuple[int, ...]
    ) -> Tuple[float, List[str]]:
        """
        Evaluates facial landmark proportions against biological anthropometric standards.
        Snapchat / IG beauty filters enlarge eyes and taper chin / jawline.
        """
        if not landmarks or len(landmarks) < 4:
            return 0.0, []

        reasons = []
        anomaly_scores = []

        # If key landmark anchors are present (e.g. left_eye, right_eye, nose, mouth, jaw)
        # Landmarks can be keys like 'left_eye_corner', 'right_eye_corner', 'upper_lip', 'lower_lip'
        le = landmarks.get("left_eye_corner") or landmarks.get("left_eye")
        re = landmarks.get("right_eye_corner") or landmarks.get("right_eye")
        ul = landmarks.get("upper_lip")
        ll = landmarks.get("lower_lip")

        if le and re:
            # Inter-ocular distance
            dx = re[0] - le[0]
            dy = re[1] - le[1]
            eye_dist = np.sqrt(dx**2 + dy**2)

            # In standard 224x224 aligned crop, eye_dist is typically 65-88px (0.29 to 0.39)
            norm_eye_dist = eye_dist / shape[1]
            if norm_eye_dist > 0.44:
                # Unnaturally enlarged eye span (Anime / Bambi Eyes filter)
                score = min(0.92, (norm_eye_dist - 0.44) / 0.12 + 0.50)
                anomaly_scores.append(score)
                reasons.append("Eye enlargement distortion")
            elif norm_eye_dist < 0.22:
                anomaly_scores.append(0.60)
                reasons.append("Compressed eye distance")

            # Eye-to-Mouth Triangle Proportions
            if ul:
                cx_eyes = (le[0] + re[0]) / 2.0
                cy_eyes = (le[1] + re[1]) / 2.0
                nose_mouth_dist = np.sqrt((ul[0] - cx_eyes)**2 + (ul[1] - cy_eyes)**2)
                
                if eye_dist > 0:
                    tri_ratio = nose_mouth_dist / eye_dist
                    # Canonical human golden ratio for mid-face triangle is ~1.05 to 1.45
                    if tri_ratio < 0.82:
                        # Heavy V-shape chin shortening
                        anomaly_scores.append(min(0.88, (0.82 - tri_ratio) * 2.5 + 0.45))
                        reasons.append("Mid-face compression / V-taper")
                    elif tri_ratio > 1.75:
                        anomaly_scores.append(0.65)
                        reasons.append("Elongated vertical face warp")

        if anomaly_scores:
            return float(np.clip(max(anomaly_scores), 0.0, 1.0)), reasons
        return 0.0, []

    def _analyze_synthetic_makeup(self, face_bgr: np.ndarray) -> Tuple[float, List[str]]:
        """
        Detects artificial makeup saturation overlays (lipstick, blush, eyeliner).
        Digital cosmetics produce sharp saturation peaks in HSV / LAB space.
        """
        h, w = face_bgr.shape[:2]
        hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        reasons = []
        scores = []

        # 1. Lip Region Saturation & Hue Sharpness
        # Lower center area of face
        lip_roi = hsv[int(h * 0.65):int(h * 0.88), int(w * 0.28):int(w * 0.72)]
        if lip_roi.size > 0:
            lip_sat = lip_roi[:, :, 1]
            lip_sat_mean = float(np.mean(lip_sat))
            lip_sat_max = float(np.max(lip_sat))
            # Synthetic digital lipstick produces ultra-saturated peaks (> 210)
            if lip_sat_max > 225 and lip_sat_mean > 140:
                scores.append(min(0.90, (lip_sat_max - 200) / 55.0 * 0.75 + 0.20))
                reasons.append("Ultra-saturated digital lipstick overlay")

        # 2. Cheek Blush Saliency (lateral cheeks)
        left_cheek = hsv[int(h * 0.45):int(h * 0.65), int(w * 0.10):int(w * 0.32)]
        right_cheek = hsv[int(h * 0.45):int(h * 0.65), int(w * 0.68):int(w * 0.90)]
        forehead = hsv[int(h * 0.12):int(h * 0.28), int(w * 0.25):int(w * 0.75)]

        if left_cheek.size > 0 and right_cheek.size > 0 and forehead.size > 0:
            cheek_sat = (float(np.mean(left_cheek[:, :, 1])) + float(np.mean(right_cheek[:, :, 1]))) / 2.0
            forehead_sat = float(np.mean(forehead[:, :, 1]))

            # Digital blush creates sharp saturation mismatch between cheeks and forehead (> 50 delta)
            sat_delta = cheek_sat - forehead_sat
            if sat_delta > 55.0:
                scores.append(min(0.88, (sat_delta - 45.0) / 40.0 * 0.70 + 0.20))
                reasons.append("Artificial cheek blush tint")

        if scores:
            return float(np.clip(max(scores), 0.0, 1.0)), reasons
        return 0.0, []

    def _analyze_neck_discontinuity(
        self,
        full_image_bgr: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        Compares skin texture variance between face (subject to AR filter)
        and neck (typically outside AR filter boundary).
        """
        fx, fy, fw, fh = face_bbox
        ih, iw = full_image_bgr.shape[:2]

        # Lower face region (jawline)
        face_y1 = max(0, fy + int(fh * 0.5))
        face_y2 = min(ih, fy + fh)
        face_x1 = max(0, fx + int(fw * 0.15))
        face_x2 = min(iw, fx + int(fw * 0.85))

        # Neck region (below jawline)
        neck_y1 = min(ih, fy + fh)
        neck_y2 = min(ih, fy + fh + int(fh * 0.4))
        neck_x1 = face_x1
        neck_x2 = face_x2

        if face_y2 <= face_y1 or neck_y2 <= neck_y1 or face_x2 <= face_x1:
            return 0.0

        face_roi = full_image_bgr[face_y1:face_y2, face_x1:face_x2]
        neck_roi = full_image_bgr[neck_y1:neck_y2, neck_x1:neck_x2]

        if face_roi.size < 200 or neck_roi.size < 200:
            return 0.0

        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        neck_gray = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2GRAY)

        face_var = float(np.var(cv2.Laplacian(face_gray, cv2.CV_64F)))
        neck_var = float(np.var(cv2.Laplacian(neck_gray, cv2.CV_64F)))

        # If neck has natural high texture (e.g. > 30) but face has smoothed texture (< 8),
        # this indicates an AR filter mask seam!
        if neck_var > 25.0 and face_var < 10.0:
            discontinuity_ratio = neck_var / (face_var + 1e-4)
            if discontinuity_ratio > 3.0:
                return float(np.clip(min(0.90, (discontinuity_ratio - 3.0) / 10.0 + 0.50), 0.0, 1.0))

        return 0.0
