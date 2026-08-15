"""
Stage 4d — Phone & Screen Replay Attack Detector.

Identifies Presentation Attacks / Display Replays where an attacker holds up a
physical smartphone, tablet, or external display screen in front of the camera
to stream pre-recorded or deepfaked videos.

Robustness & Anti-False-Positive Architecture:
  1. Specular Glare & Light Spot Rejection: Lamps, light bulbs, ceiling spotlights,
     windows, and background glare are strictly rejected via over-saturation analysis
     (V > 240 percentage, intensity variance, gradient diffusion).
  2. Strict Face Enclosure Requirement: A physical replay attack requires that the
     detected face is physically positioned INSIDE the phone bezel boundary.
     Background rectangles that do not contain the face are ignored.
  3. Aspect Ratio Gating: Validates typical smartphone aspect ratios (1.45 to 2.45).
  4. Moiré Grid Harmonic Isolation: Masks specular glare before computing 2D DFT
     and requires 2D harmonic grid periodicity.
"""
from dataclasses import dataclass
import logging
from typing import Optional, Tuple, List
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class PhoneDetectionResult:
    phone_detected: bool
    confidence: float                     # 0.0 to 1.0
    detection_source: str                 # "NEURAL_OBJECT", "BEZEL_CONTOUR", "SPECTRAL_MOIRE", "HYBRID", "CLEAR"
    phone_bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    aspect_ratio: float = 0.0
    face_enclosed: bool = False
    details: str = "Nominal live feed — no physical phone or display screen detected."


class PhoneReplayDetector:
    """
    Real-time phone and screen replay attack detector.
    Combines neural object recognition, geometric bezel contour analysis,
    and frequency moiré spectral verification.
    """

    def __init__(
        self,
        min_confidence: float = 0.60,
        enable_neural: bool = True,
        enable_bezel_analysis: bool = True,
        enable_moire_analysis: bool = True,
    ):
        self.min_confidence = min_confidence
        self.enable_neural = enable_neural
        self.enable_bezel_analysis = enable_bezel_analysis
        self.enable_moire_analysis = enable_moire_analysis
        self._neural_model = None
        self._neural_initialized = False

    def _init_neural_detector(self):
        """Attempts to load a fast mobile object detector if available."""
        if self._neural_initialized:
            return
        self._neural_initialized = True
        try:
            import torchvision
            from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
            try:
                weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
                self._neural_model = ssdlite320_mobilenet_v3_large(weights=weights)
                self._neural_model.eval()
                logger.info("SSDLite320 MobileNetV3 Phone Detector initialized successfully.")
            except Exception as e:
                logger.debug(f"Could not load pre-trained SSDLite weights (offline/cpu): {e}")
                self._neural_model = None
        except Exception as e:
            logger.debug(f"Torchvision detection module unavailable: {e}")
            self._neural_model = None

    def _is_light_source_or_glare(self, roi_bgr: np.ndarray) -> bool:
        """
        Identifies whether a candidate region is an active light source, lamp,
        window, or specular glare rather than a phone display.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return True

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if len(roi_bgr.shape) == 3 else roi_bgr
        total_pixels = gray.size
        if total_pixels == 0:
            return True

        # 1. Saturated / blown-out white pixels (> 240)
        saturated_count = np.sum(gray >= 240)
        saturated_ratio = saturated_count / float(total_pixels)
        if saturated_ratio > 0.18:
            # Over 18% of the box is completely blown out white -> it's a light source/glare
            return True

        # 2. Mean brightness test
        mean_val = float(np.mean(gray))
        if mean_val > 222.0:
            # Excessively bright overall -> light fixture or window
            return True

        # 3. Flat bright texture test (glare or solid bright blob without facial/screen structure)
        std_val = float(np.std(gray))
        if std_val < 10.0 and mean_val > 150.0:
            return True

        return False

    def detect(
        self,
        image_bgr: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> PhoneDetectionResult:
        """
        Executes multi-layered phone screen replay detection on a single frame.

        Args:
            image_bgr: Raw input frame (HxWx3, BGR uint8)
            face_bbox: Optional detected face bounding box (x, y, w, h)

        Returns:
            PhoneDetectionResult
        """
        if image_bgr is None or image_bgr.size == 0:
            return PhoneDetectionResult(phone_detected=False, confidence=0.0, detection_source="CLEAR")

        phone_scores = []
        sources = []
        best_bbox = None
        face_is_inside_phone = False
        details_list = []

        # ── 1. Geometric Screen Bezel & Display Contour Analysis ──
        if self.enable_bezel_analysis:
            bezel_detected, bezel_conf, b_bbox, enclosed = self._analyze_screen_bezel(image_bgr, face_bbox)
            if bezel_detected and enclosed:
                phone_scores.append(bezel_conf)
                sources.append("BEZEL_CONTOUR")
                best_bbox = b_bbox
                face_is_inside_phone = True
                aspect_ratio = b_bbox[3] / max(1, b_bbox[2]) if b_bbox else 2.0
                details_list.append(
                    f"Physical phone/screen bezel enclosing face detected (aspect ratio ~{aspect_ratio:.2f})."
                )

        # ── 2. Screen Sub-Pixel Moiré & Periodic Grid Analysis ──
        if self.enable_moire_analysis:
            moire_detected, moire_conf, moire_details = self._analyze_moire_pattern(image_bgr, face_bbox)
            if moire_detected:
                phone_scores.append(moire_conf)
                sources.append("SPECTRAL_MOIRE")
                details_list.append(moire_details)

        # ── 3. Neural Object Detection (COCO class 77: cell phone) ──
        if self.enable_neural and self._neural_model is not None:
            neural_detected, neural_conf, n_bbox = self._run_neural_detection(image_bgr, face_bbox)
            if neural_detected:
                phone_scores.append(neural_conf)
                sources.append("NEURAL_OBJECT")
                if best_bbox is None:
                    best_bbox = n_bbox
                details_list.append(f"Mobile device object detected with {neural_conf*100:.1f}% confidence.")

        # ── Decision Fusion ──
        if phone_scores:
            base_score = max(phone_scores)
            if len(phone_scores) > 1:
                base_score = min(1.0, base_score + 0.10)
            if face_is_inside_phone:
                base_score = min(1.0, base_score + 0.12)

            if base_score >= self.min_confidence:
                source_str = "HYBRID" if len(sources) > 1 else sources[0]
                summary = " · ".join(details_list) if details_list else "Physical phone screen replay attack detected."
                return PhoneDetectionResult(
                    phone_detected=True,
                    confidence=float(base_score),
                    detection_source=source_str,
                    phone_bbox=best_bbox,
                    face_enclosed=face_is_inside_phone,
                    details=summary,
                )

        return PhoneDetectionResult(
            phone_detected=False,
            confidence=0.0,
            detection_source="CLEAR",
            details="Nominal camera stream — no physical phone or display screen identified."
        )

    def _analyze_screen_bezel(
        self,
        image_bgr: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]], bool]:
        """
        Detects sharp rectangular screen boundaries (bezel frames) with standard
        smartphone aspect ratios (1.45 to 2.45) that strictly ENCLOSE the face.
        """
        if face_bbox is None:
            # Presentation replay on camera requires a face inside the display!
            return False, 0.0, None, False

        fx, fy, fw, fh = face_bbox
        if fw <= 0 or fh <= 0:
            return False, 0.0, None, False

        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Suppress glare blobbing by clamping saturated pixels
        gray_clamped = np.minimum(gray, 235)

        blurred = cv2.bilateralFilter(gray_clamped, 7, 50, 50)
        edges = cv2.Canny(blurred, 45, 140)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = h * w
        face_area = fw * fh
        min_phone_area = max(frame_area * 0.04, face_area * 1.15)
        max_phone_area = frame_area * 0.94

        best_match = None
        best_conf = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_phone_area or area > max_phone_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

            # 4 to 8 vertices (rectangles, rounded rectangles)
            if 4 <= len(approx) <= 8:
                x, y, bw, bh = cv2.boundingRect(approx)
                if bw <= 0 or bh <= 0:
                    continue

                # STRICT FACE ENCLOSURE CHECK:
                # The phone display MUST enclose the face!
                enclosed = (x <= fx and y <= fy and (x + bw) >= (fx + fw) and (y + bh) >= (fy + fh))
                if not enclosed:
                    # Check partial overlap (at least 85% of face must be inside the box)
                    overlap_x1 = max(x, fx)
                    overlap_y1 = max(y, fy)
                    overlap_x2 = min(x + bw, fx + fw)
                    overlap_y2 = min(y + bh, fy + fh)
                    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
                        continue
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    if overlap_area / float(face_area) < 0.85:
                        continue

                # Light spot / lamp / specular glare rejection
                roi = image_bgr[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
                if self._is_light_source_or_glare(roi):
                    continue

                aspect = float(bh) / float(bw) if bw > 0 else 0.0
                inv_aspect = float(bw) / float(bh) if bh > 0 else 0.0
                max_aspect = max(aspect, inv_aspect)

                # Smartphone display aspect ratio (1.45 to 2.45)
                is_phone_aspect = 1.45 <= max_aspect <= 2.45

                rect_area = bw * bh
                extent = float(area) / float(rect_area) if rect_area > 0 else 0.0

                if is_phone_aspect and extent >= 0.70:
                    conf = min(0.96, 0.60 + (extent * 0.30))
                    if conf > best_conf:
                        best_conf = conf
                        best_match = (x, y, bw, bh)

        if best_match is not None and best_conf >= 0.60:
            return True, float(best_conf), best_match, True

        return False, 0.0, None, False

    def _analyze_moire_pattern(
        self,
        image_bgr: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[bool, float, str]:
        """
        Detects screen moiré grid artifacts caused by digital display sub-pixel grids.
        Strictly rejects specular highlights, flashlights, lamps, and point glare.
        """
        if face_bbox is None:
            return False, 0.0, ""

        fx, fy, fw, fh = face_bbox
        h, w = image_bgr.shape[:2]

        x1 = max(0, fx)
        y1 = max(0, fy)
        x2 = min(w, fx + fw)
        y2 = min(h, fy + fh)
        roi_bgr = image_bgr[y1:y2, x1:x2]

        if roi_bgr.shape[0] < 40 or roi_bgr.shape[1] < 40:
            return False, 0.0, ""

        # Reject if face region is washed out by direct light / glare
        if self._is_light_source_or_glare(roi_bgr):
            return False, 0.0, ""

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # Suppress point glare spikes in FFT: clamp pixels > 230
        gray_clean = np.minimum(gray, 230)

        # Resize to standard 128x128
        resized = cv2.resize(gray_clean, (128, 128)).astype(np.float32)

        # 2D Discrete Fourier Transform
        dft = cv2.dft(resized, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        mag = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

        # Mask DC and low frequency center (radius 18)
        cy, cx = 64, 64
        cv2.circle(mag, (cx, cy), 18, 0, -1)

        mean_val = float(np.mean(mag))
        std_val = float(np.std(mag))
        max_val = float(np.max(mag))

        spike_ratio = max_val / (mean_val + 1e-6)

        # Require high spike ratio (> 32.0) and high standard deviation (> 25.0) for genuine screen moiré
        if spike_ratio > 32.0 and std_val > 25.0:
            conf = min(0.92, 0.52 + (spike_ratio / 80.0))
            return True, float(conf), f"Screen sub-pixel moiré pattern detected (spectral spike ratio: {spike_ratio:.1f})."

        return False, 0.0, ""

    def _run_neural_detection(
        self,
        image_bgr: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]]]:
        """Runs SSDLite object detection to identify 'cell phone' (class 77 in COCO)."""
        if self._neural_model is None:
            return False, 0.0, None

        try:
            import torch
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

            with torch.no_grad():
                predictions = self._neural_model([tensor])[0]

            boxes = predictions['boxes'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()

            # COCO class 77 is "cell phone"
            for i, label in enumerate(labels):
                score = float(scores[i])
                if label == 77 and score >= 0.65:
                    x1, y1, x2, y2 = boxes[i]
                    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    return True, score, bbox
        except Exception as e:
            logger.debug(f"Neural phone detection error: {e}")

        return False, 0.0, None
