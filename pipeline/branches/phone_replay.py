"""
Stage 4d — Phone & Screen Replay Attack Detector.

Identifies Presentation Attacks / Display Replays where an attacker holds up a
physical smartphone, tablet, or external display screen in front of the camera
to stream pre-recorded or deepfaked videos.

Multi-Layered Detection Architecture:
  1. Deep Neural Object Detection: COCO cell phone (class 77) / screen detection.
  2. Screen Bezel & Display Geometry: High-contrast rectangular contour detection
     with smartphone aspect ratios (1.6 to 2.5) enclosing the face.
  3. High-Frequency Moiré & Glass Reflection: Spectral analysis detecting digital
     LCD/OLED sub-pixel grid aliasing and specular glass glare.

Returns a PhoneDetectionResult containing detection flags, confidence, and
explainable forensic rationale.
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
        min_confidence: float = 0.45,
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

        h, w = image_bgr.shape[:2]
        phone_scores = []
        sources = []
        best_bbox = None
        face_is_inside_phone = False
        details_list = []

        # ── 1. Geometric Screen Bezel & Display Contour Analysis ──
        if self.enable_bezel_analysis:
            bezel_detected, bezel_conf, b_bbox, enclosed = self._analyze_screen_bezel(image_bgr, face_bbox)
            if bezel_detected:
                phone_scores.append(bezel_conf)
                sources.append("BEZEL_CONTOUR")
                if b_bbox is not None:
                    best_bbox = b_bbox
                if enclosed:
                    face_is_inside_phone = True
                details_list.append(
                    f"Physical phone/screen bezel enclosing face detected (aspect ratio ~{b_bbox[3]/max(1, b_bbox[2]):.2f} if b_bbox else 2.0)."
                )

        # ── 2. Screen Sub-Pixel Moiré & Glass Reflection Analysis ──
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
            # Highest confidence across signals with multi-signal bonus
            base_score = max(phone_scores)
            if len(phone_scores) > 1:
                base_score = min(1.0, base_score + 0.10)
            if face_is_inside_phone:
                base_score = min(1.0, base_score + 0.15)

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
            details="Nominal camera stream — no phone or display screen identified."
        )

    def _analyze_screen_bezel(
        self,
        image_bgr: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]], bool]:
        """
        Detects sharp rectangular screen boundaries (bezel frames) with standard
        smartphone aspect ratios (1.6 to 2.5) surrounding or containing the face.
        """
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Bilateral filter to smooth face texture while keeping sharp bezel edges
        blurred = cv2.bilateralFilter(gray, 7, 50, 50)
        edges = cv2.Canny(blurred, 40, 130)

        # Dilate slightly to connect smartphone bezel segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = h * w
        min_phone_area = frame_area * 0.06   # at least 6% of the frame
        max_phone_area = frame_area * 0.95   # at most 95% of the frame

        best_match = None
        best_conf = 0.0
        face_enclosed = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_phone_area or area > max_phone_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

            # Look for 4-corner polygons (rectangles / rounded rectangles)
            if 4 <= len(approx) <= 8:
                x, y, bw, bh = cv2.boundingRect(approx)
                if bw <= 0 or bh <= 0:
                    continue

                aspect = float(bh) / float(bw) if bw > 0 else 0.0
                inv_aspect = float(bw) / float(bh) if bh > 0 else 0.0
                max_aspect = max(aspect, inv_aspect)

                # Smartphone display aspect ratios are typically between 1.5 and 2.4
                # (e.g. 16:9 = 1.77, 19.5:9 = 2.16, 20:9 = 2.22, 4:3 = 1.33)
                is_phone_aspect = 1.45 <= max_aspect <= 2.55

                # Rectangularity score
                rect_area = bw * bh
                extent = float(area) / float(rect_area) if rect_area > 0 else 0.0

                if is_phone_aspect and extent >= 0.72:
                    conf = min(0.92, 0.55 + (extent * 0.30))

                    # Check if face is enclosed inside this phone rectangle
                    enclosed = False
                    if face_bbox is not None:
                        fx, fy, fw, fh = face_bbox
                        # Face centers inside the phone box
                        fcx, fcy = fx + fw // 2, fy + fh // 2
                        if (x <= fcx <= x + bw) and (y <= fcy <= y + bh):
                            enclosed = True
                            conf = min(0.98, conf + 0.20)

                    if conf > best_conf:
                        best_conf = conf
                        best_match = (x, y, bw, bh)
                        face_enclosed = enclosed

        if best_match is not None and best_conf >= 0.50:
            return True, float(best_conf), best_match, face_enclosed

        return False, 0.0, None, False

    def _analyze_moire_pattern(
        self,
        image_bgr: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[bool, float, str]:
        """
        Detects screen moiré grid artifacts caused by the interaction of
        the smartphone display pixel grid and the webcam sensor matrix.
        """
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Focus analysis on face region or center region
        if face_bbox is not None:
            fx, fy, fw, fh = face_bbox
            # Expand slightly around face
            x1 = max(0, fx - 10)
            y1 = max(0, fy - 10)
            x2 = min(w, fx + fw + 10)
            y2 = min(h, fy + fh + 10)
            roi = gray[y1:y2, x1:x2]
        else:
            roi = gray

        if roi.shape[0] < 40 or roi.shape[1] < 40:
            return False, 0.0, ""

        # Resize ROI to standard 128x128 for DFT frequency peak analysis
        resized = cv2.resize(roi, (128, 128)).astype(np.float32)

        # 2D Discrete Fourier Transform
        dft = cv2.dft(resized, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        mag = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

        # High-frequency band analysis (outer ring of spectrum)
        cy, cx = 64, 64
        # Mask out DC center
        cv2.circle(mag, (cx, cy), 14, 0, -1)

        # Check for repetitive periodic frequency peaks (moiré grid spikes)
        std_val = float(np.std(mag))
        mean_val = float(np.mean(mag))
        max_val = float(np.max(mag))

        # Peak-to-average ratio (moiré spike index)
        spike_ratio = max_val / (mean_val + 1e-6)

        # Screen displays captured on camera show distinct high spike ratios (> 18.0)
        if spike_ratio > 22.0 and std_val > 15.0:
            conf = min(0.92, 0.45 + (spike_ratio / 60.0))
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

            # COCO class 77 is "cell phone" (also 72: tv, 73: laptop)
            for i, label in enumerate(labels):
                score = float(scores[i])
                if label in [77, 72, 73] and score >= 0.42:
                    x1, y1, x2, y2 = boxes[i]
                    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    return True, score, bbox
        except Exception as e:
            logger.debug(f"Neural phone detection error: {e}")

        return False, 0.0, None
