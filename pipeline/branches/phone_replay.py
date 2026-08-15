"""
Stage 4d — Real-Time Phone & Screen Replay Attack Detector.

Identifies Presentation Attacks / Display Replays where an attacker holds up a
physical smartphone, tablet, or display screen in front of the camera
to stream pre-recorded or deepfaked videos.

Employs a multi-modal defense combining:
1. Deep Learning Object Detection (YOLOv8, COCO classes: cell phone, screen/tv, laptop)
2. Rectilinear Screen Bezel & Screen Contour Geometry (aspect ratio, face enclosure, extent)
3. High-Frequency Moiré Subpixel Grid 2D Spectral Analysis
4. Per-Participant Temporal History & Hysteresis Smoothing to prevent flickering
"""
from dataclasses import dataclass
import logging
import os
from collections import defaultdict, deque
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# COCO Class IDs for physical display presentation devices
COCO_CELL_PHONE_CLASS_ID = 67  # 'cell phone'
COCO_TV_MONITOR_CLASS_ID = 62  # 'tv / display monitor'
COCO_LAPTOP_CLASS_ID = 63      # 'laptop screen / tablet'
COCO_DISPLAY_CLASS_IDS = [67, 62, 63]


@dataclass
class PhoneDetectionResult:
    phone_detected: bool
    confidence: float                     # 0.0 to 1.0
    detection_source: str                 # "YOLOV8_NEURAL", "BEZEL_CONTOUR", "MOIRE_SPECTRAL", "HYBRID", "CLEAR"
    phone_bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    aspect_ratio: float = 0.0
    face_enclosed: bool = False
    details: str = "Nominal live feed — no physical phone or display screen detected."


class PhoneReplayDetector:
    """
    Real-time phone and screen replay attack detector powered by YOLOv8,
    rectilinear bezel geometry, and moiré subpixel spectral analysis.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        conf_threshold: float = 0.22,
        iou_threshold: float = 0.45,
        smoothing_window: int = 6,
        min_frames_present: int = 2,
        enable_bezel_analysis: bool = True,
        enable_moire_analysis: bool = True,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.smoothing_window = smoothing_window
        self.min_frames_present = min_frames_present
        self.enable_bezel_analysis = enable_bezel_analysis
        self.enable_moire_analysis = enable_moire_analysis

        # Per-participant state to prevent crosstalk in multi-person calls
        self.p_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=smoothing_window))
        self.p_confs: Dict[str, deque] = defaultdict(lambda: deque(maxlen=smoothing_window))
        self.p_sources: Dict[str, deque] = defaultdict(lambda: deque(maxlen=smoothing_window))
        self.p_bboxes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=smoothing_window))
        self.p_confirmed_state: Dict[str, bool] = defaultdict(bool)

        self._yolo_model = None
        self._yolo_initialized = False

        self._init_yolo_model()

    def reset_participant(self, participant_id: str):
        """Flushes participant history on leave or stream reset."""
        self.p_history.pop(participant_id, None)
        self.p_confs.pop(participant_id, None)
        self.p_sources.pop(participant_id, None)
        self.p_bboxes.pop(participant_id, None)
        self.p_confirmed_state.pop(participant_id, None)

    def _init_yolo_model(self):
        """Initializes YOLOv8 neural phone detector with automatic virtualenv resolution."""
        if self._yolo_initialized:
            return
        self._yolo_initialized = True

        candidate_paths = [
            self.model_path,
            "models/yolov8n.pt",
            "models/yolov8s.pt",
            "yolov8n.pt",
        ]
        chosen_path = next((p for p in candidate_paths if p and os.path.exists(p)), "yolov8n.pt")

        try:
            try:
                from ultralytics import YOLO
            except ImportError:
                import sys
                from pathlib import Path
                venv_site = Path(__file__).resolve().parents[2] / ".venv" / "Lib" / "site-packages"
                if venv_site.exists() and str(venv_site) not in sys.path:
                    sys.path.insert(0, str(venv_site))
                from ultralytics import YOLO

            self._yolo_model = YOLO(chosen_path)
            logger.info(f"YOLOv8 Phone Detector initialized successfully with {chosen_path}")
        except Exception as e:
            logger.warning(f"Could not load YOLOv8 model from {chosen_path}: {e}")
            self._yolo_model = None

    def detect(
        self,
        image_bgr: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        participant_id: str = "default"
    ) -> PhoneDetectionResult:
        """
        Executes multi-modal phone & screen replay detection on a single frame.
        Tracks state per-participant with temporal confirmation & hysteresis.
        """
        if image_bgr is None or image_bgr.size == 0:
            return PhoneDetectionResult(phone_detected=False, confidence=0.0, detection_source="CLEAR")

        instant_detected = False
        instant_conf = 0.0
        detection_source = "CLEAR"
        best_bbox = None
        face_inside = False
        aspect = 0.0
        details = "Nominal live feed — no physical phone or display screen detected."

        # ─── 1. Neural YOLOv8 Phone & Screen Detection ───
        yolo_detections = self._run_yolo_detection(image_bgr)
        if yolo_detections:
            best_det = max(yolo_detections, key=lambda d: d["conf"])
            best_bbox = best_det["bbox"]
            instant_conf = best_det["conf"]
            instant_detected = True
            detection_source = "YOLOV8_NEURAL"
            device_name = best_det.get("name", "smartphone display")

            if best_bbox and face_bbox:
                bx, by, bw, bh = best_bbox
                fx, fy, fw, fh = face_bbox
                ox1 = max(bx, fx)
                oy1 = max(by, fy)
                ox2 = min(bx + bw, fx + fw)
                oy2 = min(by + bh, fy + fh)
                if ox2 > ox1 and oy2 > oy1:
                    face_inside = True
                    instant_conf = max(instant_conf, 0.88)

            aspect = float(best_bbox[3]) / max(1, best_bbox[2]) if best_bbox else 1.8
            details = f"Physical {device_name} identified via YOLOv8 (conf: {instant_conf*100:.1f}%)."

        # ─── 2. Geometric Bezel & Screen Edge Contour Analysis ───
        if not instant_detected and self.enable_bezel_analysis and face_bbox:
            bezel_det, bezel_conf, b_bbox, enclosed, bezel_details = self._analyze_screen_bezel(image_bgr, face_bbox)
            if bezel_det:
                instant_detected = True
                instant_conf = bezel_conf
                detection_source = "BEZEL_CONTOUR"
                best_bbox = b_bbox
                face_inside = enclosed
                details = bezel_details
                aspect = float(b_bbox[3]) / max(1, b_bbox[2]) if b_bbox else 1.8

        # ─── 3. Moiré Subpixel Grid 2D Spectral Analysis ───
        if not instant_detected and self.enable_moire_analysis and face_bbox:
            moire_det, moire_conf, moire_details = self._analyze_moire_frequency(image_bgr, face_bbox)
            if moire_det:
                instant_detected = True
                instant_conf = moire_conf
                detection_source = "MOIRE_SPECTRAL"
                details = moire_details
                face_inside = True

        # Update per-participant temporal smoothing buffers
        hist = self.p_history[participant_id]
        confs = self.p_confs[participant_id]
        sources = self.p_sources[participant_id]
        bboxes = self.p_bboxes[participant_id]

        hist.append(instant_detected)
        confs.append(instant_conf)
        sources.append(detection_source)
        if best_bbox:
            bboxes.append(best_bbox)

        # Hysteresis & multi-frame confirmation:
        # Require >= min_frames_present to flip ON, and 0 in recent 3 frames to flip OFF
        history_hits = sum(hist)
        was_confirmed = self.p_confirmed_state[participant_id]

        if not was_confirmed:
            if instant_detected and (history_hits >= self.min_frames_present or instant_conf >= 0.80):
                self.p_confirmed_state[participant_id] = True
        else:
            # Active attack state — persist until multiple consecutive clean frames
            if history_hits == 0:
                self.p_confirmed_state[participant_id] = False

        is_confirmed = self.p_confirmed_state[participant_id]

        if is_confirmed:
            final_conf = max(instant_conf, max(confs, default=0.85))
            final_source = detection_source if detection_source != "CLEAR" else next((s for s in reversed(sources) if s != "CLEAR"), "HYBRID")
            final_bbox = best_bbox or (bboxes[-1] if bboxes else None)

            return PhoneDetectionResult(
                phone_detected=True,
                confidence=float(np.clip(max(final_conf, 0.88), 0.0, 1.0)),
                detection_source=final_source,
                phone_bbox=final_bbox,
                aspect_ratio=aspect,
                face_enclosed=face_inside,
                details=details if details != "Nominal live feed — no physical phone or display screen detected." else "Physical smartphone / display screen attack confirmed.",
            )

        return PhoneDetectionResult(
            phone_detected=False,
            confidence=0.0,
            detection_source="CLEAR",
            details="Nominal live feed — no physical phone or display screen detected."
        )

    def _run_yolo_detection(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Runs YOLOv8 inference for COCO presentation display classes (cell phone, tv, laptop)."""
        if self._yolo_model is None:
            self._init_yolo_model()
            if self._yolo_model is None:
                return []

        try:
            # Use native 480/640 resolution with low latency CPU optimization
            results = self._yolo_model.predict(
                frame_bgr,
                classes=COCO_DISPLAY_CLASS_IDS,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=480,
                verbose=False,
            )

            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    name = "cell phone" if cls_id == COCO_CELL_PHONE_CLASS_ID else ("screen display" if cls_id == COCO_TV_MONITOR_CLASS_ID else "laptop screen")
                    detections.append({
                        "bbox": (int(x1), int(y1), w, h),
                        "conf": conf,
                        "center": (cx, cy),
                        "cls_id": cls_id,
                        "name": name,
                    })
            return detections
        except Exception as e:
            logger.debug(f"YOLOv8 detection error: {e}")
            return []

    def _analyze_moire_frequency(
        self,
        image_bgr: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[bool, float, str]:
        """
        Analyzes high-frequency periodic moiré patterns caused by digital screen subpixel grids.
        """
        if face_bbox is None or image_bgr is None or image_bgr.size == 0:
            return False, 0.0, ""

        fx, fy, fw, fh = face_bbox
        h, w = image_bgr.shape[:2]

        x1 = max(0, fx)
        y1 = max(0, fy)
        x2 = min(w, fx + fw)
        y2 = min(h, fy + fh)
        if x2 - x1 < 32 or y2 - y1 < 32:
            return False, 0.0, ""

        roi = image_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # High-pass filter via Laplacian to isolate high-frequency textures
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap_energy = float(np.var(lap))
        if lap_energy < 50.0:  # Smooth natural skin has low Laplacian variance
            return False, 0.0, ""

        # 2D FFT
        F = np.fft.fft2(lap)
        Fshift = np.fft.fftshift(F)
        mag = np.abs(Fshift)

        # Mask DC component (center)
        cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
        r = min(cy, cx) // 4
        if r > 0:
            cv2.circle(mag, (cx, cy), r, 0, -1)

        # Periodic screen grids cause distinct, sharp high-frequency harmonic peaks
        peak_val = float(np.max(mag))
        mean_val = float(np.mean(mag)) + 1e-6
        papr = peak_val / mean_val

        if papr > 40.0 and peak_val > 500.0:
            conf = float(np.clip(0.55 + (papr / 150.0) * 0.40, 0.55, 0.95))
            return True, conf, f"High-frequency screen moiré grid interference detected (PAPR: {papr:.1f})."

        return False, 0.0, ""

    def _analyze_screen_bezel(
        self,
        image_bgr: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]], bool, str]:
        """
        Robust geometric detector for physical smartphone / display screen bezels enclosing a face.
        """
        if face_bbox is None or image_bgr is None or image_bgr.size == 0:
            return False, 0.0, None, False, ""

        fx, fy, fw, fh = face_bbox
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray_clamped = np.minimum(gray, 235)

        face_area = fw * fh
        frame_area = h * w
        min_phone_area = max(frame_area * 0.03, face_area * 1.05)
        max_phone_area = frame_area * 0.96

        best_match = None
        best_conf = 0.0
        best_aspect = 0.0

        for (th1, th2) in [(30, 100), (50, 150), (20, 70), (15, 45)]:
            blurred = cv2.GaussianBlur(gray_clamped, (5, 5), 0)
            edges = cv2.Canny(blurred, th1, th2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_phone_area or area > max_phone_area:
                    continue

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)

                if 4 <= len(approx) <= 12:
                    bx, by, bw, bh = cv2.boundingRect(approx)
                    if bw <= 0 or bh <= 0:
                        continue

                    ox1 = max(bx, fx)
                    oy1 = max(by, fy)
                    ox2 = min(bx + bw, fx + fw)
                    oy2 = min(by + bh, fy + fh)
                    if ox2 <= ox1 or oy2 <= oy1:
                        continue
                    overlap_area = (ox2 - ox1) * (oy2 - oy1)
                    if overlap_area / float(face_area) < 0.60:
                        continue

                    aspect = float(bh) / float(bw) if bw > 0 else 0.0
                    inv_aspect = float(bw) / float(bh) if bh > 0 else 0.0
                    max_aspect = max(aspect, inv_aspect)

                    if 1.20 <= max_aspect <= 2.85:
                        rect_area = bw * bh
                        extent = float(area) / float(rect_area) if rect_area > 0 else 0.0
                        if extent >= 0.52:
                            conf = min(0.96, 0.65 + (extent * 0.28))
                            if conf > best_conf:
                                best_conf = conf
                                best_match = (bx, by, bw, bh)
                                best_aspect = max_aspect

        if best_match is not None and best_conf >= 0.55:
            return True, float(best_conf), best_match, True, f"Physical device rectangular bezel enclosing face detected (aspect: {best_aspect:.2f})."

        return False, 0.0, None, False, ""
