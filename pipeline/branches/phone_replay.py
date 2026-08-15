"""
Stage 4d — Real-Time Phone & Screen Replay Attack Detector using YOLOv8.

Identifies Presentation Attacks / Display Replays where an attacker holds up a
physical smartphone, tablet, or display screen in front of the camera
to stream pre-recorded or deepfaked videos.

Uses deep learning object detection (YOLOv8, COCO class 67 'cell phone')
based on shape and structural visual features, rather than brittle intensity/glare
thresholds. Temporal smoothing filters out transient single-frame glitches.
"""
from dataclasses import dataclass
import logging
import os
import time
from collections import deque
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)

COCO_CELL_PHONE_CLASS_ID = 67  # 'cell phone' class in standard COCO dataset


@dataclass
class PhoneDetectionResult:
    phone_detected: bool
    confidence: float                     # 0.0 to 1.0
    detection_source: str                 # "YOLOV8_NEURAL", "BEZEL_CONTOUR", "HYBRID", "CLEAR"
    phone_bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    aspect_ratio: float = 0.0
    face_enclosed: bool = False
    details: str = "Nominal live feed — no physical phone or display screen detected."


class PhoneReplayDetector:
    """
    Real-time phone and screen replay attack detector powered by YOLOv8.
    Distinguishes physical smartphones from glare or ambient lighting
    based on visual geometry and neural features.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        conf_threshold: float = 0.38,
        iou_threshold: float = 0.45,
        smoothing_window: int = 5,
        min_frames_present: int = 2,
        enable_bezel_analysis: bool = True,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.smoothing_window = smoothing_window
        self.min_frames_present = min_frames_present
        self.enable_bezel_analysis = enable_bezel_analysis

        self.history: deque = deque(maxlen=smoothing_window)
        self.last_confidence_history: deque = deque(maxlen=smoothing_window)
        self._yolo_model = None
        self._yolo_initialized = False

        self._init_yolo_model()

    def _init_yolo_model(self):
        """Initializes YOLOv8 neural phone detector."""
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
            from ultralytics import YOLO
            self._yolo_model = YOLO(chosen_path)
            logger.info(f"YOLOv8 Phone Detector initialized successfully with {chosen_path}")
        except Exception as e:
            logger.warning(f"Could not load YOLOv8 model from {chosen_path}: {e}")
            self._yolo_model = None

    def detect(
        self,
        image_bgr: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> PhoneDetectionResult:
        """
        Executes YOLOv8 phone detection + temporal smoothing on a single frame.
        """
        if image_bgr is None or image_bgr.size == 0:
            return PhoneDetectionResult(phone_detected=False, confidence=0.0, detection_source="CLEAR")

        yolo_detections = self._run_yolo_detection(image_bgr)
        instant_detected = len(yolo_detections) > 0
        instant_conf = max([d["conf"] for d in yolo_detections], default=0.0)

        # Update temporal smoothing buffer
        self.history.append(instant_detected)
        self.last_confidence_history.append(instant_conf)

        # ─── 1. Neural YOLOv8 Phone Presence ───
        if instant_detected or sum(self.history) >= self.min_frames_present:
            best_det = max(yolo_detections, key=lambda d: d["conf"]) if yolo_detections else None
            best_bbox = best_det["bbox"] if best_det else None
            conf = max(instant_conf, max(self.last_confidence_history, default=0.85))

            face_inside = False
            if best_bbox and face_bbox:
                bx, by, bw, bh = best_bbox
                fx, fy, fw, fh = face_bbox
                overlap_x1 = max(bx, fx)
                overlap_y1 = max(by, fy)
                overlap_x2 = min(bx + bw, fx + fw)
                overlap_y2 = min(by + bh, fy + fh)
                if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                    face_inside = True

            aspect = float(best_bbox[3]) / max(1, best_bbox[2]) if best_bbox else 1.8
            return PhoneDetectionResult(
                phone_detected=True,
                confidence=float(np.clip(max(conf, 0.88), 0.0, 1.0)),
                detection_source="YOLOV8_NEURAL",
                phone_bbox=best_bbox,
                aspect_ratio=aspect,
                face_enclosed=face_inside,
                details=f"Physical smartphone display screen identified via YOLOv8 (conf: {conf*100:.1f}%).",
            )

        # ─── 2. Fallback Geometric Bezel Analysis (Offline / Synthetic) ───
        if self.enable_bezel_analysis and face_bbox:
            bezel_detected, bezel_conf, b_bbox, enclosed = self._analyze_screen_bezel(image_bgr, face_bbox)
            if bezel_detected:
                return PhoneDetectionResult(
                    phone_detected=True,
                    confidence=float(bezel_conf),
                    detection_source="BEZEL_CONTOUR",
                    phone_bbox=b_bbox,
                    face_enclosed=enclosed,
                    details="Physical phone/screen bezel enclosing face detected.",
                )

        return PhoneDetectionResult(
            phone_detected=False,
            confidence=0.0,
            detection_source="CLEAR",
            details="Nominal live feed — no physical phone or display screen detected."
        )

    def _run_yolo_detection(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Runs YOLOv8 inference for COCO class 67 ('cell phone')."""
        if self._yolo_model is None:
            self._init_yolo_model()
            if self._yolo_model is None:
                return []

        try:
            # Fast 320px inference for ultra-low latency (< 12ms on CPU)
            results = self._yolo_model.predict(
                frame_bgr,
                classes=[COCO_CELL_PHONE_CLASS_ID],
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=320,
                verbose=False,
            )

            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections.append({
                        "bbox": (int(x1), int(y1), w, h),
                        "conf": conf,
                        "center": (cx, cy),
                    })
            return detections
        except Exception as e:
            logger.debug(f"YOLOv8 detection error: {e}")
            return []

    def _analyze_screen_bezel(
        self,
        image_bgr: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]], bool]:
        """Fallback geometric screen bezel contour analyzer."""
        if face_bbox is None:
            return False, 0.0, None, False

        fx, fy, fw, fh = face_bbox
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray_clamped = np.minimum(gray, 235)

        frame_area = h * w
        face_area = fw * fh
        min_phone_area = max(frame_area * 0.03, face_area * 1.05)
        max_phone_area = frame_area * 0.96

        best_match = None
        best_conf = 0.0

        for (th1, th2) in [(30, 100), (50, 150), (20, 70)]:
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

                if 4 <= len(approx) <= 10:
                    x, y, bw, bh = cv2.boundingRect(approx)
                    if bw <= 0 or bh <= 0:
                        continue

                    overlap_x1 = max(x, fx)
                    overlap_y1 = max(y, fy)
                    overlap_x2 = min(x + bw, fx + fw)
                    overlap_y2 = min(y + bh, fy + fh)
                    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
                        continue
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    if overlap_area / float(face_area) < 0.65:
                        continue

                    aspect = float(bh) / float(bw) if bw > 0 else 0.0
                    inv_aspect = float(bw) / float(bh) if bh > 0 else 0.0
                    max_aspect = max(aspect, inv_aspect)

                    if 1.25 <= max_aspect <= 2.65:
                        rect_area = bw * bh
                        extent = float(area) / float(rect_area) if rect_area > 0 else 0.0
                        if extent >= 0.55:
                            conf = min(0.96, 0.65 + (extent * 0.25))
                            if conf > best_conf:
                                best_conf = conf
                                best_match = (x, y, bw, bh)

        if best_match is not None and best_conf >= 0.60:
            return True, float(best_conf), best_match, True

        return False, 0.0, None, False
