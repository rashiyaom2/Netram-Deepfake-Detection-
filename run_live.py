import asyncio
import logging
import argparse
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

# Optional pyvirtualcam for Google Meet/Zoom integration
try:
    import pyvirtualcam
    HAS_VIRTUAL_CAM = True
except ImportError:
    HAS_VIRTUAL_CAM = False

from pipeline.orchestrator import PipelineOrchestrator
from pipeline.config import PipelineConfig
from pipeline.types import FrameDecision, RawFrame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("live_runner")


class ThreadedCamera:
    """
    Dedicated high-speed camera reader running on a separate OS thread.
    Continuously polls frames from hardware at 30-60 FPS without ever blocking
    the GUI, asyncio event loop, or neural network inference.
    """
    def __init__(self, src: int = 0):
        # On Windows, try DSHOW backend for fastest initialization
        if sys.platform == "win32":
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(src)
        else:
            self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video camera device {src}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30

        self.frame: Optional[np.ndarray] = None
        self.timestamp: float = time.monotonic()
        self.running = True
        self.lock = threading.Lock()

        # Initial read
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.timestamp = time.monotonic()
            else:
                time.sleep(0.005)

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray], float]:
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy(), self.timestamp
            return False, None, self.timestamp

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


def draw_hud(frame: np.ndarray, decision: Optional[FrameDecision], fps: float, has_audio: bool) -> np.ndarray:
    """Renders a modern, responsive HUD bar with deepfake score and system status."""
    h, w = frame.shape[:2]
    out = frame.copy()

    # Default state: Awaiting or Real
    score = decision.smoothed_score if decision else 0.0
    is_blocked = decision.block_flag if decision else False
    is_review = decision.review_flag if decision else False
    has_mismatch = (decision.av_mismatch_flag if decision else False) and has_audio

    if is_blocked or score >= 0.80:
        banner_color = (30, 30, 220)       # Red
        status_text = "DEEPFAKE DETECTED"
    elif is_review or score >= 0.50:
        banner_color = (0, 165, 255)      # Orange/Amber
        status_text = "ANOMALY DETECTED"
    elif score >= 0.25:
        banner_color = (220, 160, 40)     # Cyan/Blue
        status_text = "MINOR VARIANCE"
    else:
        banner_color = (40, 190, 60)      # Emerald Green
        status_text = "AUTHENTIC FEED"

    # Semi-transparent top banner
    banner_h = 64
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (16, 18, 24), -1)
    cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)

    # Status indicator pill
    cv2.rectangle(out, (12, 12), (210, 52), banner_color, -1, cv2.LINE_AA)
    cv2.putText(out, status_text, (20, 38), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Deepfake probability bar
    bar_x, bar_y, bar_w, bar_h = 230, 20, max(120, w - 420), 24
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 48, 56), -1, cv2.LINE_AA)
    fill_w = int(np.clip(score, 0.0, 1.0) * bar_w)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), banner_color, -1, cv2.LINE_AA)
    cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 125, 140), 1, cv2.LINE_AA)

    # Score text
    score_label = f"Risk: {score:.1%}"
    cv2.putText(out, score_label, (bar_x + bar_w + 14, 38), cv2.FONT_HERSHEY_DUPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    # FPS / Live indicator on top right
    mismatch_str = " [A/V MISMATCH]" if has_mismatch else ""
    info_label = f"{fps:.0f} FPS{mismatch_str}"
    cv2.putText(out, info_label, (w - 150, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Border glow
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), banner_color, 3)

    return out



def run_live_gui(video_device: int = 0, audio_device: Optional[int] = None, use_virtual_cam: bool = False, debug: bool = False):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Initializing Live Deepfake Detection Engine...")
    config = PipelineConfig()
    # Ensure cascade does not drop live evaluation frames prematurely
    config.cascade.suspicion_threshold = 0.05
    orchestrator = PipelineOrchestrator(config)

    logger.info(f"Opening camera {video_device} in high-speed capture mode...")
    cam_reader = ThreadedCamera(src=video_device)

    virtual_cam = None
    if use_virtual_cam and HAS_VIRTUAL_CAM:
        try:
            virtual_cam = pyvirtualcam.Camera(width=cam_reader.width, height=cam_reader.height, fps=cam_reader.fps)
            logger.info(f"Virtual camera started: {virtual_cam.device} (Select in Zoom/Google Meet)")
        except Exception as e:
            logger.warning(f"Could not open virtual camera: {e}")

    latest_decision: Optional[FrameDecision] = None
    inference_busy = False
    stop_event = threading.Event()
    has_audio = audio_device is not None

    def _inference_worker():
        nonlocal latest_decision, inference_busy
        last_inferred_time = 0.0
        min_interval = 1.0 / max(1.0, config.ingestion.video_fps_sample)  # ~3 FPS sample rate

        frame_idx = 0
        while not stop_event.is_set():
            now = time.monotonic()
            if (now - last_inferred_time) >= min_interval and not inference_busy:
                ok, frame_bgr, ts = cam_reader.read_latest()
                if ok and frame_bgr is not None:
                    inference_busy = True
                    frame_idx += 1
                    raw_frame = RawFrame(
                        participant_id="local_user",
                        frame_idx=frame_idx,
                        timestamp=ts,
                        image_bgr=frame_bgr,
                        audio_window=None,
                    )
                    try:
                        decision = orchestrator._sync_process_frame(raw_frame)
                        if decision is not None:
                            latest_decision = decision
                    except Exception as e:
                        logger.debug(f"Inference error: {e}")
                    finally:
                        last_inferred_time = time.monotonic()
                        inference_busy = False
            time.sleep(0.01)

    # Start background inference worker thread
    inf_thread = threading.Thread(target=_inference_worker, daemon=True)
    inf_thread.start()

    logger.info("Live Deepfake Detector running. Press 'q' in the window to quit.")
    cv2.namedWindow("Deepfake Detection (Live)", cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    fps_smooth = 30.0

    try:
        while not stop_event.is_set():
            ok, frame, _ = cam_reader.read_latest()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)

            # Draw HUD
            display_frame = draw_hud(frame, latest_decision, fps_smooth, has_audio)

            # Render to screen
            cv2.imshow("Deepfake Detection (Live)", display_frame)

            # Output to Virtual Camera
            if virtual_cam:
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                virtual_cam.send(rgb_frame)
                virtual_cam.sleep_until_next_frame()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break

    finally:
        stop_event.set()
        inf_thread.join(timeout=1.0)
        cam_reader.stop()
        cv2.destroyAllWindows()
        if virtual_cam:
            virtual_cam.close()
        logger.info("Live pipeline stopped cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Deepfake Detection Pipeline Runner")
    parser.add_argument("--video-device", type=int, default=0, help="Video device index (default: 0)")
    parser.add_argument("--audio-device", type=int, default=None, help="Audio device index (default: None)")
    parser.add_argument("--virtual-cam", action="store_true", help="Output to OBS Virtual Camera for Google Meet/Zoom")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()
    run_live_gui(args.video_device, args.audio_device, args.virtual_cam, args.debug)
