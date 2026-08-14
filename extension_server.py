"""
Native High-Throughput WebSocket Deepfake Inference Server (Enterprise Production).
Serves the Chrome Extension by running the complete deepfake detection pipeline:
- Quality Gate & Cascade Router
- MediaPipe Face Mesh 468-point Alignment & Eye-leveling
- Spatial ViT / EfficientNet Neural Branch (ONNX / PyTorch)
- Frequency 2D Spectral CNN & FFT Analysis
- Temporal Bi-GRU & Micro-Jitter Tracking
- Eye Aspect Ratio (EAR) Blink Rate Anti-Spoofing
- AASIST Audio Anti-Spoofing & Synthetic Voice Detection
- Calibrated Multi-Branch Decision Fusion Engine
- Cryptographic SHA-256 Forensic Audit Hashing

Listens on ws://localhost:8765
"""
import asyncio
import base64
import hashlib
import json
import logging
import os
import sys
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import websockets

from pipeline.config import PipelineConfig
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.types import RawFrame, FrameDecision

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("extension_server")


class DeepfakeExtensionServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.config = PipelineConfig()
        # Sensitive cascade threshold for extension inspection
        self.config.cascade.suspicion_threshold = 0.05

        
        # Initialize full pipeline orchestrator
        logger.info("Initializing Pipeline Orchestrator with all trained models & weights...")
        self.orchestrator = PipelineOrchestrator(self.config)
        
        # Track active client connections and participant stats
        self.clients = set()
        self.frame_counters: Dict[str, int] = {}
        self.participant_last_seen: Dict[str, float] = {}
        self.last_verdicts: Dict[str, Dict] = {}


    def decode_frame(self, data_str: str) -> Optional[np.ndarray]:
        """Decodes base64 JPEG/PNG string into OpenCV BGR numpy array."""
        try:
            if "," in data_str:
                data_str = data_str.split(",", 1)[1]
            img_bytes = base64.b64decode(data_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img_bgr
        except Exception as e:
            logger.debug(f"Failed to decode image frame: {e}")
            return None

    def decode_audio(self, audio_b64: Optional[str]) -> Optional[np.ndarray]:
        """Decodes base64 float32 PCM byte array into 16kHz mono NumPy array."""
        if not audio_b64:
            return None
        try:
            if "," in audio_b64:
                audio_b64 = audio_b64.split(",", 1)[1]
            raw_bytes = base64.b64decode(audio_b64)
            audio_arr = np.frombuffer(raw_bytes, dtype=np.float32)
            if audio_arr.size > 0:
                return audio_arr
        except Exception as e:
            logger.debug(f"Failed to decode audio window: {e}")
        return None

    def process_participant_frame(
        self,
        participant_id: str,
        frame_bgr: np.ndarray,
        audio_pcm: Optional[np.ndarray],
        timestamp: float
    ) -> Optional[Dict]:
        """Runs the complete multi-stage pipeline on a participant frame + audio window."""
        t_start = time.perf_counter()
        
        frame_idx = self.frame_counters.get(participant_id, 0) + 1
        self.frame_counters[participant_id] = frame_idx
        self.participant_last_seen[participant_id] = timestamp

        raw_frame = RawFrame(
            participant_id=participant_id,
            frame_idx=frame_idx,
            timestamp=timestamp,
            image_bgr=frame_bgr,
            audio_window=audio_pcm,
        )

        # Stage 1: Quality Gate
        quality = self.orchestrator._quality_gate.run(raw_frame)
        if not quality.passed:
            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "type": "telemetry",
                "participant_id": participant_id,
                "frame_idx": frame_idx,
                "status": "AWAITING_FACE",
                "score": 0.0,
                "raw_score": 0.0,
                "review_flag": False,
                "block_flag": False,
                "av_mismatch": False,
                "p_spatial": 0.0,
                "p_freq": 0.0,
                "p_temporal": 0.0,
                "p_liveness": 0.0,
                "jitter": 0.0,
                "p_voice_clone": 0.0,
                "latency_ms": round(t_elapsed_ms, 2),
                "timestamp": timestamp,
                "reject_reason": quality.reject_reason,
            }

        # Stage 2: Cascade Router (quarter-resolution neural triage)
        cascade = self.orchestrator._cascade_router.route(raw_frame, quality)

        # Stage 3: Face Alignment (468 landmarks & eye-leveling)
        try:
            aligned = self.orchestrator._face_aligner.align(raw_frame, quality)
        except Exception as e:
            logger.debug(f"Face alignment exception on frame {frame_idx}: {e}")
            aligned = None

        if aligned is None:
            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "type": "telemetry",
                "participant_id": participant_id,
                "frame_idx": frame_idx,
                "status": "ALIGN_FAILED",
                "score": 0.0,
                "raw_score": 0.0,
                "review_flag": False,
                "block_flag": False,
                "av_mismatch": False,
                "p_spatial": 0.0,
                "p_freq": 0.0,
                "p_temporal": 0.0,
                "p_liveness": 0.0,
                "jitter": 0.0,
                "p_voice_clone": 0.0,
                "latency_ms": round(t_elapsed_ms, 2),
                "timestamp": timestamp,
            }

        # Stage 4: Multi-Modal Neural Branches (Spatial ONNX ViT + Frequency CNN + AASIST)
        branch_scores = self.orchestrator._branch_runner.run(aligned)

        # Stage 5: Temporal Verification & EAR Blink Anti-Spoofing
        tracker = self.orchestrator._get_temporal_tracker(participant_id)
        temporal_result = tracker.update(aligned, branch_scores)

        # Stage 6: Calibrated Multi-Branch Decision Fusion Engine
        decision = self.orchestrator._decision_engine.decide(
            participant_id=participant_id,
            frame_idx=frame_idx,
            timestamp=timestamp,
            branch_scores=branch_scores,
            temporal_result=temporal_result,
            pose_confidence=quality.pose_confidence,
        )

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # ── Explainable AI: generate human-readable recommendation ──
        score = float(decision.smoothed_score)
        p_sp = float(branch_scores.p_spatial)
        p_fr = float(branch_scores.p_freq)
        p_te = float(temporal_result.p_temporal)
        p_li = float(temporal_result.p_liveness)
        jitter_val = float(temporal_result.jitter_score)
        p_sy = float(branch_scores.p_sync) if branch_scores.p_sync is not None else 0.0
        p_vc = float(branch_scores.p_voice_clone) if branch_scores.p_voice_clone is not None else 0.0

        # Calibration Warmup Phase (first 8 frames / ~2.5s) to prevent false alerts
        if frame_idx <= 8:
            threat_level = "CALIBRATING"
            threat_label = "Calibrating Neural Baseline (2–4s)..."
            confidence_tier = "Calibrating"
            caution_note = "Initial baseline calibration in progress. Confidence increases as temporal sequence accumulates."
            recommendations = ["Gathering multi-frame temporal embeddings and calibrating eye-blink kinematics. Please wait a moment for stabilized analysis."]
        else:
            caution_note = "Results reflect multi-branch neural probabilities. Analysis deepens over time."
            # Threat level (5 tiers, stabilized)
            if score >= 0.82:
                threat_level = "CRITICAL"
                threat_label = "High-Confidence Synthetic Media"
                confidence_tier = "Very High"
            elif score >= 0.65:
                threat_level = "HIGH"
                threat_label = "Probable Manipulated Feed"
                confidence_tier = "High"
            elif score >= 0.45:
                threat_level = "MODERATE"
                threat_label = "Anomalies Detected — Manual Review Advised"
                confidence_tier = "Moderate"
            elif score >= 0.25:
                threat_level = "LOW"
                threat_label = "Minor Irregularities — Likely Authentic"
                confidence_tier = "Low"
            else:
                threat_level = "CLEAR"
                threat_label = "Verified Authentic Feed"
                confidence_tier = "High"

            # Dominant signal attribution (XAI)
            signals = {
                "Spatial Artifacts (ViT Neural Analysis)": p_sp,
                "Spectral Pattern (Frequency CNN)": p_fr,
                "Temporal Inconsistency (Bi-GRU)": p_te,
                "Liveness Failure (EAR Blink Rate)": p_li,
                "Landmark Micro-Jitter": jitter_val,
            }
            if p_vc > 0.4:
                signals["Voice Cloning / Synthetic Audio (AASIST)"] = p_vc

            dominant_signal = max(signals, key=signals.get)
            dominant_value = signals[dominant_signal]

            # Professional recommendation
            recommendations = []
            if p_sp > 0.6:
                recommendations.append("Visual artifacts detected around face boundaries — consistent with GAN/diffusion generation or face-swap blending.")
            if p_fr > 0.5:
                recommendations.append("Spectral analysis shows atypical high-frequency energy distribution — may indicate AI-upscaled or synthetically generated content.")
            if p_te > 0.5:
                recommendations.append("Temporal embedding drift detected — frame-to-frame facial features lack natural consistency.")
            if p_li > 0.6:
                recommendations.append("Insufficient physiological activity (blink rate / micro-motion) — possible static image replay or presentation attack.")
            if jitter_val > 0.35:
                recommendations.append("Elevated facial landmark instability — face mask boundaries may be jittering between frames.")
            if p_sy > 0.5:
                recommendations.append("Audio-visual synchronisation mismatch detected — lip movement does not correlate with speech patterns.")
            if p_vc > 0.5:
                recommendations.append("AASIST acoustic analysis flagged synthetic speech patterns / voice cloning artifacts.")

            if not recommendations:
                if score < 0.25:
                    recommendations.append("All neural analysis branches report nominal readings. No indicators of synthetic manipulation detected in this participant's video stream.")
                else:
                    recommendations.append("Mild statistical deviations observed but within normal operating range. Continue monitoring — no action required at this time.")

        # Cryptographic forensic hash
        audit_raw = f"{participant_id}:{frame_idx}:{round(score,4)}:{timestamp}"
        audit_hash = hashlib.sha256(audit_raw.encode("utf-8")).hexdigest()[:16]

        return {
            "type": "verdict",
            "participant_id": participant_id,
            "frame_idx": frame_idx,
            # Granular threat assessment
            "threat_level": threat_level,
            "threat_label": threat_label,
            "confidence_tier": confidence_tier,
            "caution_note": caution_note,
            "score": round(score, 4),
            "raw_score": round(float(decision.p_frame), 4),
            "review_flag": bool(decision.review_flag),
            "block_flag": bool(decision.block_flag),
            "av_mismatch": bool(decision.av_mismatch_flag) if decision.av_mismatch_flag is not None else False,
            # Branch telemetry
            "p_spatial": round(p_sp, 4),
            "p_freq": round(p_fr, 4),
            "p_sync": round(p_sy, 4),
            "p_temporal": round(p_te, 4),
            "p_liveness": round(p_li, 4),
            "p_voice_clone": round(p_vc, 4),
            "jitter": round(jitter_val, 4),
            # Explainable AI fields
            "dominant_signal": dominant_signal if frame_idx > 8 else "Baseline Calibrating",
            "dominant_value": round(dominant_value, 4) if frame_idx > 8 else 0.0,
            "recommendation": " ".join(recommendations),
            "audit_hash": audit_hash,
            "latency_ms": round(t_elapsed_ms, 2),
            "timestamp": timestamp,
        }


    def get_models_status(self) -> Dict[str, bool]:
        """Inspects disk to determine exact readiness of all neural models."""
        return {
            "spatial_onnx": os.path.exists("models/deepfake_detector.onnx") or os.path.exists("model.onnx"),
            "temporal_gru": os.path.exists("models/temporal_gru.pt"),
            "face_mesh": os.path.exists("models/face_landmarker.task"),
            "blazeface": os.path.exists("models/blaze_face_short_range.tflite"),
            "freq_classifier": os.path.exists("models/freq_classifier.pt"),
            "fusion_head": os.path.exists("models/fusion_head.pkl"),
            "aasist_audio": os.path.exists("assist/weights/AASIST.pth"),
            "syncnet_lip_sync": os.path.exists("models/sync_net.pt"),
        }

    async def handle_client(self, websocket):
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"Chrome Extension client connected: {client_addr}")

        # Send initial handshake with model verification
        await websocket.send(json.dumps({
            "type": "handshake_ack",
            "server_version": "2.1.0",
            "status": "ready",
            "models_loaded": self.get_models_status(),
        }))

        # Send active cached verdicts to newly connected client (popup or new tab)
        for cached in self.last_verdicts.values():
            try:
                await websocket.send(json.dumps(cached))
            except Exception:
                pass

        try:
            async for message in websocket:
                if isinstance(message, str):
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        continue

                    msg_type = payload.get("type", "frame")

                    if msg_type == "ping":
                        await websocket.send(json.dumps({"type": "pong", "time": time.time()}))
                        continue

                    if msg_type == "frame":
                        participant_id = payload.get("participant_id", "local_user")
                        image_data = payload.get("image")
                        audio_data = payload.get("audio")
                        timestamp = payload.get("timestamp", time.time())

                        if not image_data:
                            continue

                        frame_bgr = self.decode_frame(image_data)
                        audio_pcm = self.decode_audio(audio_data) if audio_data else None

                        if frame_bgr is not None:
                            # Run inference in worker thread to prevent event loop blocking
                            result = await asyncio.to_thread(
                                self.process_participant_frame,
                                participant_id,
                                frame_bgr,
                                audio_pcm,
                                timestamp
                            )
                            if result:
                                self.last_verdicts[participant_id] = result
                                msg_json = json.dumps(result)
                                # Broadcast verdict to all connected extension clients (content scripts AND popup)
                                for client in list(self.clients):
                                    try:
                                        await client.send(msg_json)
                                    except Exception:
                                        pass

                    elif msg_type == "reset_participant":
                        pid = payload.get("participant_id")
                        if pid:
                            self.orchestrator.reset_participant(pid)
                            if pid in self.frame_counters:
                                del self.frame_counters[pid]
                            if pid in self.last_verdicts:
                                del self.last_verdicts[pid]
                            logger.info(f"Reset participant state: {pid}")


        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Chrome Extension client disconnected: {client_addr}")
        finally:
            if websocket in self.clients:
                self.clients.remove(websocket)

    async def start(self):
        logger.info(f"🚀 Starting Netram AI Deepfake Server on ws://{self.host}:{self.port}")
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=15 * 1024 * 1024,
            origins=None,              # Allow all origins (extension runs from meet.google.com, zoom.us, etc.)
            ping_interval=20,          # Keep connection alive through Render's proxy
            ping_timeout=20,
        ):
            await asyncio.Future()  # run forever



def run_server(host: str = "localhost", port: int = 8765):
    server = DeepfakeExtensionServer(host=host, port=port)
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")


if __name__ == "__main__":
    import argparse
    default_port = int(os.environ.get("PORT", 8765))
    parser = argparse.ArgumentParser(description="Deepfake Detection Extension WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host binding (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port binding (default: {default_port})")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)

