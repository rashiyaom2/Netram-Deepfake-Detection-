import asyncio
import base64
import json
import cv2
import numpy as np
import pytest
import websockets

from extension_server import DeepfakeExtensionServer


def _generate_synthetic_face_jpeg(is_fake: bool = False) -> str:
    """Creates a synthetic face frame and encodes it as base64 JPEG."""
    img = np.full((240, 320, 3), 40, dtype=np.uint8)
    center = (160, 120)
    cv2.ellipse(img, center, (50, 70), 0, 0, 360, (140, 180, 220), -1)
    cv2.circle(img, (140, 100), 8, (255, 255, 255), -1)
    cv2.circle(img, (140, 100), 4, (20, 20, 20), -1)
    cv2.circle(img, (180, 100), 8, (255, 255, 255), -1)
    cv2.circle(img, (180, 100), 4, (20, 20, 20), -1)
    cv2.ellipse(img, (160, 150), (16, 8), 0, 0, 360, (50, 50, 180), -1)
    if is_fake:
        noise = np.random.randint(0, 100, (40, 40, 3), dtype=np.uint8)
        img[90:130, 140:180] = cv2.add(img[90:130, 140:180], noise)
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def _generate_synthetic_audio_pcm() -> str:
    """Generates 16kHz float32 audio and encodes as base64."""
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
    return base64.b64encode(samples.tobytes()).decode("utf-8")


class TestExtensionServer:
    def test_server_frame_decoding(self):
        server = DeepfakeExtensionServer()
        b64 = _generate_synthetic_face_jpeg(is_fake=False)
        frame = server.decode_frame(b64)
        assert frame is not None
        assert frame.shape == (240, 320, 3)

    def test_server_audio_decoding(self):
        server = DeepfakeExtensionServer()
        audio_b64 = _generate_synthetic_audio_pcm()
        audio_pcm = server.decode_audio(audio_b64)
        assert audio_pcm is not None
        assert audio_pcm.dtype == np.float32
        assert len(audio_pcm) == 16000

    def test_multi_participant_processing_with_audio(self):
        server = DeepfakeExtensionServer()
        audio_b64 = _generate_synthetic_audio_pcm()
        audio_pcm = server.decode_audio(audio_b64)

        for i in range(1, 5):
            pid = f"participant_{i}"
            b64 = _generate_synthetic_face_jpeg(is_fake=(i == 2))
            frame_bgr = server.decode_frame(b64)
            result = server.process_participant_frame(
                participant_id=pid,
                frame_bgr=frame_bgr,
                audio_pcm=audio_pcm,
                timestamp=float(i)
            )
            assert result is not None
            assert result["participant_id"] == pid
            assert "score" in result
            assert "p_spatial" in result
            assert "p_freq" in result
            assert "p_temporal" in result
            assert "p_liveness" in result
            assert "latency_ms" in result
            assert result["latency_ms"] >= 0.0
            # Explainable AI & Audit fields
            if result["type"] == "verdict":
                assert "threat_level" in result
                assert result["threat_level"] in ("CLEAR", "LOW", "MODERATE", "HIGH", "CRITICAL", "CALIBRATING")
                assert "threat_label" in result
                assert "recommendation" in result
                assert len(result["recommendation"]) > 0
                assert "dominant_signal" in result
                assert "audit_hash" in result
                assert len(result["audit_hash"]) > 0


    @pytest.mark.asyncio
    async def test_websocket_e2e_communication(self):
        server = DeepfakeExtensionServer(host="127.0.0.1", port=8768)
        async with websockets.serve(server.handle_client, server.host, server.port):
            async with websockets.connect(f"ws://{server.host}:{server.port}") as ws:
                handshake = json.loads(await ws.recv())
                assert handshake["type"] == "handshake_ack"
                assert handshake["status"] == "ready"
                assert "models_loaded" in handshake
                assert handshake["models_loaded"]["spatial_onnx"] is True

                await ws.send(json.dumps({"type": "ping"}))
                pong = json.loads(await ws.recv())
                assert pong["type"] == "pong"

                frame_b64 = _generate_synthetic_face_jpeg(is_fake=False)
                audio_b64 = _generate_synthetic_audio_pcm()
                await ws.send(json.dumps({
                    "type": "frame",
                    "participant_id": "google_meet_user_1",
                    "image": frame_b64,
                    "audio": audio_b64,
                    "timestamp": 1.0,
                }))
                verdict = json.loads(await ws.recv())
                assert verdict["type"] in ("verdict", "telemetry")
                assert verdict["participant_id"] == "google_meet_user_1"
                assert "score" in verdict
                assert "p_spatial" in verdict
