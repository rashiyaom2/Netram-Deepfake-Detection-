"""
Stage 1 — Stream Ingestion & Sampling (doc §1).

Two capture front-ends are provided, both feeding the same core buffering
logic (`ParticipantBuffer` + `IngestionCore`):

  1. AiortcIngestion    - bot-join capture: one aiortc MediaStreamTrack per
                           remote participant (video + audio), the intended
                           production path for Meet/Zoom.
  2. LocalCameraIngestion - OBS virtual camera / FFmpeg loopback via cv2,
                           for local single-participant testing without
                           standing up a real WebRTC session.

Both produce `RawFrame` objects (pipeline/types.py) onto an asyncio.Queue
that Stage 2 (quality_gate.py) consumes from — that queue boundary is the
only thing downstream stages need to know about.
"""
import asyncio
import time
from collections import deque
from typing import Dict, Deque, Optional, Tuple, TYPE_CHECKING

import numpy as np
import cv2

from pipeline.config import IngestionConfig
from pipeline.types import RawFrame

if TYPE_CHECKING:
    # av is an optional runtime dependency (only needed by aiortc front-end);
    # import here for type-checkers to resolve the `av.AudioResampler` hint
    import av  # noqa: F401


# --------------------------------------------------------------------------- #
# Rolling audio ring buffer
# --------------------------------------------------------------------------- #
class RollingAudioBuffer:
    """
    Fixed-length ring buffer of mono float32 audio, sized to
    `config.audio_buffer_seconds` seconds at `config.audio_sample_rate` Hz.
    """

    def __init__(self, sample_rate: int, seconds: float):
        self.sample_rate = sample_rate
        self.capacity = int(sample_rate * seconds)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._write_pos = 0
        self._filled = 0

    def append(self, samples: np.ndarray) -> None:
        """Append mono float32 samples, wrapping around the ring."""
        samples = samples.astype(np.float32, copy=False)
        n = len(samples)
        if n == 0:
            return
        if n >= self.capacity:
            # Only the tail fits; buffer is now fully overwritten.
            self._buf[:] = samples[-self.capacity:]
            self._write_pos = 0
            self._filled = self.capacity
            return

        end = self._write_pos + n
        if end <= self.capacity:
            self._buf[self._write_pos:end] = samples
        else:
            first_part = self.capacity - self._write_pos
            self._buf[self._write_pos:] = samples[:first_part]
            self._buf[: end - self.capacity] = samples[first_part:]
        self._write_pos = end % self.capacity
        self._filled = min(self.capacity, self._filled + n)

    def snapshot(self) -> np.ndarray:
        """Return the current window in chronological order (zero-padded if not yet full)."""
        if self._filled < self.capacity:
            # Not full yet: valid audio is the first `_filled` samples written
            # from position 0 (ring hasn't wrapped), left-padded with zeros.
            valid = self._buf[: self._filled]
            pad = np.zeros(self.capacity - self._filled, dtype=np.float32)
            return np.concatenate([pad, valid])
        # Full ring: unwrap starting at the oldest sample (current write pos).
        return np.concatenate([self._buf[self._write_pos:], self._buf[: self._write_pos]])


# --------------------------------------------------------------------------- #
# Per-participant state
# --------------------------------------------------------------------------- #
class ParticipantBuffer:
    def __init__(self, participant_id: str, cfg: IngestionConfig):
        self.participant_id = participant_id
        self.cfg = cfg
        self.video_buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=cfg.video_buffer_frames)
        self.audio_buffer = RollingAudioBuffer(cfg.audio_sample_rate, cfg.audio_buffer_seconds)
        self._last_sampled_at: float = float("-inf")  # ensures the very first frame is always accepted
        self._frame_idx: int = 0

    def should_sample(self, now: float) -> bool:
        """FPS gate: only accept a video frame every 1/fps seconds (doc §1: 2-5 FPS)."""
        min_interval = 1.0 / self.cfg.video_fps_sample
        return (now - self._last_sampled_at) >= min_interval

    def push_video_frame(self, image_bgr: np.ndarray, timestamp: float) -> RawFrame:
        self._last_sampled_at = timestamp
        self._frame_idx += 1
        self.video_buffer.append((timestamp, image_bgr))
        return RawFrame(
            participant_id=self.participant_id,
            frame_idx=self._frame_idx,
            timestamp=timestamp,
            image_bgr=image_bgr,
            audio_window=self.audio_buffer.snapshot().copy(),
        )

    def push_audio_chunk(self, samples: np.ndarray) -> None:
        self.audio_buffer.append(samples)


# --------------------------------------------------------------------------- #
# Core orchestrator (shared by both front-ends)
# --------------------------------------------------------------------------- #
class IngestionCore:
    def __init__(self, cfg: Optional[IngestionConfig] = None, queue_maxsize: int = 256):
        self.cfg = cfg or IngestionConfig()
        self.participants: Dict[str, ParticipantBuffer] = {}
        self.output_queue: "asyncio.Queue[RawFrame]" = asyncio.Queue(maxsize=queue_maxsize)

    def _get_participant(self, participant_id: str) -> ParticipantBuffer:
        if participant_id not in self.participants:
            self.participants[participant_id] = ParticipantBuffer(participant_id, self.cfg)
        return self.participants[participant_id]

    async def on_video_frame(self, participant_id: str, image_bgr: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Call for every decoded video frame; internally rate-limited to `video_fps_sample`."""
        ts = timestamp if timestamp is not None else time.monotonic()
        pbuf = self._get_participant(participant_id)
        if not pbuf.should_sample(ts):
            return  # drop: keeps us at the 2-5 FPS budget regardless of source frame rate
        raw_frame = pbuf.push_video_frame(image_bgr, ts)
        try:
            self.output_queue.put_nowait(raw_frame)
        except asyncio.QueueFull:
            # Backpressure: drop oldest-style — downstream (Stage 0 cascade) is the
            # bottleneck in practice, so we prioritize freshness over completeness.
            _ = self.output_queue.get_nowait()
            self.output_queue.put_nowait(raw_frame)

    async def on_audio_chunk(self, participant_id: str, samples: np.ndarray) -> None:
        """Call for every decoded audio chunk (any size); resample to `audio_sample_rate` first."""
        pbuf = self._get_participant(participant_id)
        pbuf.push_audio_chunk(samples)

    def remove_participant(self, participant_id: str) -> None:
        self.participants.pop(participant_id, None)


# --------------------------------------------------------------------------- #
# Front-end 1: aiortc bot-join capture (production path)
# --------------------------------------------------------------------------- #
class AiortcIngestion:
    """
    Wraps aiortc `MediaStreamTrack`s (one video + one audio track per remote
    participant, obtained from the bot's RTCPeerConnection `on('track')`
    callback) and feeds them into `IngestionCore`.

    Usage:
        core = IngestionCore(cfg)
        ingestion = AiortcIngestion(core)

        @pc.on("track")
        def on_track(track):
            asyncio.ensure_future(ingestion.consume_track(participant_id, track))
    """

    def __init__(self, core: IngestionCore):
        self.core = core
        self._resamplers: Dict[str, "av.AudioResampler"] = {}  # noqa: F821 (av imported lazily)

    async def consume_track(self, participant_id: str, track) -> None:
        if track.kind == "video":
            await self._consume_video(participant_id, track)
        elif track.kind == "audio":
            await self._consume_audio(participant_id, track)

    async def _consume_video(self, participant_id: str, track) -> None:
        while True:
            try:
                frame = await track.recv()
            except Exception:
                break  # track ended (participant left / connection closed)
            image_bgr = frame.to_ndarray(format="bgr24")
            # frame.pts/time_base give the media timestamp; fall back to wall clock.
            ts = float(frame.time) if frame.time is not None else time.monotonic()
            await self.core.on_video_frame(participant_id, image_bgr, ts)

    async def _consume_audio(self, participant_id: str, track) -> None:
        import av  # local import: only required if this front-end is used

        target_rate = self.core.cfg.audio_sample_rate
        if participant_id not in self._resamplers:
            self._resamplers[participant_id] = av.AudioResampler(format="flt", layout="mono", rate=target_rate)
        resampler = self._resamplers[participant_id]

        while True:
            try:
                frame = await track.recv()
            except Exception:
                break
            resampled_frames = resampler.resample(frame)
            for rframe in resampled_frames:
                samples = rframe.to_ndarray().flatten().astype(np.float32)
                await self.core.on_audio_chunk(participant_id, samples)


# --------------------------------------------------------------------------- #
# Front-end 2: local OBS / FFmpeg virtual-camera capture (test harness)
# --------------------------------------------------------------------------- #
class LocalCameraIngestion:
    """
    Reads from a local video device (e.g. OBS Virtual Camera, index 0/1/2...)
    via OpenCV, and optionally a local audio input device via `sounddevice`,
    for testing the pipeline without a live WebRTC session.

    Video and audio are captured on independent loops but share the same
    `time.monotonic()` clock so the audio_window attached to each RawFrame
    stays correctly time-aligned per doc §1.
    """

    def __init__(self, core: IngestionCore, participant_id: str = "local_test_user",
                 video_device: int = 0, audio_device: Optional[int] = None):
        self.core = core
        self.participant_id = participant_id
        self.video_device = video_device
        self.audio_device = audio_device
        self._running = False

    async def run(self) -> None:
        self._running = True
        cap = cv2.VideoCapture(self.video_device)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video device {self.video_device}")

        audio_task = None
        if self.audio_device is not None:
            audio_task = asyncio.ensure_future(self._run_audio())

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    await asyncio.sleep(0.01)
                    continue
                await self.core.on_video_frame(self.participant_id, frame, time.monotonic())
                await asyncio.sleep(0.001)  # yield to event loop; FPS gate does the real limiting
        finally:
            cap.release()
            if audio_task:
                audio_task.cancel()

    async def _run_audio(self) -> None:
        import sounddevice as sd  # local import: only required if using local audio capture

        sample_rate = self.core.cfg.audio_sample_rate
        block_size = int(sample_rate * 0.1)  # 100ms chunks

        loop = asyncio.get_event_loop()
        q: "asyncio.Queue[np.ndarray]" = asyncio.Queue()

        def _callback(indata, frames, time_info, status):
            loop.call_soon_threadsafe(q.put_nowait, indata[:, 0].copy())

        with sd.InputStream(samplerate=sample_rate, channels=1, blocksize=block_size,
                             device=self.audio_device, callback=_callback):
            while self._running:
                samples = await q.get()
                await self.core.on_audio_chunk(self.participant_id, samples)

    def stop(self) -> None:
        self._running = False
