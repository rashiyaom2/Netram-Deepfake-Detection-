"""
Tests for Stage 1 ingestion. Pure synthetic data — no camera, no aiortc,
no live call needed. Run with: pytest tests/test_ingestion.py -v
"""
import asyncio
import numpy as np
import pytest

from pipeline.config import IngestionConfig
from pipeline.ingestion import IngestionCore, RollingAudioBuffer, ParticipantBuffer


def test_rolling_audio_buffer_wraps_correctly():
    buf = RollingAudioBuffer(sample_rate=100, seconds=1.0)  # capacity = 100
    buf.append(np.arange(60, dtype=np.float32))
    buf.append(np.arange(60, 100, dtype=np.float32))  # 40 more, total 100 written, wraps by 0

    snap = buf.snapshot()
    assert len(snap) == 100
    # Should be fully filled, chronological order preserved
    assert snap[0] == 0.0
    assert snap[-1] == 99.0


def test_rolling_audio_buffer_zero_pads_when_not_full():
    buf = RollingAudioBuffer(sample_rate=100, seconds=1.0)
    buf.append(np.ones(30, dtype=np.float32))
    snap = buf.snapshot()
    assert len(snap) == 100
    assert np.all(snap[:70] == 0.0)
    assert np.all(snap[70:] == 1.0)


def test_video_buffer_respects_maxlen():
    cfg = IngestionConfig(video_buffer_frames=5, video_fps_sample=100.0)  # high fps so gate never blocks
    pbuf = ParticipantBuffer("p1", cfg)
    for i in range(10):
        pbuf.push_video_frame(np.zeros((4, 4, 3), dtype=np.uint8), timestamp=float(i))
        # video_fps_sample=100.0 -> min_interval=0.01s, so integer-second steps always pass the gate
    assert len(pbuf.video_buffer) == 5
    # Should hold the most recent 5 timestamps: 5,6,7,8,9
    timestamps = [t for t, _ in pbuf.video_buffer]
    assert timestamps == [5.0, 6.0, 7.0, 8.0, 9.0]


@pytest.mark.asyncio
async def test_fps_gate_drops_frames_faster_than_target():
    cfg = IngestionConfig(video_fps_sample=2.0)  # min interval = 0.5s
    core = IngestionCore(cfg)

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    await core.on_video_frame("p1", frame, timestamp=0.0)
    await core.on_video_frame("p1", frame, timestamp=0.1)   # too soon, dropped
    await core.on_video_frame("p1", frame, timestamp=0.4)   # still too soon, dropped
    await core.on_video_frame("p1", frame, timestamp=0.6)   # >= 0.5s since last accepted, kept

    accepted = []
    while not core.output_queue.empty():
        accepted.append(await core.output_queue.get())

    assert len(accepted) == 2
    assert [f.timestamp for f in accepted] == [0.0, 0.6]


@pytest.mark.asyncio
async def test_raw_frame_carries_matching_audio_window():
    cfg = IngestionConfig(video_fps_sample=100.0, audio_sample_rate=100, audio_buffer_seconds=1.0)
    core = IngestionCore(cfg)

    await core.on_audio_chunk("p1", np.full(50, 7.0, dtype=np.float32))
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    await core.on_video_frame("p1", frame, timestamp=0.0)

    raw = await core.output_queue.get()
    assert raw.audio_window is not None
    assert len(raw.audio_window) == 100
    assert raw.audio_window[-1] == 7.0  # most recent audio samples present at the tail


@pytest.mark.asyncio
async def test_multiple_participants_are_isolated():
    cfg = IngestionConfig(video_fps_sample=100.0)
    core = IngestionCore(cfg)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    await core.on_video_frame("alice", frame, timestamp=0.0)
    await core.on_video_frame("bob", frame, timestamp=0.0)

    assert set(core.participants.keys()) == {"alice", "bob"}
    assert core.output_queue.qsize() == 2
