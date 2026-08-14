"""
Tests for the deployment orchestrator (Chunk 10).
Uses fully-injected stages with fake scorers — end-to-end integration test
without any trained models, GPU, or camera needed.
Run with: pytest tests/test_orchestrator.py -v
"""
import asyncio
import numpy as np
import pytest

from pipeline.config import PipelineConfig, CascadeConfig, FusionConfig
from pipeline.types import RawFrame, QualityResult, CascadeResult, AlignedFace
from pipeline.quality_gate import QualityGate, FaceDetection
from pipeline.cascade_router import CascadeRouter
from pipeline.face_align import FaceAligner
from pipeline.branches.branch_runner import BranchRunner
from pipeline.branches.spatial_branch import heuristic_spatial
from pipeline.branches.frequency_branch import heuristic_frequency_scorer
from pipeline.branches.av_sync_branch import heuristic_sync, heuristic_voice_clone
from pipeline.temporal import heuristic_sequence_scorer
from pipeline.fusion import DecisionEngine, heuristic_fusion
from pipeline.ingestion import IngestionCore
from pipeline.orchestrator import PipelineOrchestrator


def _frontal_keypoints_for_shape(w, h):
    """Synthetic frontal-face 6 keypoints (matching quality_gate._MODEL_POINTS_3D)."""
    from pipeline.quality_gate import _MODEL_POINTS_3D
    cx, cy = w / 2, h / 2
    scale = 3.0
    pts = []
    for X, Y, _Z in _MODEL_POINTS_3D:
        pts.append((cx + X * scale, cy - Y * scale))
    return pts


def _make_468_landmarks(w, h):
    """468 synthetic landmarks for face alignment."""
    from pipeline.face_align import (
        LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX,
        MOUTH_TOP_IDX, MOUTH_BOTTOM_IDX, FACE_OVAL_IDX,
    )
    cx, cy = w / 2, h / 2
    n_points = 468
    pts = np.zeros((n_points, 2), dtype=np.float32)
    face_radius = 80
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts[:, 0] = cx + face_radius * np.cos(angles)
    pts[:, 1] = cy + face_radius * np.sin(angles)

    for idx in FACE_OVAL_IDX:
        a = 2 * np.pi * idx / n_points
        pts[idx] = [cx + face_radius * np.cos(a), cy + face_radius * np.sin(a)]

    eye_half_sep = 30.0
    for idx in LEFT_EYE_IDX:
        pts[idx] = [cx - eye_half_sep, cy]
    for idx in RIGHT_EYE_IDX:
        pts[idx] = [cx + eye_half_sep, cy]

    pts[MOUTH_LEFT_IDX] = [cx - 15, cy + 40]
    pts[MOUTH_RIGHT_IDX] = [cx + 15, cy + 40]
    pts[MOUTH_TOP_IDX] = [cx, cy + 30]
    pts[MOUTH_BOTTOM_IDX] = [cx, cy + 50]

    return [tuple(p) for p in pts]


def _build_orchestrator(
    cascade_score=0.5,
    fusion_score=0.3,
):
    """Build a fully-injected orchestrator for testing."""
    w, h = 300, 300
    kps = _frontal_keypoints_for_shape(w, h)
    landmarks_468 = _make_468_landmarks(w, h)

    config = PipelineConfig()

    # Inject face detector (QC)
    def fake_qc_detector(img):
        return [FaceDetection(
            bbox=(0, 0, w, h),
            keypoints=kps,
            detection_confidence=0.95,
        )]

    # Inject landmarker (face align)
    def fake_landmarker(img):
        return landmarks_468

    qc = QualityGate(
        cfg=config.quality_gate,
        face_detector=fake_qc_detector,
    )
    # Override thresholds for testing
    qc.cfg.laplacian_var_threshold = 1.0
    qc.cfg.min_mean_brightness = 0.0
    qc.cfg.max_mean_brightness = 255.0

    cascade = CascadeRouter(
        cfg=config.cascade,
        scorer=lambda x: cascade_score,
    )

    aligner = FaceAligner(
        cfg=config.alignment,
        landmarker=fake_landmarker,
    )

    branch_runner = BranchRunner(
        spatial_scorer=heuristic_spatial(),
        freq_scorer=heuristic_frequency_scorer(),
        sync_scorer=heuristic_sync(),
        voice_clone_scorer=heuristic_voice_clone(),
    )

    decision_engine = DecisionEngine(
        cfg=config.fusion,
        fusion_model=lambda f: fusion_score,
    )

    ingestion = IngestionCore(config.ingestion)

    return PipelineOrchestrator(
        config=config,
        ingestion_core=ingestion,
        quality_gate=qc,
        cascade_router=cascade,
        face_aligner=aligner,
        branch_runner=branch_runner,
        decision_engine=decision_engine,
    )


def _make_test_frame(w=300, h=300, participant_id="p1", frame_idx=1):
    """A test frame that passes QC (random noise = non-blurry)."""
    return RawFrame(
        participant_id=participant_id,
        frame_idx=frame_idx,
        timestamp=float(frame_idx) * 0.33,
        image_bgr=np.random.randint(0, 255, (h, w, 3), dtype=np.uint8),
        audio_window=np.random.randn(16000).astype(np.float32) * 0.1,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestPipelineOrchestrator:
    @pytest.mark.asyncio
    async def test_process_single_frame_end_to_end(self):
        orch = _build_orchestrator()
        frame = _make_test_frame()
        decision = await orch.process_frame(frame)
        assert decision is not None
        assert decision.participant_id == "p1"
        assert 0.0 <= decision.smoothed_score <= 1.0
        assert orch.stats["frames_processed"] == 1

    @pytest.mark.asyncio
    async def test_cascade_drops_low_suspicion_frames(self):
        orch = _build_orchestrator(cascade_score=0.01)  # below 0.15 threshold
        frame = _make_test_frame()
        decision = await orch.process_frame(frame)
        assert decision is None
        assert orch.stats["frames_cascade_dropped"] == 1

    @pytest.mark.asyncio
    async def test_multiple_participants(self):
        orch = _build_orchestrator()
        frame_alice = _make_test_frame(participant_id="alice", frame_idx=1)
        frame_bob = _make_test_frame(participant_id="bob", frame_idx=1)

        d_alice = await orch.process_frame(frame_alice)
        d_bob = await orch.process_frame(frame_bob)

        assert d_alice.participant_id == "alice"
        assert d_bob.participant_id == "bob"
        assert orch.stats["frames_processed"] == 2

    @pytest.mark.asyncio
    async def test_remove_participant_cleans_state(self):
        orch = _build_orchestrator()
        frame = _make_test_frame(participant_id="alice")
        await orch.process_frame(frame)
        assert "alice" in orch._temporal_trackers

        orch.remove_participant("alice")
        assert "alice" not in orch._temporal_trackers

    @pytest.mark.asyncio
    async def test_multiple_frames_accumulate_stats(self):
        orch = _build_orchestrator()
        for i in range(5):
            frame = _make_test_frame(frame_idx=i + 1)
            await orch.process_frame(frame)

        assert orch.stats["frames_received"] == 5
        assert orch.stats["frames_processed"] == 5

    @pytest.mark.asyncio
    async def test_output_queue_receives_decisions(self):
        orch = _build_orchestrator()
        frame = _make_test_frame()
        await orch.process_frame(frame)

        assert not orch.output_queue.empty()
        decision = await orch.output_queue.get()
        assert decision.participant_id == "p1"

    @pytest.mark.asyncio
    async def test_callback_is_invoked(self):
        received = []
        orch = _build_orchestrator()
        orch._on_decision = lambda d: received.append(d)

        frame = _make_test_frame()
        await orch.process_frame(frame)

        assert len(received) == 1
        assert received[0].participant_id == "p1"
