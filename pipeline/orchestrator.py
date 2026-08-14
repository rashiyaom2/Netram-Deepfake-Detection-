"""
Stage 10 — Deployment Orchestrator (doc Deployment Notes).

Ties all pipeline stages (0-6) together into a single async runner that:
  - Consumes RawFrames from the ingestion queue
  - Runs quality gating → cascade routing → alignment → branches →
    temporal → fusion for each frame
  - Manages per-participant state (temporal tracker, fusion engine)
  - Emits FrameDecision results via an output callback or queue

Also includes:
  - Compliance notes (BIPA, GDPR — doc: "Privacy/consent")
  - Human-in-the-loop recommendation (flag-for-review, not auto-block)
  - Compute budget notes (cascade is what makes >4-5 participants tractable)

Usage:
    from pipeline.orchestrator import PipelineOrchestrator
    from pipeline.config import PipelineConfig

    config = PipelineConfig()
    orch = PipelineOrchestrator(config)
    await orch.run()   # blocks, consuming from the ingestion queue
"""
import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from pipeline.config import PipelineConfig
from pipeline.types import (
    RawFrame, QualityResult, CascadeResult, AlignedFace,
    BranchScores, TemporalResult, FrameDecision,
)
from pipeline.quality_gate import QualityGate
from pipeline.cascade_router import CascadeRouter
from pipeline.face_align import FaceAligner
from pipeline.branches.branch_runner import BranchRunner
from pipeline.temporal import TemporalTracker
from pipeline.fusion import DecisionEngine
from pipeline.ingestion import IngestionCore


logger = logging.getLogger(__name__)

# Type for the output callback
DecisionCallback = Callable[[FrameDecision], None]


# --------------------------------------------------------------------------- #
# Compliance checklist (doc: "Privacy/consent")
# --------------------------------------------------------------------------- #
COMPLIANCE_NOTES = """
=== COMPLIANCE CHECKLIST (must be reviewed before deployment) ===

1. BIOMETRIC CONSENT (BIPA-style laws):
   - Running biometric analysis (face landmarks, voice features) on meeting
     participants requires explicit consent in jurisdictions with BIPA-style
     laws (Illinois, Texas, Washington state, etc.).
   - Implement consent collection before activating the pipeline.

2. GDPR (EU participants):
   - Face/voice analysis constitutes processing of biometric data under
     GDPR Article 9 — requires explicit consent or a legal basis.
   - Implement data minimisation: do not store raw frames/audio beyond
     the rolling buffer lifetime.
   - Provide a right-to-erasure mechanism for any stored analysis results.

3. DISCLOSURE:
   - All meeting participants must be informed that deepfake detection
     is active, even if they are not the subject of analysis.
   - Consider a visible indicator in the meeting UI.

4. HUMAN-IN-THE-LOOP (doc recommendation):
   - Production behaviour should be flag-for-review, NOT auto-block.
   - The cost of wrongly accusing a real participant is much higher than
     the cost of a missed detection needing follow-up review.
   - Auto-block should only trigger at the higher calibrated threshold
     after sustained detection (3+ seconds per doc §6).

5. DATA RETENTION:
   - Define retention policies for detection results and any audit logs.
   - Do not retain raw video/audio frames beyond the pipeline's rolling buffers.
"""


# --------------------------------------------------------------------------- #
# Pipeline orchestrator
# --------------------------------------------------------------------------- #
class PipelineOrchestrator:
    """
    Async orchestrator that connects all pipeline stages.

    Consumes RawFrames from an IngestionCore's output queue and produces
    FrameDecisions via a callback or output queue.
    """

    def __init__(self, config: Optional[PipelineConfig] = None,
                 ingestion_core: Optional[IngestionCore] = None,
                 on_decision: Optional[DecisionCallback] = None,
                 # Dependency injection for all stages (for testing)
                 quality_gate: Optional[QualityGate] = None,
                 cascade_router: Optional[CascadeRouter] = None,
                 face_aligner: Optional[FaceAligner] = None,
                 branch_runner: Optional[BranchRunner] = None,
                 decision_engine: Optional[DecisionEngine] = None):
        self.config = config or PipelineConfig()
        self.ingestion = ingestion_core or IngestionCore(self.config.ingestion)

        # Pipeline stages
        self._quality_gate = quality_gate or QualityGate(self.config.quality_gate)
        self._cascade_router = cascade_router or CascadeRouter(self.config.cascade)
        self._face_aligner = face_aligner or FaceAligner(self.config.alignment)
        self._branch_runner = branch_runner or BranchRunner(self.config.branches)
        self._decision_engine = decision_engine or DecisionEngine(self.config.fusion)

        # Per-participant temporal trackers
        self._temporal_trackers: Dict[str, TemporalTracker] = {}

        # Output
        self._on_decision = on_decision
        self.output_queue: "asyncio.Queue[FrameDecision]" = asyncio.Queue(maxsize=256)

        # Statistics
        self._stats = {
            "frames_received": 0,
            "frames_qc_rejected": 0,
            "frames_cascade_dropped": 0,
            "frames_alignment_failed": 0,
            "frames_processed": 0,
            "decisions_review": 0,
            "decisions_block": 0,
        }
        self._running = False

    def _get_temporal_tracker(self, participant_id: str) -> TemporalTracker:
        if participant_id not in self._temporal_trackers:
            self._temporal_trackers[participant_id] = TemporalTracker(
                self.config.temporal
            )
        return self._temporal_trackers[participant_id]

    def _sync_process_frame(self, raw_frame: RawFrame) -> Optional[FrameDecision]:
        """
        Synchronous CPU/GPU inference pipeline for a single frame.
        Executed in a background thread to prevent starving camera capture / UI.
        """
        self._stats["frames_received"] += 1
        pid = raw_frame.participant_id

        # Stage 2: Quality gating
        quality = self._quality_gate.run(raw_frame)
        if not quality.passed:
            self._stats["frames_qc_rejected"] += 1
            logger.debug(f"[{pid}] frame {raw_frame.frame_idx} rejected by QC: "
                         f"{quality.reject_reason}")
            return None

        # Stage 0: Cascade routing
        cascade = self._cascade_router.route(raw_frame, quality)
        if cascade is None or not cascade.escalate:
            self._stats["frames_cascade_dropped"] += 1
            logger.debug(f"[{pid}] frame {raw_frame.frame_idx} dropped by cascade "
                         f"(score={cascade.suspicion_score if cascade else 'N/A'})")
            # Keep temporal tracking updated even on cascade-dropped frames
            aligned = self._face_aligner.align(raw_frame, quality)
            if aligned is not None:
                tracker = self._get_temporal_tracker(pid)
                dummy_scores = BranchScores(
                    p_spatial=cascade.suspicion_score if cascade else 0.0,
                    p_freq=0.0,
                    embedding=np.zeros(512, dtype=np.float32),
                )
                tracker.update(aligned, dummy_scores)
            return None

        # Stage 3: Face alignment
        aligned = self._face_aligner.align(raw_frame, quality)
        if aligned is None:
            self._stats["frames_alignment_failed"] += 1
            logger.debug(f"[{pid}] frame {raw_frame.frame_idx} alignment failed")
            return None

        # Stage 4: Multi-modal neural branches
        branch_scores = self._branch_runner.run(aligned)

        # Blend cascade triage score if elevated
        if cascade and cascade.suspicion_score > branch_scores.p_spatial:
            branch_scores = BranchScores(
                p_spatial=max(branch_scores.p_spatial, cascade.suspicion_score),
                p_freq=branch_scores.p_freq,
                embedding=branch_scores.embedding,
                p_sync=branch_scores.p_sync,
                av_mismatch_flag=branch_scores.av_mismatch_flag,
                p_voice_clone=branch_scores.p_voice_clone,
            )

        # Stage 5: Temporal verification (continuous landmark & feature tracking)
        tracker = self._get_temporal_tracker(pid)
        temporal_result = tracker.update(aligned, branch_scores)

        # Stage 6: Decision engine & Multi-Branch Fusion
        decision = self._decision_engine.decide(
            participant_id=pid,
            frame_idx=raw_frame.frame_idx,
            timestamp=raw_frame.timestamp,
            branch_scores=branch_scores,
            temporal_result=temporal_result,
            pose_confidence=quality.pose_confidence,
        )

        self._stats["frames_processed"] += 1
        if decision.review_flag:
            self._stats["decisions_review"] += 1
        if decision.block_flag:
            self._stats["decisions_block"] += 1

        return decision

    async def process_frame(self, raw_frame: RawFrame) -> Optional[FrameDecision]:
        """
        Process a single frame through the full pipeline.
        Runs heavy inference in a worker thread so the main asyncio event loop
        (and camera capture) is never blocked.
        """
        decision = await asyncio.to_thread(self._sync_process_frame, raw_frame)

        # Emit decision on the main loop
        if decision is not None:
            if self._on_decision:
                try:
                    self._on_decision(decision)
                except Exception as e:
                    logger.debug(f"Error in on_decision callback: {e}")
            try:
                self.output_queue.put_nowait(decision)
            except asyncio.QueueFull:
                _ = self.output_queue.get_nowait()
                self.output_queue.put_nowait(decision)

        return decision

    async def run(self) -> None:
        """
        Main loop: consume frames from ingestion queue and process them.
        Runs until stop() is called.
        """
        self._running = True
        logger.info("Pipeline orchestrator started")
        logger.info(COMPLIANCE_NOTES)

        while self._running:
            try:
                raw_frame = await asyncio.wait_for(
                    self.ingestion.output_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            try:
                await self.process_frame(raw_frame)
            except Exception as e:
                logger.error(f"Error processing frame: {e}", exc_info=True)

        logger.info("Pipeline orchestrator stopped")
        logger.info(f"Stats: {self._stats}")

    def stop(self) -> None:
        """Signal the run loop to exit."""
        self._running = False

    def remove_participant(self, participant_id: str) -> None:
        """Clean up all state for a participant who left the call."""
        self.ingestion.remove_participant(participant_id)
        self._temporal_trackers.pop(participant_id, None)
        self._decision_engine.reset_participant(participant_id)
        logger.info(f"Removed participant {participant_id}")

    def reset_tracker(self, participant_id: str) -> None:
        """Explicitly reset state for a participant who rejoined."""
        if participant_id in self._temporal_trackers:
            self._temporal_trackers[participant_id].reset()
        self._decision_engine.reset_participant(participant_id)
        logger.info(f"Reset tracker state for participant {participant_id}")

    def cleanup_inactive_trackers(self, timeout_s: float = 5.0) -> List[str]:
        """Prune stale trackers for participants inactive for > timeout_s seconds."""
        stale_ids = []
        for pid, tracker in list(self._temporal_trackers.items()):
            if tracker.is_inactive(timeout_s):
                stale_ids.append(pid)
                self.remove_participant(pid)
        return stale_ids

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)


# --------------------------------------------------------------------------- #
# Convenience: quick-start for local testing
# --------------------------------------------------------------------------- #
async def run_local_pipeline(video_device: int = 0,
                               audio_device: Optional[int] = None,
                               config: Optional[PipelineConfig] = None):
    """
    Quick-start function for local testing with a webcam/virtual camera.
    Runs the full pipeline on a single participant feed.
    """
    from pipeline.ingestion import LocalCameraIngestion

    config = config or PipelineConfig()
    orchestrator = PipelineOrchestrator(config)

    camera = LocalCameraIngestion(
        orchestrator.ingestion,
        participant_id="local_test",
        video_device=video_device,
        audio_device=audio_device,
    )

    def print_decision(decision: FrameDecision):
        status = "🟢 REAL"
        if decision.block_flag:
            status = "🔴 BLOCKED"
        elif decision.review_flag:
            status = "🟡 REVIEW"
        logger.info(
            f"[{decision.participant_id}] frame={decision.frame_idx} "
            f"score={decision.smoothed_score:.3f} {status}"
            + (f" ⚠️ A/V MISMATCH" if decision.av_mismatch_flag else "")
        )

    orchestrator._on_decision = print_decision

    # Run camera capture and pipeline processing concurrently
    await asyncio.gather(
        camera.run(),
        orchestrator.run(),
    )
