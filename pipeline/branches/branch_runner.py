"""
Stage 4 — Branch Runner (doc §4a-4c combined).

Orchestrates the three inference branches (spatial, frequency, A/V sync)
over a single AlignedFace and assembles a BranchScores output.

Handles missing audio gracefully: sync and voice-clone scores default
to None when no audio window is available.
"""
from typing import Optional, Tuple
import numpy as np

from pipeline.config import BranchConfig
from pipeline.types import AlignedFace, BranchScores
from pipeline.branches.spatial_branch import SpatialBranch, SpatialScorerFn
from pipeline.branches.frequency_branch import FrequencyBranch, FreqScorerFn
from pipeline.branches.av_sync_branch import AVSyncBranch, SyncScorerFn, VoiceCloneScorerFn
from pipeline.branches.phone_replay import PhoneReplayDetector
from pipeline.branches.ar_filter_detector import ARFilterDetector


class BranchRunner:
    """
    Runs all Stage 4 sub-branches on one AlignedFace.

    Constructor accepts optional injected scorers for each branch
    (matching the dependency-injection pattern used in previous stages)
    so the full pipeline is testable without any trained model weights.
    """

    def __init__(self, cfg: Optional[BranchConfig] = None,
                 spatial_scorer: Optional[SpatialScorerFn] = None,
                 freq_scorer: Optional[FreqScorerFn] = None,
                 sync_scorer: Optional[SyncScorerFn] = None,
                 voice_clone_scorer: Optional[VoiceCloneScorerFn] = None,
                 phone_detector: Optional[PhoneReplayDetector] = None,
                 ar_detector: Optional[ARFilterDetector] = None):
        self.cfg = cfg or BranchConfig()
        self.spatial = SpatialBranch(self.cfg, scorer=spatial_scorer)
        self.frequency = FrequencyBranch(self.cfg, scorer=freq_scorer)
        self.av_sync = AVSyncBranch(self.cfg, sync_scorer=sync_scorer,
                                      voice_clone_scorer=voice_clone_scorer)
        self.phone_detector = phone_detector or PhoneReplayDetector()
        self.ar_detector = ar_detector or ARFilterDetector()

    def run(self, aligned_face: AlignedFace,
            raw_image_bgr: Optional[np.ndarray] = None,
            face_bbox: Optional[Tuple[int, int, int, int]] = None) -> BranchScores:
        """
        Run all sub-branches (Spatial ViT, Frequency CNN, A/V Sync, Phone Replay, AR Filter)
        and return a unified BranchScores.
        """
        # 4b: Spatial branch
        p_spatial, embedding = self.spatial.run(aligned_face.face_crop)

        # 4c: Frequency branch
        p_freq = self.frequency.run(aligned_face.face_crop)

        # 4a: Audio-visual sync branch (may return None for all fields)
        p_sync, av_mismatch, p_voice_clone = self.av_sync.run(
            aligned_face.mouth_crop,
            aligned_face.audio_window,
        )

        # 4d: Phone & Screen Replay Attack detection
        phone_detected = False
        phone_confidence = 0.0
        if raw_image_bgr is not None:
            pres = self.phone_detector.detect(raw_image_bgr, face_bbox)
            phone_detected = pres.phone_detected
            phone_confidence = pres.confidence

        # 4e: Snapchat / Instagram / AR Beauty Filter detection
        ar_res = self.ar_detector.detect(
            aligned_face.face_crop,
            aligned_face.landmarks,
            raw_image_bgr,
            face_bbox
        )

        return BranchScores(
            p_spatial=p_spatial,
            p_freq=p_freq,
            embedding=embedding,
            p_sync=p_sync,
            av_mismatch_flag=av_mismatch,
            p_voice_clone=p_voice_clone,
            phone_detected=phone_detected,
            phone_confidence=phone_confidence,
            ar_filter_detected=ar_res.filter_detected,
            ar_filter_confidence=ar_res.confidence,
            filter_type=ar_res.filter_type,
        )


