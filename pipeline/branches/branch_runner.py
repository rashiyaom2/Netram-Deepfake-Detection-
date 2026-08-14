"""
Stage 4 — Branch Runner (doc §4a-4c combined).

Orchestrates the three inference branches (spatial, frequency, A/V sync)
over a single AlignedFace and assembles a BranchScores output.

Handles missing audio gracefully: sync and voice-clone scores default
to None when no audio window is available.
"""
from typing import Optional

import numpy as np

from pipeline.config import BranchConfig
from pipeline.types import AlignedFace, BranchScores
from pipeline.branches.spatial_branch import SpatialBranch, SpatialScorerFn
from pipeline.branches.frequency_branch import FrequencyBranch, FreqScorerFn
from pipeline.branches.av_sync_branch import AVSyncBranch, SyncScorerFn, VoiceCloneScorerFn


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
                 voice_clone_scorer: Optional[VoiceCloneScorerFn] = None):
        self.cfg = cfg or BranchConfig()
        self.spatial = SpatialBranch(self.cfg, scorer=spatial_scorer)
        self.frequency = FrequencyBranch(self.cfg, scorer=freq_scorer)
        self.av_sync = AVSyncBranch(self.cfg, sync_scorer=sync_scorer,
                                      voice_clone_scorer=voice_clone_scorer)

    def run(self, aligned_face: AlignedFace) -> BranchScores:
        """
        Run all three branches and return a unified BranchScores.

        The spatial and frequency branches always produce scores.
        The A/V sync branch produces scores only when audio is available.
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

        return BranchScores(
            p_spatial=p_spatial,
            p_freq=p_freq,
            embedding=embedding,
            p_sync=p_sync,
            av_mismatch_flag=av_mismatch,
            p_voice_clone=p_voice_clone,
        )
