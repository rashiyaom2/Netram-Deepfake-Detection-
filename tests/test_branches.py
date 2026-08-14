"""
Tests for Stage 4 branches (spatial, frequency, A/V sync) + branch runner.
Uses injected heuristic scorers — no trained models or GPU required.
Run with: pytest tests/test_branches.py -v
"""
import numpy as np
import pytest

from pipeline.config import BranchConfig
from pipeline.types import AlignedFace, BranchScores
from pipeline.branches.spatial_branch import (
    SpatialBranch, heuristic_spatial, preprocess_spatial,
)
from pipeline.branches.frequency_branch import (
    FrequencyBranch, heuristic_frequency_scorer,
    compute_dct_spectrum, compute_fft_spectrum,
    high_freq_energy_ratio, mid_freq_variance,
)
from pipeline.branches.av_sync_branch import (
    AVSyncBranch, heuristic_sync, heuristic_voice_clone,
    compute_mel_spectrogram, spectral_discontinuity_score,
    mouth_motion_energy,
)
from pipeline.branches.branch_runner import BranchRunner


def _make_face_crop(size=128):
    """Random RGB float32 [0,1] face crop."""
    return np.random.rand(size, size, 3).astype(np.float32)


def _make_mouth_crop(size=96):
    return np.random.rand(size, size, 3).astype(np.float32)


def _make_audio_window(n_samples=16000):
    return np.random.randn(n_samples).astype(np.float32) * 0.1


def _make_aligned_face(with_audio=True):
    return AlignedFace(
        participant_id="p1",
        frame_idx=1,
        timestamp=0.0,
        face_crop=_make_face_crop(),
        mouth_crop=_make_mouth_crop(),
        landmarks={
            "left_eye_outer": (50.0, 40.0), "left_eye_inner": (70.0, 40.0),
            "right_eye_outer": (90.0, 40.0), "right_eye_inner": (110.0, 40.0),
            "mouth_left": (60.0, 80.0), "mouth_right": (100.0, 80.0),
            "mouth_top": (80.0, 70.0), "mouth_bottom": (80.0, 90.0),
            "nose_tip": (80.0, 55.0), "chin": (80.0, 110.0),
        },
        pose_confidence=0.9,
        audio_window=_make_audio_window() if with_audio else None,
    )


# --------------------------------------------------------------------------- #
# Spatial branch tests
# --------------------------------------------------------------------------- #
class TestSpatialBranch:
    def test_preprocess_spatial_output_shape(self):
        crop = _make_face_crop(128)
        chw = preprocess_spatial(crop, (224, 224))
        assert chw.shape == (3, 224, 224)
        assert chw.dtype == np.float32

    def test_heuristic_spatial_returns_valid_score_and_embedding(self):
        scorer = heuristic_spatial()
        crop = _make_face_crop()
        p_spatial, emb = scorer(crop)
        assert 0.0 <= p_spatial <= 1.0
        assert emb.shape == (512,)
        assert emb.dtype == np.float32

    def test_spatial_branch_with_heuristic(self):
        branch = SpatialBranch(scorer=heuristic_spatial())
        p, emb = branch.run(_make_face_crop())
        assert 0.0 <= p <= 1.0
        assert emb.shape == (512,)

    def test_heuristic_spatial_deterministic_for_same_input(self):
        scorer = heuristic_spatial()
        crop = _make_face_crop()
        p1, e1 = scorer(crop)
        p2, e2 = scorer(crop)
        assert p1 == pytest.approx(p2)
        assert np.allclose(e1, e2)

    def test_onnx_spatial_with_model_if_present(self):
        import os
        from pipeline.branches.spatial_branch import default_onnx_spatial
        cfg = BranchConfig(spatial_onnx_path="models/deepfake_detector.onnx", spatial_input_size=(224, 224))
        if os.path.exists("models/deepfake_detector.onnx") or os.path.exists("model.onnx"):
            scorer = default_onnx_spatial(cfg)
            crop = _make_face_crop(224)
            p_spatial, emb = scorer(crop)
            assert 0.0 <= p_spatial <= 1.0
            assert emb.shape == (512,)
            assert emb.dtype == np.float32

    def test_onnx_spatial_missing_fallback(self):
        from pipeline.branches.spatial_branch import default_onnx_spatial
        cfg = BranchConfig(spatial_onnx_path="models/missing_model_99999.onnx", spatial_input_size=(224, 224))
        scorer = default_onnx_spatial(cfg)
        crop = _make_face_crop(224)
        p_spatial, emb = scorer(crop)
        assert 0.0 <= p_spatial <= 1.0
        assert emb.shape == (512,)


# --------------------------------------------------------------------------- #
# Frequency branch tests
# --------------------------------------------------------------------------- #
class TestFrequencyBranch:
    def test_compute_dct_spectrum_shape(self):
        crop = _make_face_crop(64)
        spec = compute_dct_spectrum(crop)
        assert spec.shape == (64, 64)

    def test_compute_fft_spectrum_shape(self):
        crop = _make_face_crop(64)
        spec = compute_fft_spectrum(crop)
        assert spec.shape == (64, 64)

    def test_high_freq_energy_ratio_in_unit_range(self):
        spec = np.random.rand(64, 64).astype(np.float64)
        ratio = high_freq_energy_ratio(spec)
        assert 0.0 <= ratio <= 1.0

    def test_mid_freq_variance_non_negative(self):
        spec = np.random.rand(64, 64).astype(np.float64)
        var = mid_freq_variance(spec)
        assert var >= 0.0

    def test_frequency_branch_returns_valid_score(self):
        branch = FrequencyBranch(scorer=heuristic_frequency_scorer())
        p_freq = branch.run(_make_face_crop())
        assert 0.0 <= p_freq <= 1.0

    def test_flat_image_different_from_noisy_image(self):
        scorer = heuristic_frequency_scorer()
        flat = np.full((64, 64, 3), 0.5, dtype=np.float32)
        noisy = np.random.rand(64, 64, 3).astype(np.float32)
        score_flat = scorer(flat)
        score_noisy = scorer(noisy)
        # They should produce different scores
        assert score_flat != pytest.approx(score_noisy, abs=0.01)


# --------------------------------------------------------------------------- #
# A/V sync branch tests
# --------------------------------------------------------------------------- #
class TestAVSyncBranch:
    def test_mel_spectrogram_shape(self):
        audio = _make_audio_window(8000)
        mel = compute_mel_spectrogram(audio, sample_rate=16000)
        assert mel.ndim == 2
        assert mel.shape[0] == 80  # n_mels

    def test_mel_spectrogram_handles_short_audio(self):
        audio = np.zeros(100, dtype=np.float32)
        mel = compute_mel_spectrogram(audio, sample_rate=16000)
        assert mel.ndim == 2

    def test_spectral_discontinuity_score_in_unit_range(self):
        audio = _make_audio_window()
        score = spectral_discontinuity_score(audio)
        assert 0.0 <= score <= 1.0

    def test_mouth_motion_energy_non_negative(self):
        crop = _make_mouth_crop()
        energy = mouth_motion_energy(crop)
        assert energy >= 0.0

    def test_heuristic_sync_returns_valid_output(self):
        scorer = heuristic_sync()
        crop = _make_mouth_crop()
        audio = _make_audio_window()
        p_sync, av_mismatch = scorer(crop, audio)
        assert 0.0 <= p_sync <= 1.0
        assert isinstance(av_mismatch, bool)

    def test_heuristic_voice_clone_returns_valid_score(self):
        scorer = heuristic_voice_clone()
        audio = _make_audio_window()
        p_vc = scorer(audio)
        assert 0.0 <= p_vc <= 1.0

    def test_av_sync_branch_with_audio(self):
        branch = AVSyncBranch(
            sync_scorer=heuristic_sync(),
            voice_clone_scorer=heuristic_voice_clone(),
        )
        p_sync, av_mm, p_vc = branch.run(_make_mouth_crop(), _make_audio_window())
        assert p_sync is not None
        assert 0.0 <= p_sync <= 1.0
        assert p_vc is not None
        assert 0.0 <= p_vc <= 1.0

    def test_av_sync_branch_without_audio(self):
        branch = AVSyncBranch(
            sync_scorer=heuristic_sync(),
            voice_clone_scorer=heuristic_voice_clone(),
        )
        p_sync, av_mm, p_vc = branch.run(_make_mouth_crop(), None)
        assert p_sync is None
        assert av_mm is None
        assert p_vc is None

    def test_default_torch_sync_with_real_model(self):
        from pipeline.branches.av_sync_branch import default_torch_sync
        cfg = BranchConfig(sync_weights_path="models/sync_net.pt")
        scorer = default_torch_sync(cfg)
        crop = _make_mouth_crop()
        audio = _make_audio_window()
        p_sync, av_mismatch = scorer(crop, audio)
        assert 0.0 <= p_sync <= 1.0
        assert isinstance(av_mismatch, bool)

    def test_default_torch_voice_clone_with_real_model(self):
        from pipeline.branches.av_sync_branch import default_torch_voice_clone
        cfg = BranchConfig(voice_clone_weights_path="assist/weights/AASIST.pth")
        scorer = default_torch_voice_clone(cfg)
        audio = _make_audio_window()
        p_vc = scorer(audio)
        assert 0.0 <= p_vc <= 1.0


# --------------------------------------------------------------------------- #
# Branch runner integration tests
# --------------------------------------------------------------------------- #
class TestBranchRunner:
    def _make_runner(self):
        return BranchRunner(
            spatial_scorer=heuristic_spatial(),
            freq_scorer=heuristic_frequency_scorer(),
            sync_scorer=heuristic_sync(),
            voice_clone_scorer=heuristic_voice_clone(),
        )

    def test_produces_branch_scores_with_audio(self):
        runner = self._make_runner()
        aligned = _make_aligned_face(with_audio=True)
        scores = runner.run(aligned)
        assert isinstance(scores, BranchScores)
        assert 0.0 <= scores.p_spatial <= 1.0
        assert 0.0 <= scores.p_freq <= 1.0
        assert scores.embedding.shape == (512,)
        assert scores.p_sync is not None
        assert 0.0 <= scores.p_sync <= 1.0
        assert scores.p_voice_clone is not None

    def test_produces_branch_scores_without_audio(self):
        runner = self._make_runner()
        aligned = _make_aligned_face(with_audio=False)
        scores = runner.run(aligned)
        assert isinstance(scores, BranchScores)
        assert 0.0 <= scores.p_spatial <= 1.0
        assert 0.0 <= scores.p_freq <= 1.0
        assert scores.p_sync is None
        assert scores.p_voice_clone is None
