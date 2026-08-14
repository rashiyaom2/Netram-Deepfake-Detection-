"""
Central configuration for the deepfake detection pipeline.

Every stage (Stage 0 - Stage 6) reads its knobs from here so thresholds
can be tuned/calibrated in one place without touching stage code.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class IngestionConfig:
    video_fps_sample: float = 3.0          # 2-5 FPS per doc §1
    audio_sample_rate: int = 16000
    video_buffer_frames: int = 15          # circular buffer size (10-15 per doc)
    audio_buffer_seconds: float = 4.0      # rolling window (3-5s per doc)


@dataclass
class QualityGateConfig:
    laplacian_var_threshold: float = 15.0      # realistic threshold for compressed WebRTC webcam frames
    max_yaw_pitch_deg: float = 48.0            # hard reject beyond this
    frontal_deg_for_full_weight: float = 22.0  # <=22deg contributes full weight
    min_face_size_px: int = 35                 # discard crops smaller than this
    min_mean_brightness: float = 15.0          # handle dim/evening webcam lighting
    max_mean_brightness: float = 245.0         # handle bright background lighting
    # MediaPipe Tasks model path (blaze_face_short_range.tflite)
    face_detector_model_path: str = "models/blaze_face_short_range.tflite"



@dataclass
class AlignmentConfig:
    margin_pct: float = 0.25          # 20-30% border padding per doc §3
    output_size: Tuple[int, int] = (299, 299)  # 224 or 299, model-dependent
    mouth_crop_size: Tuple[int, int] = (96, 96)   # for the audio-visual sync branch (§4a)
    mouth_margin_pct: float = 0.35    # generous margin so lip motion isn't clipped at crop edges
    # MediaPipe Tasks model path (face_landmarker.task)
    face_landmarker_model_path: str = "models/face_landmarker.task"


@dataclass
class CascadeConfig:
    suspicion_threshold: float = 0.15   # below this, frame is dropped, not escalated
    model_name: str = "efficientnet_b0"
    onnx_path: str = "models/cascade_b0.onnx"
    input_size: Tuple[int, int] = (224, 224)
    crop_margin_pct: float = 0.15       # looser than QC's tight bbox, tighter than Stage 3's aligned crop


@dataclass
class BranchConfig:
    spatial_model_name: str = "deepfake_detector_vit"   # or "efficientnet_b4", "swin_tiny_patch4_window7_224"
    spatial_weights_path: str = "models/spatial_b4.pt"
    spatial_onnx_path: str = "models/deepfake_detector.onnx"
    spatial_input_size: Tuple[int, int] = (224, 224)

    freq_input_size: Tuple[int, int] = (224, 224)

    sync_model_name: str = "wav2lip_sync_distilled"
    sync_weights_path: str = "models/sync_net.pt"
    voice_clone_weights_path: str = "assist/weights/AASIST.pth"


@dataclass
class TemporalConfig:
    embedding_dim: int = 512
    sequence_len: int = 15           # matches video_buffer_frames
    gru_hidden_dim: int = 128
    jitter_landmark_ids: Tuple[str, ...] = ("left_eye_corner", "right_eye_corner",
                                             "upper_lip", "lower_lip")
    temporal_weights_path: str = "models/temporal_gru.pt"
    liveness_min_variance: float = 0.5   # below this variance across the window, assume presentation attack (static photo)
    frozen_threshold: float = 4e-5       # non-rigid strain variance threshold for static 2D plane detection
    inactivity_timeout_s: float = 5.0    # auto reset participant state if no updates received for > 5s
    gru_stride: int = 2                  # run heavy GRU every N frames to optimize latency
    max_pose_angle_deg: float = 30.0     # gate EAR blink detection if head pose pitch/yaw > 30 degrees
    blink_window_seconds: float = 15.0   # rolling window duration for blink rate checking
    min_blinks_in_window: int = 1        # minimum natural blinks expected in rolling window
    ear_relative_drop: float = 0.30      # relative dip from adaptive baseline (30% drop below personal open-eye EAR)
    ear_blink_threshold: float = 0.18    # fallback static threshold during initial baseline calibration
    min_blink_duration_s: float = 0.06   # 60ms minimum duration for a physiological blink (filters 1-frame glitches)
    max_blink_duration_s: float = 0.65   # 650ms maximum duration for a natural blink (filters prolonged unnatural closures)



@dataclass
class FusionConfig:
    fusion_model_path: str = "models/fusion_head.pkl"   # sklearn LogisticRegression/MLP, pickled
    smoothing_alpha: float = 0.7                         # S_t = 0.7*P + 0.3*S_{t-1} for responsive smoothing
    review_threshold: float = 0.55     # "flag for human review" - calibrate via ROC
    block_threshold: float = 0.80      # "auto-block/warn all participants" - calibrate via ROC
    sustained_seconds: float = 2.5     # must exceed threshold this long to trigger



@dataclass
class PipelineConfig:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    quality_gate: QualityGateConfig = field(default_factory=QualityGateConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    branches: BranchConfig = field(default_factory=BranchConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    device: str = "cuda"   # falls back to "cpu" automatically at load time if unavailable


DEFAULT_CONFIG = PipelineConfig()
