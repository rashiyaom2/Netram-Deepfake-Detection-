"""
Calibrate and export the Decision Fusion Head (Stage 6).

Generates synthetic calibration data by running the full pipeline across
diverse synthetic face frames, collects the 7-D branch feature vectors,
and fits a calibrated LogisticRegression → exports to models/fusion_head.pkl.

The spatial ViT ONNX model acts as the "ground truth" label source:
- ViT score > 0.65  → label=FAKE (1)
- ViT score < 0.30  → label=REAL (0)
- In between         → discarded (ambiguous)

Usage:
    python -m scripts.calibrate_fusion
"""
import logging
import os
import sys

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import PipelineConfig
from pipeline.quality_gate import QualityGate
from pipeline.face_align import FaceAligner
from pipeline.branches.branch_runner import BranchRunner
from pipeline.temporal import TemporalTracker
from pipeline.fusion import build_fusion_feature_vector
from pipeline.types import RawFrame, FusionInput


def generate_synthetic_face(idx: int, is_fake: bool = False) -> np.ndarray:
    """Generate a 240×320 BGR synthetic face with or without deepfake artifacts."""
    img = np.full((240, 320, 3), 40 + np.random.randint(0, 30), dtype=np.uint8)
    cx, cy = 160 + np.random.randint(-15, 15), 120 + np.random.randint(-10, 10)

    # Face ellipse with skin tones
    skin_b = np.random.randint(130, 200)
    skin_g = np.random.randint(150, 220)
    skin_r = np.random.randint(170, 240)
    cv2.ellipse(img, (cx, cy), (50 + np.random.randint(-5, 10), 70 + np.random.randint(-5, 10)),
                0, 0, 360, (skin_b, skin_g, skin_r), -1)

    # Eyes
    eye_offset = np.random.randint(15, 25)
    cv2.circle(img, (cx - eye_offset, cy - 20), 8, (255, 255, 255), -1)
    cv2.circle(img, (cx - eye_offset, cy - 20), 4, (20, 20, 20), -1)
    cv2.circle(img, (cx + eye_offset, cy - 20), 8, (255, 255, 255), -1)
    cv2.circle(img, (cx + eye_offset, cy - 20), 4, (20, 20, 20), -1)

    # Mouth
    cv2.ellipse(img, (cx, cy + 30), (16, 8), 0, 0, 360, (50, 50, 180), -1)

    if is_fake:
        # Inject various deepfake-like artifacts
        artifact = np.random.choice(["noise", "blur_boundary", "color_shift", "grid"])
        if artifact == "noise":
            region = img[cy-40:cy+40, cx-30:cx+30]
            if region.size > 0:
                noise = np.random.randint(0, 80, region.shape, dtype=np.uint8)
                img[cy-40:cy+40, cx-30:cx+30] = cv2.add(region, noise)
        elif artifact == "blur_boundary":
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask, (cx, cy), (52, 72), 0, 0, 360, 255, 3)
            blurred = cv2.GaussianBlur(img, (15, 15), 0)
            img = np.where(mask[:, :, None] > 0, blurred, img)
        elif artifact == "color_shift":
            img[cy-35:cy+35, cx-25:cx+25, 0] = np.clip(
                img[cy-35:cy+35, cx-25:cx+25, 0].astype(int) + 40, 0, 255
            ).astype(np.uint8)
        elif artifact == "grid":
            for r in range(cy-40, cy+40, 4):
                for c in range(cx-30, cx+30, 4):
                    if 0 <= r < 240 and 0 <= c < 320:
                        img[r, c] = np.clip(img[r, c].astype(int) + 30, 0, 255).astype(np.uint8)

    # Add natural-looking background variation
    noise = np.random.randint(0, 8, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


def calibrate():
    config = PipelineConfig()
    config.cascade.suspicion_threshold = 0.01  # Let everything through for calibration

    logger.info("Initializing pipeline components for calibration...")
    quality_gate = QualityGate(config.quality_gate)
    aligner = FaceAligner(config.alignment)
    branch_runner = BranchRunner(config.branches)
    temporal_tracker = TemporalTracker(config.temporal)

    features_list = []
    labels_list = []
    n_samples = 600  # 300 real + 300 fake attempts
    n_collected = 0
    n_ambiguous = 0

    logger.info(f"Generating {n_samples} synthetic calibration frames...")

    for idx in range(n_samples):
        is_fake = idx >= n_samples // 2
        frame_bgr = generate_synthetic_face(idx, is_fake=is_fake)

        raw = RawFrame(
            participant_id="calibration",
            frame_idx=idx,
            timestamp=float(idx),
            image_bgr=frame_bgr,
            audio_window=None,
        )

        quality = quality_gate.run(raw)
        if not quality.passed:
            continue

        aligned = aligner.align(raw, quality)
        if aligned is None:
            continue

        branches = branch_runner.run(aligned)
        temporal = temporal_tracker.update(aligned, branches)

        # Use spatial ViT score as label source
        p_spatial = float(branches.p_spatial)
        if p_spatial > 0.65:
            label = 1  # FAKE
        elif p_spatial < 0.30:
            label = 0  # REAL
        else:
            n_ambiguous += 1
            continue  # Ambiguous — skip

        fusion_input = FusionInput(
            p_spatial=branches.p_spatial,
            p_freq=branches.p_freq,
            p_temporal=temporal.p_temporal,
            p_sync=branches.p_sync,
            jitter=temporal.jitter_score,
            pose_confidence=quality.pose_confidence,
            p_liveness=temporal.p_liveness,
        )
        fv = build_fusion_feature_vector(fusion_input)
        features_list.append(fv)
        labels_list.append(label)
        n_collected += 1

    logger.info(f"Collected {n_collected} calibration samples ({n_ambiguous} ambiguous discarded)")

    if n_collected < 20:
        logger.warning("Too few samples for calibration. Generating fallback synthetic data.")
        # Generate fallback calibration data from known distributions
        np.random.seed(42)
        for _ in range(200):
            # Real-like: low spatial, low freq, low temporal
            fv = np.array([
                np.random.uniform(0.05, 0.25),  # p_spatial
                np.random.uniform(0.10, 0.40),  # p_freq
                np.random.uniform(0.00, 0.15),  # p_temporal
                0.5,                              # p_sync (neutral)
                np.random.uniform(0.00, 0.10),  # jitter
                np.random.uniform(0.7, 1.0),    # pose_confidence
                np.random.uniform(0.00, 0.15),  # p_liveness
            ], dtype=np.float32)
            features_list.append(fv)
            labels_list.append(0)

        for _ in range(200):
            # Fake-like: high spatial, moderate-high freq, varied temporal
            fv = np.array([
                np.random.uniform(0.55, 0.95),  # p_spatial
                np.random.uniform(0.35, 0.80),  # p_freq
                np.random.uniform(0.10, 0.60),  # p_temporal
                0.5,                              # p_sync (neutral)
                np.random.uniform(0.05, 0.40),  # jitter
                np.random.uniform(0.5, 1.0),    # pose_confidence
                np.random.uniform(0.10, 0.50),  # p_liveness
            ], dtype=np.float32)
            features_list.append(fv)
            labels_list.append(1)

        n_collected = len(features_list)
        logger.info(f"Added synthetic calibration data. Total: {n_collected} samples")

    X = np.array(features_list)
    y = np.array(labels_list)

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Label distribution: {np.bincount(y)}")

    # Fit calibrated logistic regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
        )),
    ])

    # Cross-validation
    if n_collected >= 20:
        cv_scores = cross_val_score(pipeline, X, y, cv=min(5, n_collected // 4), scoring="accuracy")
        logger.info(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit on full data
    pipeline.fit(X, y)

    # Save
    import joblib
    os.makedirs("models", exist_ok=True)
    save_path = "models/fusion_head.pkl"
    joblib.dump(pipeline, save_path)
    logger.info(f"✅ Saved calibrated fusion head to {save_path}")

    # Verify
    loaded = joblib.load(save_path)
    test_features = np.array([[0.1, 0.2, 0.05, 0.5, 0.02, 0.95, 0.05]], dtype=np.float32)
    prob = loaded.predict_proba(test_features)[0, 1]
    logger.info(f"Verification — clean face P(fake): {prob:.4f} (expected < 0.3)")

    test_features_fake = np.array([[0.8, 0.6, 0.4, 0.5, 0.3, 0.8, 0.3]], dtype=np.float32)
    prob_fake = loaded.predict_proba(test_features_fake)[0, 1]
    logger.info(f"Verification — fake face P(fake): {prob_fake:.4f} (expected > 0.7)")


if __name__ == "__main__":
    calibrate()
