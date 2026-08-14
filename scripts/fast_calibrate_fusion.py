"""
Fast Direct Logistic Regression Fusion Head Calibration & Export.
"""
import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Generate robust synthetic calibration dataset covering all threat vectors
np.random.seed(42)
features_list = []
labels_list = []

# Class 0: Clean, authentic video frames
# Features: [p_spatial, p_freq, p_temporal, p_sync, jitter, pose_confidence, p_liveness]
for _ in range(600):
    fv = np.array([
        np.random.uniform(0.02, 0.25),  # p_spatial: low
        np.random.uniform(0.05, 0.28),  # p_freq: low
        np.random.uniform(0.00, 0.12),  # p_temporal: low
        np.random.uniform(0.40, 0.60),  # p_sync: neutral/synced
        np.random.uniform(0.00, 0.08),  # jitter: low
        np.random.uniform(0.80, 1.00),  # pose_confidence: high
        np.random.uniform(0.00, 0.10),  # p_liveness: zero suspicion
    ], dtype=np.float32)
    features_list.append(fv)
    labels_list.append(0)

# Class 1a: Visual Deepfakes / Face-Swaps (high spatial, freq, jitter)
for _ in range(250):
    fv = np.array([
        np.random.uniform(0.65, 0.98),  # p_spatial: high
        np.random.uniform(0.40, 0.85),  # p_freq: elevated
        np.random.uniform(0.20, 0.70),  # p_temporal: elevated
        np.random.uniform(0.30, 0.70),  # p_sync
        np.random.uniform(0.12, 0.45),  # jitter: elevated
        np.random.uniform(0.60, 1.00),  # pose_confidence
        np.random.uniform(0.05, 0.40),  # p_liveness
    ], dtype=np.float32)
    features_list.append(fv)
    labels_list.append(1)

# Class 1b: Presentation Attacks / Static Photos / Phone Screen Replays (high liveness)
for _ in range(250):
    fv = np.array([
        np.random.uniform(0.20, 0.65),  # p_spatial: variable
        np.random.uniform(0.15, 0.60),  # p_freq: screen noise
        np.random.uniform(0.00, 0.25),  # p_temporal: low/rigid
        np.random.uniform(0.45, 0.55),  # p_sync: no audio / neutral
        np.random.uniform(0.00, 0.15),  # jitter
        np.random.uniform(0.70, 1.00),  # pose_confidence
        np.random.uniform(0.70, 1.00),  # p_liveness: HIGH (static presentation attack)
    ], dtype=np.float32)
    features_list.append(fv)
    labels_list.append(1)

# Class 1c: Audio-Visual Lip-Sync Desynchronization Attacks
for _ in range(150):
    fv = np.array([
        np.random.uniform(0.25, 0.75),  # p_spatial
        np.random.uniform(0.20, 0.60),  # p_freq
        np.random.uniform(0.10, 0.50),  # p_temporal
        np.random.uniform(0.75, 1.00),  # p_sync: HIGH mismatch
        np.random.uniform(0.05, 0.30),  # jitter
        np.random.uniform(0.70, 1.00),  # pose_confidence
        np.random.uniform(0.05, 0.35),  # p_liveness
    ], dtype=np.float32)
    features_list.append(fv)
    labels_list.append(1)

X = np.array(features_list)
y = np.array(labels_list)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        C=2.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42
    )),
])

pipeline.fit(X, y)

os.makedirs("models", exist_ok=True)
save_path = "models/fusion_head.pkl"
joblib.dump(pipeline, save_path)
print(f"[OK] Calibrated fusion head successfully saved to {save_path}")

# Verify clean, visual deepfake, and static photo attacks
clean_test = np.array([[0.08, 0.12, 0.03, 0.50, 0.02, 0.95, 0.02]], dtype=np.float32)
p_clean = pipeline.predict_proba(clean_test)[0, 1]

fake_test = np.array([[0.85, 0.65, 0.40, 0.50, 0.25, 0.85, 0.20]], dtype=np.float32)
p_fake = pipeline.predict_proba(fake_test)[0, 1]

photo_attack_test = np.array([[0.25, 0.15, 0.02, 0.50, 0.02, 0.95, 0.90]], dtype=np.float32)
p_photo = pipeline.predict_proba(photo_attack_test)[0, 1]

print(f"Clean test score:        {p_clean:.4f} (expected < 0.15)")
print(f"Visual Deepfake score:   {p_fake:.4f} (expected > 0.85)")
print(f"Photo Attack score:      {p_photo:.4f} (expected > 0.85)")
