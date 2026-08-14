# Deepfake Detection Pipeline for Live Streams (v2)

Improved end-to-end architecture for real-time deepfake detection on platforms like Google Meet or Zoom. This version adds audio-visual sync verification, a compute-saving cascade, adaptive score fusion, and tighter quality gating on top of the original spatial/frequency/temporal design.

**Reality check, unchanged from v1:** no detector is 100% flawless. Generative models evolve constantly and compression destroys fine detail. The goal here is production-grade robustness, not perfection — and honest per-generator reporting rather than one blended accuracy number.

---

## 0. Cascade Router (new)

Runs before any heavy model, on every sampled frame, for every participant tile.

- **Purpose:** avoid running the full dual-branch + temporal + audio stack on every face in every frame — the biggest cost driver in multi-participant calls.
- **Cheap first-pass model:** EfficientNet-B0 (or the QC gate features below) produces a fast "suspicion score." Only frames above a low threshold (e.g., > 0.15) get escalated to the full pipeline in Stages 3–6.
- **Effect:** in a typical call, most frames of most participants are confidently real and get filtered here for a fraction of the cost of the full pipeline — this is what makes real-time analysis feasible with more than 3–4 participants.

---

## 1. Stream Ingestion & Sampling

- **Video capture:** browser extension intercepting the WebRTC `<video>` element, or a virtual camera loopback via OBS/FFmpeg.
- **Audio capture (new):** tap the corresponding WebRTC audio track in parallel, time-stamped to the same clock as the video frames — required for Stage 4a below.
- **Sampling rate:** 2–5 FPS for video. Full 30 FPS processing wastes GPU budget without improving detection accuracy.
- **Buffer:** circular FIFO queue holding the last 10–15 sampled frames (video) and the matching ~3–5 second rolling audio window, keyed per participant.

---

## 2. Quality Control Gating

Filters unusable frames before any heavy inference runs.

- **Laplacian blur test:** drop frames below a variance threshold (OpenCV).
- **Brightness & pose check:** skip frames outside ±45° yaw/pitch or badly lit frames.
- **Minimum face size (new):** discard crops under roughly 80×80 px pre-resize — spatial features are unreliable below this regardless of blur/pose scores.
- **Pose-weighted confidence (new):** rather than a hard ±45° cutoff acting as a binary pass/fail, carry the pose angle forward as a confidence weight — artifacts are most visible in the 0–20° range, so a 40° frame that passes the gate should still contribute less to the final score than a near-frontal one.

---

## 3. Face Detection & Alignment

Unchanged from v1:

- Landmark detection via MediaPipe Face Mesh or RetinaFace.
- Eye alignment: rotate the crop so eyes sit on a horizontal plane.
- Margin padding: 20–30% border, since artifacts cluster at the chin, hairline, and neck.
- Resize to model input size (224×224 or 299×299).

---

## 4. Multi-Modal Inference (expanded)

### 4a. Audio-Visual Sync Branch (new)

This is the single highest-value addition. Real-time face-swap pipelines frequently fail to perfectly time-align mouth movement to cloned or pass-through audio, and this signal degrades much more slowly than visual artifacts as generator quality improves.

- Run a lightweight sync-confidence model (Wav2Lip-style sync network, or a distilled equivalent) over the aligned mouth region and the parallel audio window.
- Output: $P_{sync} \in [0, 1]$, plus a secondary flag for cases where **video is manipulated but audio is untouched** (or vice versa) — a case the original visual-only pipeline could miss entirely.
- Also run a lightweight voice-cloning artifact check on the audio track independently (spectral discontinuities typical of TTS/voice-conversion output), since a fake audio track paired with a real video feed is a separate attack the sync model alone won't catch.

### 4b. RGB Spatial Branch (unchanged)

- Fine-tuned EfficientNet-B4 or Swin Transformer for pixel-level artifacts (teeth misalignment, iris distortion, skin-blend seams).
- Output: $P_{spatial} \in [0, 1]$ and a 512-d embedding $\mathbf{e}_t$.

### 4c. Frequency Branch (caveat added)

- 2D FFT/DCT on the face crop to surface power-spectrum artifacts.
- **Known limitation:** this branch is strong against GAN-era fakes (DeepFaceLab, FaceSwap) but weaker against diffusion and Gaussian-splatting generators, which produce smoother spectra that evade classic frequency detectors. Its output should be treated as one input to an adaptive fusion step (Stage 6), not a fixed-weight component — see below.
- Output: $P_{freq} \in [0, 1]$.

---

## 5. Lightweight Temporal & Landmark Verification (unchanged)

- **Landmark jitter tracking:** frame-to-frame delta on eye corners and lip edges; high micro-jitter suggests a swapped/generated mask.
- **Sequential feature pooling:** last $N$ embeddings $(\mathbf{e}_{t-N}, \dots, \mathbf{e}_t)$ through a small GRU or 1D-CNN, returning $P_{temporal}$.

---

## 6. Decision Engine (adaptive fusion, revised)

### Learned fusion instead of fixed weights

Replace hand-tuned $w_1, w_2, w_3$ with a small learned fusion layer (logistic regression or shallow MLP) trained on held-out labeled data, taking all signals as input:

$$P_{frame} = f_\theta(P_{spatial},\ P_{freq},\ P_{temporal},\ P_{sync},\ Jitter,\ \text{pose\_confidence})$$

Why: fixed weights don't generalize across lighting, hardware, or generator type, and can't reflect that the frequency branch should contribute less as diffusion-based fakes become more common. A learned layer, recalibrated periodically as new generator types appear in training data, adapts automatically.

### Smoothing (unchanged mechanism)

$$S_t = \alpha P_{frame} + (1 - \alpha) S_{t-1}$$

### Thresholding — calibrated, not fixed

- Calibrate the trigger threshold via ROC analysis on held-out data rather than assuming 0.80 is correct for every deployment.
- Expose the false-positive/false-negative tradeoff explicitly: a "flag for human review" threshold can sit lower than a "auto-block/warn all participants" threshold, since the cost of wrongly accusing a real participant is much higher than the cost of a missed detection needing follow-up review.
- Trigger **"Potential Deepfake Detected"** only if $S_t$ exceeds the calibrated threshold consistently for 3+ seconds, as before.

---

## Training Strategy (updated)

| Strategy | Implementation | Why It Matters |
|---|---|---|
| Dataset diversity | FaceForensics++, Celeb-DF v2, WildDeepfake, plus modern diffusion/Gaussian-splatting datasets | Prevents overfitting to one generator family |
| Heavy augmentation | JPEG compression ($Q \in [30,90]$), WebM downscaling, motion blur, Gaussian noise | Forces reliance on deep artifact features, not resolution that streaming destroys |
| Audio-visual pair augmentation (new) | Train sync branch on both matched and deliberately desynced real/fake audio-video pairs, plus voice-cloned audio over real video | Teaches the model to catch mixed-modality attacks, not just fully-faked streams |
| Combined loss | $\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{Contrastive}$ | Pulls real embeddings together, pushes fake embeddings apart |
| Adversarial robustness testing (new) | Periodically evaluate against fakes optimized specifically to evade the current detector (adaptive/white-box attacks) | Detectors deployed at scale become a target; static eval numbers won't reflect this |
| Per-generator evaluation (new) | Report accuracy broken out by generator type (FaceForensics++ vs. Celeb-DF vs. diffusion-based), not one blended figure | A blended 95% can hide 99% on old GAN fakes and 70% on current diffusion methods — the number that predicts real-world performance |

---

## Deployment Notes (new)

- **Compute budget:** the cascade in Stage 0 is what makes >4-5 participant calls tractable; without it, running the full stack per-face per-frame scales linearly with participants and becomes a bottleneck fast.
- **Privacy/consent:** running biometric and voice analysis on meeting participants has disclosure obligations that vary by jurisdiction (e.g., BIPA-style biometric consent laws, GDPR for EU participants) — this needs a compliance review before deployment, not just an engineering one.
- **Human-in-the-loop:** given no detector is perfect, the recommended production behavior is flag-for-review rather than auto-block on a single stream's output, particularly at the lower calibrated threshold.
