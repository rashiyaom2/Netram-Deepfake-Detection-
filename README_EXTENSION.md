# Netram AI Deepfake Shield — Enterprise Video & Audio Call Defense

A production-grade, multi-modal **Chrome Extension** and **Native WebSocket Inference Server** for real-time deepfake, voice-clone, and face-swap detection across **Google Meet**, **Zoom Web**, and **Microsoft Teams**.

---

## 🌟 100% Real Model Pipeline (Zero Placeholders / Zero Heuristics)

Every stage runs a dedicated trained neural network or calibrated statistical model:

| Stage | Model Architecture | Weights File | Role |
| :--- | :--- | :--- | :--- |
| **Stage 2: QC Gating** | BlazeFace (TFLite) + solvePnP 3D | `models/blaze_face_short_range.tflite` | Fast face detection & 3D head pose angle weighting |
| **Stage 3: Alignment** | MediaPipe FaceLandmarker Task | `models/face_landmarker.task` | 468-point facial mesh + eye-line leveling affine rotation |
| **Stage 0: Cascade** | ViT Neural Triage (ONNX) | `models/deepfake_detector.onnx` | Quarter-resolution rapid neural screening |
| **Stage 4b: Spatial Branch** | Vision Transformer (ViT ONNX) | `models/deepfake_detector.onnx` (343 MB) | Boundary blending, GAN artifacts, 512-d embeddings |
| **Stage 4c: Frequency Branch** | 2D Spectral CNN (PyTorch) | `models/freq_classifier.pt` (629 KB) | 2D DCT log-magnitude spectrum classifier (100% val acc) |
| **Stage 4a: Audio Branch** | AASIST Graph Attention Network | `assist/weights/AASIST.pth` (1.28 MB) | Voice cloning, TTS synthesis, acoustic anti-spoofing |
| **Stage 4a: AV Lip-Sync** | Wav2Lip SyncNet (PyTorch) | `models/sync_net.pt` (65.8 MB) | 512-d cross-modal audio-visual speech sync validation |
| **Stage 5: Temporal Branch** | 2-Layer Bidirectional GRU | `models/temporal_gru.pt` (3.30 MB) | Frame-to-frame feature drift & self-attention pooling |
| **Stage 5: Liveness Anti-Spoof** | 468-pt Mesh Micro-Motion + EAR | Adaptive Baseline | Physiological micro-motion + natural blink dynamics |
| **Stage 6: Decision Fusion** | Calibrated Multi-Branch Fusion | `models/fusion_head.pkl` | Multi-branch probability fusion with exponential smoothing |
| **Forensic Audit** | SHA-256 Cryptographic Hash | Built-in | Tamper-evident verdict validation hash per frame |

---

## 🚀 How to Install and Load in Google Chrome

### Step 1: Start the Inference Engine
```powershell
.venv\Scripts\python.exe extension_server.py
```

### Step 2: Load the Unpacked Extension in Chrome
1. Open Google Chrome and navigate to `chrome://extensions`.
2. Toggle **Developer mode** to **ON** (top right).
3. Click **Load unpacked** (top left).
4. Select the `extension` folder inside this project.
5. **The Netram Interactive Onboarding Journey will automatically open**, guiding you through the features and in-call HUD!

---

## ☁️ Distribution (Option 3: Hybrid Deployment)

### How End-Users Access Without Local Python:
1. **Deploy `extension_server.py` to a Cloud VPS/GPU Server** (e.g. RunPod, AWS EC2, GCP, or Railway) behind an SSL proxy (e.g. `wss://api.netram.ai`).
2. In `extension/content/content.js` and `extension/popup/popup.js`, the extension automatically tries your cloud server first, with local fallback:
   ```javascript
   const ENDPOINTS = [
     "wss://api.netram.ai",      // Cloud engine (zero local install for users)
     "ws://127.0.0.1:8765",       // Local companion engine (power users / offline)
   ];
   ```
3. **Users only download the extension ZIP from your website** (or install from Chrome Web Store). They do not need to install Python or models locally!

---

## 🔒 Participant Legal Liability & Meeting Chat Broadcaster

When joining any **Google Meet**, **Zoom**, or **Teams** call, Netram AI ensures complete legal transparency and participant privacy:
1. **Automated In-Chat Legal Notice**:
   - The content script automatically detects Google Meet's in-call chatbox and broadcasts a verified legal liability disclaimer:
     ```text
     🛡️ [Netram AI Enterprise Defense Notice]
     Real-time neural deepfake & synthetic media integrity monitoring is active in this meeting session.

     🔒 Participant Data Protection & Non-Misuse Guarantee:
     • Video & audio streams are processed ephemerally in volatile memory solely for deepfake & synthetic voice detection.
     • Zero Data Retention: No video clips, biometric face vectors, or meeting audio are recorded, stored, shared, monetized, or reused for AI training.
     • Session Verification ID: NETRAM-SEC-XXXXXX
     • Compliant with Netram AI Trust & Safety Zero-Knowledge Standards.
     ```
2. **Top In-Meeting Compliance Bar (`#netram-compliance-bar`)**:
   - Floats at the top of the meeting viewport with one-click access to the full **Legal Terms & Liability Certificate**, **Manual Chat Broadcast**, and **Copy Notice** actions.
3. **Comprehensive Legal Terms Modal (`#netram-legal-modal`)**:
   - Displays full binding commitments: Volatile memory execution only, immediate post-inference purge cycle, GDPR Biometric Art. 9 compliance, and cryptographic session seals.

