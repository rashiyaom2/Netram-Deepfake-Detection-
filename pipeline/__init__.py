"""
Real-time deepfake detection pipeline.

Stages (all implemented):
  0. cascade_router.py         - cheap suspicion pre-filter          (Chunk 5)
  1. ingestion.py              - stream capture & sampling           (Chunk 2)
  2. quality_gate.py           - blur/pose/size filtering            (Chunk 3)
  3. face_align.py             - detection + alignment               (Chunk 4)
  4. branches/                 - spatial, frequency, av_sync         (Chunk 6)
     ├── spatial_branch.py     - EfficientNet-B4 RGB artifacts
     ├── frequency_branch.py   - FFT/DCT spectral analysis
     ├── av_sync_branch.py     - Wav2Lip sync + voice-clone detect
     └── branch_runner.py      - orchestrates all three branches
  5. temporal.py               - jitter + GRU sequence model         (Chunk 7)
  6. fusion.py                 - learned fusion + smoothing          (Chunk 8)
  7. orchestrator.py           - ties it all together                (Chunk 10)

Training (training/ package):                                        (Chunk 9)
  - datasets.py               - multi-dataset loaders
  - augmentations.py           - JPEG, WebM, blur, noise augmentations
  - losses.py                  - CE + Contrastive loss
  - train.py                   - main training loop

Each module is independently unit-testable and communicates via the
plain-dataclass message types defined in `types.py`.
"""
