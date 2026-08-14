# Real-Time Deepfake Detection Pipeline (v2)

Production-oriented, per-participant, real-time deepfake detector for
video-call platforms (Google Meet / Zoom / OBS virtual camera feeds).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

GPU strongly recommended for Stages 4-5 (spatial/frequency/sync branches).
The cascade router (Stage 0) is deliberately cheap enough to run on CPU
if needed.

## Build status

- [x] Chunk 1 — Scaffold, config, shared types
- [x] Chunk 2 — Stream ingestion & sampling
- [x] Chunk 3 — Quality control gating
- [x] Chunk 4 — Face detection & alignment
- [x] Chunk 5 — Cascade router
- [x] Chunk 6 — Multi-modal inference branches
- [x] Chunk 7 — Temporal & landmark verification
- [x] Chunk 8 — Decision engine (fusion)
- [x] Chunk 9 — Training pipeline
- [x] Chunk 10 — Deployment orchestrator

## Architecture

See `config.py` for every tunable threshold, and `pipeline/types.py` for
the message contracts each stage produces/consumes. Data flows:

```
RawFrame
  -> [Stage 2: quality_gate]      -> QualityResult (attached to frame)
  -> [Stage 0: cascade_router]    -> CascadeResult (drop or escalate)
  -> [Stage 3: face_align]        -> AlignedFace
  -> [Stage 4: branches/*]        -> BranchScores
  -> [Stage 5: temporal]          -> TemporalResult
  -> [Stage 6: fusion]            -> FrameDecision
```

Each arrow is a plain-dataclass boundary — every stage can be unit
tested in isolation with synthetic `RawFrame`/`AlignedFace` objects,
no live video call required.
