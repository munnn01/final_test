# V34 Adaptive Controller Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task by task, native execution, then one fresh whole-branch reviewer.

**Goal:** Implement and submit reproducible actual-codec-supervised adaptive preprocessing experiments for AR+OD and AR+OT.

**Architecture:** Source-only temporal region maps guide bounded spatial band filters. Actual codec measurements train a temporal action/risk/rate controller; calibrated selection operates without TEST ground truth. Standard decoded packets feed frozen task heads and ByteTrack.

**Tech Stack:** Python, NumPy/OpenCV, PyTorch/torchvision, FFmpeg, pycocotools, supervision 0.27.0, pinned TrackEval, Kaggle CLI.

**Spec:** `docs/superpowers/specs/2026-10-10-v34-adaptive-controller.md`

## Global Constraints

- One geometry-preserving 8-bit RGB output, one packet shared by tasks.
- No gradients through real codec or association; missing labels masked.
- TEST ground truth excluded from support, selection and training.
- Seeds303401/303402/303403; codecs H.264/H.265; QP30/35/40/45/50; medium,25fps,closedGOP16 (images GOP1).
- Source-only public payload, secrets in subprocess environment only.

## Review Focus

- Future frames and scene cuts: causal map history cannot cross a cut.
- Empty detections: finite support/risk, low-confidence OT detections preserved.
- Unavailable task labels: mask loss and record pseudo/GT scope.
- CAL/TEST leakage: freeze calibration before TEST, sequence-disjoint OT.
- Packed notebook/checkpoint serialization: exact hashes and weights_only loading.

### Task 1: Structured filtering and temporal controller

Files: `adaptive_vcm/v34/{__init__,protocol,preprocessing,model}.py`, `configs/v34.json`, `tests/test_v34_preprocessing.py`.
Produces `region_maps(rgb, predictions, temporal=True)`, `filter_clip(rgb,maps,action)`, `Controller.forward(x,maps,condition)`.
- [x] Write tests for identity, actual 8-bit edits, causal/cut behavior, bounded output and finite controller gradients.
- [x] Run tests RED; implement; run GREEN; commit.

### Task 2: Measurement store, constrained supervision and calibration

Files: `adaptive_vcm/v34/{learning,checkpoint}.py`, `tests/test_v34_learning.py`.
Consumes Task1 actions/controller. Produces Pareto targets, masked task damage losses, CAL uncertainty bounds and calibrated action selection; strict native checkpoint metadata.
- [x] Write tests for masked missing labels, feasible improvements at same bytes, identity fallback, calibration and metadata roundtrip.
- [x] Run RED; implement; run GREEN; commit.

### Task 3: Actual codec runner and shared AR/OD receiver

Files: `adaptive_vcm/v34/{tasks,sources,runner}.py`, `tests/test_v34_runner.py`.
Reuses existing measured codec and source geometry. Measures full candidate grid on TRAIN/CAL; trains three seeds; freezes CAL thresholds before DEV/TEST. Records per-sample prediction and packet evidence; evaluates all arms without GT selection.
- [x] Test source split boundaries, real FFmpeg candidate measurements, shared decode and cache/checkpoint provenance.
- [x] Run RED; implement; run GREEN; commit.

### Task 4: ByteTrack OT and sequence-disjoint experiment

Files: `adaptive_vcm/v34/tracking.py`, `tests/test_v34_tracking.py`.
Consumes Task1/2 and common runner utilities. Produces OT source records, source/GT provenance, actual sequence tracking outputs and pinned TrackEval metrics; no AR GT claims on MOT20.
- [x] Test low-score detections, tracker persistence across windows, sequence split, coordinate mapping, GT scoring exclusion from selection.
- [x] Run RED; implement; run GREEN; commit.

### Task 5: Kaggle jobs, full review and submission

Files: `scripts/v34_kaggle.py`, `tests/test_v34_payload.py`, `docs/V34.md`.
Produces immutable packed notebooks for `ar_od` and `ar_ot`, durable submission intents and receipts. Reuses source packing/auth isolation.
- [x] Test notebook compilation, exact job recipe/hash validation, no credential members and idempotent submit intent.
- [x] Run RED; implement; run GREEN; real smoke. Full suite attempted and stalled in a legacy Windows codec test; limitation recorded in V34_EXECUTION.md, no full-suite pass claimed.
- [x] Fresh whole-branch review; resolve important findings, rerun relevant checks.
- [x] Commit, prepare/verify source payload, check account quota and submit once per accepted job. Public version1 accepted for both jobs; links and limitations recorded. No completion monitoring.
