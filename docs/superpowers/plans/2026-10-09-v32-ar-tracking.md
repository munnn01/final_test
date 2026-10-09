# V32 AR + Object Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans with the preserved native method. Implement task-by-task; one fresh whole-branch review before packaging/submission. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run only AR and object tracking through shared standard-codec/receiver infrastructure, with frozen Swin after decode and no standalone OD job.

**Architecture:** Add a small `v32at` adapter package around existing V32/RF/tracking components. Keep the old experiments intact and load RF-real with its immutable original runtime. Extend candidate/receiver hooks only where needed; hooks omitted preserve historical behavior.

**Tech Stack:** Python, NumPy, PyTorch/TorchVision, x264/x265 FFmpeg, existing Hungarian IoU tracker, pinned TrackEval, existing Kaggle REST/client helpers and lossless source packing.

**Spec:** `docs/superpowers/specs/2026-10-09-v32-ar-tracking.md`.

## Global Constraints

- Active tasks are exactly `ar` and `ot`; standalone `od` is rejected by the new CLI/payload configuration.
- Frozen torchvision `swin3d_t`, `Swin3D_T_Weights.KINETICS400_V1`, official transform; no new training or restoration head.
- x264/x265, medium, QP30/35/40/45/50, GOP16, bframes0, scenecut off, closed GOP.
- AR: Kinetics DEV32/TEST32, 16 frames, stride2, 128-square; no retarget candidate construction.
- OT: MOT20-01 first128 frames, 640-square, seqinfo FPS, temporal_subsample1, no padding, persistent per-arm state.
- Preserve RF checkpoint/source hashes from the spec; original strict loader only. No old provenance rewriting.
- MOT20 action predictions have no AR ground truth; AR accuracy/BD-rate there must be absent/null with an explicit reason.
- Exactly two public Kaggle jobs, existing account-pool authorization; no progress monitoring after accepted submission.

## Review Focus

- Identical selected packet across arms: cache model predictions, never share mutable tracker state.
- Final window shorter than16: retain true frame count and AR scope; do not invent video frames.
- Unknown person/AR source evidence: fall back conservatively rather than accepting an unaudited retarget.
- Runtime source/weights changed: fail original checkpoint validation rather than bypassing it.
- Codec window boundaries: retain IDs while rejecting out-of-order/duplicate frame ranges.

## Task 1: Task contract, immutable RF loader and candidate routing

**Files:** Create `adaptive_vcm/v32at/__init__.py`, `protocol.py`, `frozen_rf.py`; modify `adaptive_vcm/v32/selection.py`; create `tests/test_v32at_protocol.py` and `tests/test_v32at_frozen.py`.

**Interfaces:** `settings() -> dict` locks the spec values; `validate_task(task: str) -> str` accepts only ar/ot; `codec_factory(task: str)` supplies the locked standard codec. `load_frozen_ar(checkpoint: Path, runtime: Path, device='cpu')` returns the original learned preprocessor through an isolated import alias. Add keyword-only `enable_retarget=True` to `build_candidates`/`choose_stream`; false never calls `build_grids`, while default retains old behavior.

- [ ] Write failing tests: ar/ot accepted, od rejected; source/weight tamper rejected; original AR checkpoint loads unchanged; `enable_retarget=False` never calls grid construction and every candidate has empty grids.
- [ ] Run `python -m pytest tests/test_v32at_protocol.py tests/test_v32at_frozen.py -q`; confirm intended failures.
- [ ] Implement the contracts and immutable loader using the existing alias-import pattern; recover exact original bundle bytes from the returned AR notebook payload and verify the spec hashes.
- [ ] Run new tests plus `tests/test_v32_selection.py tests/test_v32rf_training.py`; verify historical defaults and overlap protection still pass.
- [ ] Commit the independently validated task with spec/plan.

## Task 2: Shared receiver and AR-only benchmark adapter

**Files:** Create `adaptive_vcm/v32at/receiver.py`, `ar.py`; add optional prediction/candidate-routing extension hooks to `adaptive_vcm/v32/evaluate.py`; create `tests/test_v32at_receiver.py`, `tests/test_v32at_ar.py`.

**Interfaces:** `SharedReceiver(action, detector=None)` operates on received decoded RGB. `predict(packet, *, sequence: str, arm: str, frame_start: int, tasks: tuple[str,...], tracker=None) -> dict` returns task results and shared packet/RGB hashes; never decodes/encodes itself. `ar.run(args)` reuses original Kinetics plan/source exclusion and evaluates guarded frozen RF-real, forced frozen RF-real and controls with Swin/R3D/R(2+1)D.

- [ ] Write failing tests for same packet/RGB at each requested head, frozen weights, cached detector predictions with separate tracker states, monotonic frame ranges and truthful short windows.
- [ ] Write AR integration tests with fixture teachers/codecs: retarget disabled in base and extra pools; paired grids complete; every backbone scores the same selected packet; no `od` task result.
- [ ] Run the new receiver/AR tests and confirm intended failures.
- [ ] Implement receiver and adapter; primary evaluator is Swin, not a replacement RGB preprocessor. Log per-head inference latency and actual frame scope.
- [ ] Run new tests plus `tests/test_v32_evaluate.py tests/test_v32rf_evaluate.py tests/test_v32rf_bridge.py`; commit when green.

## Task 3: Joint AR-output + tracking runner on MOT20

**Files:** Create `adaptive_vcm/v32at/tracking.py`; add optional extension hooks to `adaptive_vcm/v32/tracking.py`; create `tests/test_v32at_tracking.py`.

**Interfaces:** `JointTrackingTeacher(detector, action)` preserves the existing probe/commit semantics and adds action probabilities from the same RGB. `JointGuard(cfg)` records tracking details plus the existing AR semantic guard result and accepts only their conjunction. `tracking.run(args)` supplies joint factories/support/receiver hooks to the existing MOT20 runner; hooks omitted keep the old runner unchanged.

- [ ] Write failing tests: independent rejection by AR or association guard, missing source evidence fallback, union ROI support, same packet used by both receiver heads, state maintained across codec windows and isolated across arms.
- [ ] Test an unpadded short final window; test AR predictions saved but accuracy/AR BD-rate unset on MOT20; GT access remains after selection/prediction.
- [ ] Run `python -m pytest tests/test_v32at_tracking.py -q`; confirm intended failures.
- [ ] Implement optional factories/hooks with default compatibility. Use the shared receiver, correct source-coordinate mapping and existing reference tracker; codec factory supplies common GOP16 conditions.
- [ ] Run new tests plus `tests/test_v32_tracking.py tests/test_v32_codec.py tests/test_v32_selection.py`; run a small analytic actual-codec/TrackEval fixture and commit when green.

## Task 4: Two public Kaggle jobs and reviewable deployment evidence

**Files:** Create `scripts/v32at_kaggle.py`, `tests/test_v32at_payload.py`, `docs/V32_AT.md`, `docs/V32_AT_EXECUTION.md`.

**Interfaces:** `prepare(args)` accepts task ar/ot only, writes public immutable notebook/config/manifests and local size validation; no network. AR payload attaches exact original RF runtime/checkpoint and only evaluates. OT payload downloads pinned MOT20 prefix and TrackEval and runs the joint receiver. Reuse existing pool REST/client helpers for submission, never embed credentials.

- [ ] Write failing payload tests: only two task plans, no OD dataset/job, public flag, exact source/runtime/checkpoint hashes, serialized source below1000000 bytes, bounded safe extraction, original argument/FPS/frame scope preserved, exception archives saved.
- [ ] Run the payload tests and confirm intended failures.
- [ ] Implement prepare/runtime using the existing packed source transport and data helpers. Write local notebook artifacts for AR Kinetics DEV32/TEST32 and OT MOT20-01 first128/640; no training stage and no receiver RGB restoration stage.
- [ ] Run all V32AT tests plus relevant V32 codec/selection/tracking/RF/Swin/payload regressions. Perform one fresh whole-branch review and one fix pass for material findings; preserve evidence of tests and source/model checks.
- [ ] Commit source, prepare final immutable payloads, verify both complete local validation receipts, then submit exactly two public notebooks through available previously authorized pool accounts.
- [ ] Verify accepted handle/version once, attach URLs and source hash to execution ledger, and stop monitoring. Report queued/submitted status without inventing performance results.

## Handoff

The user approved implementation after viewing this plan and the pipeline image. All four tasks were implemented, the final72-test integrated suite passed, one fresh review and one tested fix pass completed, and both public Kaggle version1 submissions were accepted on2026-10-09. The durable execution ledger records individual test evidence, workflow deviations, immutable deployed source and acceptance receipts. No monitoring follows submission. Benchmark performance remains uncollected.
