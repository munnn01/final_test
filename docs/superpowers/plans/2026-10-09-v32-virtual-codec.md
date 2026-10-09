# V32-VC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Native implementation in this session, then one fresh whole-branch reviewer.

**Goal:** Simplify preprocessing training through a virtual codec and measure AR/OD transfer with optional ROI retargeting.

**Architecture:** A bounded residual CNN feeds a fixed differentiable DCT surrogate and frozen supervised task networks. Existing V32 transport/receiver/guards remain the real-codec evaluation infrastructure; new training/evaluation CLIs are separate.

**Tech Stack:** Existing Python/PyTorch/torchvision/OpenCV/FFmpeg; no new product dependency.

**Spec:** docs/superpowers/specs/2026-10-09-v32-virtual-codec.md

## Global Constraints

- V32/V30-B checkpoints and old CLI behavior preserved; task loss never uses DEV/TEST.
- Seed 303221, width 12, residual bound 0.1, 64 TRAIN = 48 FIT + 16 CAL.
- H264/H265, QP30/35/40/45/50, medium, AR16x128 stride2, OD320, FPS25.
- Four epochs, Adam1e-4, clip1.0, loss task + 0.01*bpp + 10*MSE, LAST only.
- Protection source-only; retarget2/4 with transmitted mapping; measured full bytes.

## Review Focus

- Frozen task models must still provide image gradients and preserve running statistics.
- Zero or constant coefficient variance must not yield nonfinite entropy/gradients.
- Retarget source/coded shapes and metadata must remain consistent for all scene segments.
- CAL/FIT IDs and pixel hashes must both be rejected from held-out evaluation.
- New families must not silently enter historical V32 arms or invalidate paired-grid audits.

### Task 1: CNN, virtual codec and differentiable grids

**Files:** adaptive_vcm/v32vc/{__init__,model,virtual,geometry}.py; tests/test_v32vc_model.py.
**Interfaces:** ResidualPreprocessor(task,width=12).forward(video,qp,codec,protection,motion,cuts); VirtualCodec.forward(video,qp,codec,cuts) -> (RGB,bpp); transform(video,grids,inverse=False) -> B3THW.

- [ ] Write tests for identity/bounds/protection, scene reset, DCT/color inversion, parameter-free entropy gradients and retarget gradient/geometry.
- [ ] Run `python -m pytest tests/test_v32vc_model.py -q`; expected missing-module failure.
- [ ] Implement the interfaces and finite/shape validation, uniform-noise quantization for training.
- [ ] Run the same tests; expected all pass; commit model task.

### Task 2: TRAIN-only calibration, supervised fitting and checkpoints

**Files:** adaptive_vcm/v32vc/{train,checkpoint}.py; configs/v32vc.json; tests/test_v32vc_training.py.
**Interfaces:** loss(task,teachers,reconstruction,label/targets); fit/run CLI; load_checkpoint(path,task,device); model.proposals(rgb,support,codec) and model.assert_source(id,rgb).

- [ ] Write failing tests for task input gradients with immutable teacher buffers, proper TRAIN split, invalid checkpoints, overlap rejection and finite calibration mappings.
- [ ] Verify RED; implement frozen supervised task adapters, CAL real-codec calibration, two FIT CNNs, strict metadata/state loading.
- [ ] Verify GREEN with training tests plus task1; commit trainer task.

### Task 3: Paired real-codec evaluation

**Files:** adaptive_vcm/v32vc/evaluate.py; minimal backward-compatible optional candidate/arm extensions in adaptive_vcm/v32/{selection,evaluate}.py; tests/test_v32vc_evaluate.py.
**Interfaces:** separate V32-VC CLI consuming strict new checkpoints and original frozen V30 baseline; explicit additional candidates/families without changing old defaults.

- [ ] Write tests for explicit family isolation, complete added paired grids, total bytes and fallback; verify RED.
- [ ] Implement new arms and checkpoint/overlap validation; verify GREEN with inherited V32 selection/evaluate/transport/roi/baseline tests.
- [ ] Run real H264/H265 plain/retarget receiver smoke; commit evaluation task.

### Task 4: Cloud screen and evidence

**Files:** scripts/v32vc_kaggle.py; tests/test_v32vc_payload.py; output/v32vc_kaggle artifacts and report; docs/V32_VC.md.

- [ ] Write payload tests for safe snapshot, known mount resolution, exact checkpoint embedding and no whole-input scan; verify RED.
- [ ] Implement reviewable private Kaggle payload reusing safe archive helpers; verify GREEN.
- [ ] Fresh whole-branch reviewer (most capable available model); fix important findings with RED/GREEN, run affected complete test set and commit frozen snapshot.
- [ ] Check Kaggle quota/inventory; run paired AR/OD TRAIN+DEV32 and locked TEST32 as budget permits; inspect complete artifacts and independently decode selected real packets.
- [ ] Report actual metrics, confidence limits, calibration limitations and remaining work; do not substitute synthetic checks for measured improvements.
