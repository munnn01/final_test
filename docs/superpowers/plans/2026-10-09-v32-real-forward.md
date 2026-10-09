# V32-RF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan natively task-by-task. The user's implementation instruction follows the in-chat design; preserve the existing native execution method and one final fresh review.

**Goal:** Ship a provenance-checked real-forward CNN experiment and public AR/OD Kaggle jobs with diagnostics that isolate proxy and selector effects.

**Architecture:** New `adaptive_vcm/v32rf` modules reuse the residual CNN, fixed virtual codec, support construction and paid receiver. Strict historical checkpoint loading uses an immutable aliased runtime. Optional codec/evaluation hooks preserve all existing defaults; forced arms are explicit diagnostics.

**Tech Stack:** PyTorch, torchvision frozen teachers, FFmpeg x264/x265, NumPy/SciPy, Kaggle official API, pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-v32-real-forward.md`

## Global Constraints

Warm origin28cc97db6708c87346c025107a8a7ea485c21163; seed303231; FIT48/CAL12/PROXYVAL4; two epochs×480 steps/arm; width12; Adam1e-4; weights task1/rate.01/MSE10/feature.1; LAST; primary medium GOP16 AR/GOP1 OD, no Bframes/scenecut; fast transfer TEST8. Public authorized; no ongoing cloud monitoring. Historical loaders and default evaluation retain their contracts.

## Review Focus

1. Proxy input/output dtype and uint8 rounding must exactly match real inference; forward substitution must not leak actual-codec gradients.
2. Teacher hooks must preserve BN/dropout state and be removed even on failure; source features must be detached.
3. Warm runtime must validate original source bytes and strict original checkpoint metadata; no train/eval source overlap including proxy validation.
4. Forced arms must score CNN output when guards reject it or OD foreground is unknown; missing forced names must fail explicitly.
5. Claimed codec conditions, schedules, public visibility and archive contents must match the executed notebook, including transfer and failure paths.

## Task 1: Codec bridge and diagnostic primitives

Files: create `adaptive_vcm/v32rf/{__init__,bridge,features,diagnostics,protocol}.py`, `configs/v32rf.json`; modify `adaptive_vcm/v32/codec.py`, `adaptive_vcm/v32vc/train.py` optional codec factory only; tests `tests/test_v32rf_bridge.py`.
Interfaces: `substitute(proxy, actual)`; `real_forward(video, codec, task)` returns detached RGB tensor/paid bpp/seconds; `feature_vectors(task,teachers,video)`; `feature_distance(features,targets)`; `settings()`, `split_cal(ids)`, `codec_factory(task,preset='medium')`; `selector_diagnostics(audits,cfg)`.
- [ ] Write tests for exact substituted values/Jacobian, invalid geometry/nonfinite input, actual H264/H265 gradient through CNN, frozen feature gradients, CAL split rejection, guard/rate rejection and explicit GOP flags.
- [ ] Run tests and observe missing module/function RED.
- [ ] Implement minimal primitives and optional codec factory, with no historical default change.
- [ ] Run GREEN plus affected codec/geometry/teacher tests; commit.

## Task 2: Warm provenance, paired finetuning and TRAIN proxy validation

Files: create `adaptive_vcm/v32rf/{warm,checkpoint,train}.py`; tests `tests/test_v32rf_training.py`.
Interfaces: `load_warm(path,runtime,task,device)` immutable strict old loader; `make_checkpoint`, `load_checkpoint`, `validate_pair`; CLI train task/root/annotations/warm checkpoint/runtime/out/device. Metadata follows spec; returned wrappers implement pixels/assert_source for evaluation.
- [ ] Write tests for warm source tampering, deterministic CAL12/PROXYVAL4, state/meta/recipe/history rejection and pair compatibility, paired optimizer step with same initial state/schedule, teacher-state preservation, held-out proxy diagnostics.
- [ ] Observe RED, then implement paired training/calibration/diagnostics and strict checkpoint artifacts.
- [ ] Run GREEN including real torchvision OD gradient and real codec numerical tests; commit.

## Task 3: Guarded and forced paired evaluation

Files: create `adaptive_vcm/v32rf/evaluate.py`; modify `adaptive_vcm/v32/{selection,evaluate}.py` explicit optional hooks; tests `tests/test_v32rf_evaluate.py`.
Interfaces: `run(args, extension=None, codec_factory=None)` uses extension models/families/preparer/forced names/manifest; `choose_stream(..., forced_names=None, unconditional_extra=False)`; `prepare_extension` yields candidates/families/forced names and measured seconds. Historical arguments default unchanged.
- [ ] Write RED tests for forced CNN despite rejected guards/unknown foreground, missing names, complete16-arm primary/14-arm transfer grids and packet receiver with paid retarget.
- [ ] Implement extensions with source overlap checks, selected packet/evaluator reuse and rejection diagnostics.
- [ ] Run new+inherited V32 evaluation/selection/transport tests GREEN; commit.

## Task 4: Immutable public Kaggle payload and execution documentation

Files: create `scripts/v32rf_kaggle.py`, `docs/V32_RF.md`, `docs/V32_RF_EXECUTION.md`; tests `tests/test_v32rf_payload.py`.
Interfaces: CLI with task/account/originalV30 checkpoint/runtime/warm checkpoint/runtime/out; exact source/warm ZIPs and manifests; known dataset mounts; public GPU notebook with training, primary DEV32/TEST32 and fast TEST8.
- [ ] Write RED tests for public metadata, exact checkpoint/runtime bytes, compiled cells, bound dataset discovery, complete/failure archive path and missing/invalid warm artifacts.
- [ ] Implement generator and runtime; run full affected GREEN suite; commit frozen source.
- [ ] Fresh whole-branch review, one RED→GREEN fix pass for important findings, preserve source hashes.
- [ ] Verify payloads/account quota, submit public AR/OD notebooks, save receipts and return links. Do not follow cloud execution; await user completion notice.

## Task 5: User-authorized Video Swin restoration experiment

Files: `adaptive_vcm/v32rf/{post,post_train,post_evaluate}.py`; `tests/test_v32rf_post.py`; extend public payload mode and docs.
- [ ] RED tests for frozen Swin backbone and input/output geometry, zero-initialized identity, head gradients without teacher updates, bounded restoration, checkpoint integrity/overlap, identical paired bitstreams, official classifier transform.
- [ ] GREEN implementation: frozen pretrained Swin feature backbone + trainable bounded RGB head; FIT48 real decoded RF-real inputs; LAST two epochs; all three AR evaluators; anchor/RF-real streams with/without restoration.
- [ ] Separate AR post job repeats the fixed paired RF fit; archive both TRAIN and held-out restoration evidence. Review all changes together before final cloud submission.
