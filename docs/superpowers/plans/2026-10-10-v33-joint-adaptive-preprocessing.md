# V33-AP Joint Adaptive Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and submit three public Kaggle experiments testing one adaptive preprocessing CNN for OD and AR before standard H.264/H.265.

**Architecture:** One bounded residual CNN serves single images and video clips, conditioned on QP/codec and source support. Actual codec reconstructions drive frozen OD/Swin losses; the fixed proxy supplies backward derivatives. Joint, AR-only and OD-only arms use identical initialization/source exposure and each produce one shared checkpoint, followed by paired AR and OD evaluations.

**Tech Stack:** Existing Python/PyTorch/torchvision, NumPy/SciPy/OpenCV/Pillow/pycocotools, FFmpeg x264/x265, existing bounded Kaggle source transport and credential-pool CLI helpers. No NeRV or additional task backbone dependency.

**Spec:** `docs/superpowers/specs/2026-10-10-v33-joint-adaptive-preprocessing-design.md`, approved by the user's request to implement after reviewing the linked spec.

## Global Constraints

- Preserve existing dedicated checkout `D:/STUDY/LAB/bao_1/adaptive_v32_build` and native execution method established in the previous campaign; leave historical V32 APIs/checkpoints/jobs intact.
- Campaign `v33ap`; exactly OD and whole-clip AR; no NeRV, tracking or receiver restoration network.
- One width12 CNN, zero-initialized RGB head, residual amplitude0.1, causal temporal contribution0.1 for T>1 with scene-cut reset; protected pixels remain exact.
- Seed303251, TRAIN64/task split FIT48/CAL12/PROXYVAL4; DEV16/TEST32/task, fast-transfer TEST8; no held-out weight updates, hyperparameter choice or threshold tuning.
- Two epochs, batch1, Adam lr1e-4, gradient clipping1.0,960 steps/epoch, LAST checkpoint; each FIT source uses both codecs and five QPs.
- Loss weights: AR/OD arm masks0/1, feature0.1, RGB distortion10, estimated paid rate0.01, temporal0.1. AR CE+0.25 source-to-received KL at temperature1; Kinetics OD pseudo loss scale0.25 with threshold0.5/top20/frame.
- Codec grid: h264/h265, QP30/35/40/45/50, medium primary/fast transfer,25fps, bframes0, scenecut off, closed GOP, AR GOP16/OD GOP1.
- Source geometry: AR16 frames/stride2/128-square and official Swin spatial transform; OD320-square letterbox with exact x/y inverse. Preserve and report historical short-video repetition; never make an AR clip from COCO.
- Swin3D-T K400 teacher/primary AR evaluator; R3D-18/R(2+1)D-18 independent evaluators. MobileNet FPN OD teacher and ResNet50 FPN OD evaluator. All task parameters/stats stay frozen.
- Adaptive guard: identity fallback, minimum savings0.01, AR KL slack0.1/anchor decision requirement, OD slack0.03/score threshold0.25. Clip candidates pass both AR and sampled-frame OD guards; COCO uses OD only.
- Retarget disabled for AR and training; OD-only raw/CNN retarget factor2/4 evaluation ablations use serialized inverse grids and full paid bytes.
- Public Kaggle/any configured account with quota already authorized. Check quota only to submit new payloads; do not inspect progress or cancel old jobs. Verify acceptance once, then stop monitoring.

## Review Focus

1. Kinetics frame with no confident object: background pseudo supervision must be finite; no sampled frame with reliable OD evidence must cause guarded identity fallback, not a false pass. Audit frames omitted for lack of evidence. Tests in Tasks2/4.
2. Same pixel bytes with different shape/frame indices or single-frame COCO: receiver cache must respect geometry and request scope, and must not fabricate AR results. Tests in Task4.
3. Duplicate source pixels across FIT/CAL/VAL/evaluation, including namespaced IDs: strict loading and evaluation must reject leakage and tampering. Tests in Tasks2/3.
4. Frozen detector temporarily entering training loss mode: normalization/dropout/RNG state and source targets must remain valid; both task losses must still differentiate the same CNN. Tests in Tasks2/3.
5. Half-prepared or ambiguously submitted Kaggle payload: archive/runtime failure evidence must survive, source must be immutable, secrets absent, and accepted jobs must not be pushed twice. Tests in Task5.

## File structure and shared interfaces

Create `adaptive_vcm/v33ap/{__init__,protocol,model,sources,tasks,calibration,train,checkpoint,receiver,selection,evaluate}.py`, `configs/v33ap.json`, `scripts/v33ap_kaggle.py`, the tests named below and `docs/V33_AP.md`/`docs/V33_AP_EXECUTION.md`. Reuse existing data/codec/transport/ROI/proxy/bridge/metric primitives; do not route new single-Swin guards through the historical V32 selector that requires two AR teachers.

Protocol produces a validated settings dictionary and namespaced source IDs. Sources produce records consumed by task losses and training. Checkpoint exposes one learned model consumed by evaluation. Receiver/selection consume only reconstructed packet RGB and label-free predictions; evaluation owns GT scoring. The Kaggle builder owns source packing and orchestration, not model/training policy.

Local tests run from the checkout with `D:/STUDY/AI/envs/ten_env/python.exe -m pytest`. Create workspace-local `outputs/temp`, set TEMP/TMP there and use a unique pytest `--basetemp` per task to avoid the previously observed Windows host-temp permission failure. Tests inject small teacher fixtures and use FFmpeg for actual-codec checks; no local pretrained downloads or GPU are required. CPU smoke cannot establish pretrained performance or GPU fit.

## Task 1: Shared CNN and locked protocol

**Files:** Create `adaptive_vcm/v33ap/__init__.py`, `protocol.py`, `model.py`, `configs/v33ap.json`; test `tests/test_v33ap_model.py`, `tests/test_v33ap_protocol.py`.

**Interfaces:**
- `settings(path=None) -> dict` validates the locked campaign recipe; `arm_weights(arm: str) -> tuple[float,float]` accepts only `joint/ar_only/od_only`.
- `source_id(task: str, raw_id: str) -> str`; `split_train(ids: list[str], task: str) -> tuple[list[str],list[str],list[str]]` returns FIT48/CAL12/VAL4 with original partition validation.
- `codec_factory(task: str, preset: str='medium') -> callable(codec, qp, base_preset='medium', fps=Fraction(25))` returns existing V32Codec with locked execution settings.
- `JointPreprocessor(width: int=12).forward(video, qp, codec, protection, motion, cuts) -> Tensor` consumes B3THW and returns the same shape without any task argument.

- [ ] Write failing tests `test_one_state_serves_image_and_clip`, `test_identity_and_exact_protection`, `test_temporal_cut_isolation`, `test_invalid_conditions`, `test_arm_masks_and_locked_grid`, `test_train_split_namespace_and_order`. Assert shapes T=1/3, identical parameter IDs across calls, initial exact identity, max edit<=0.100001, cut suffix equality, exact arm masks(1,1)/(1,0)/(0,1), and48/12/4 disjoint IDs.

```python
assert torch.equal(initial_image, source_image) and torch.equal(initial_clip, source_clip)
assert torch.equal(edited[protected], source[protected]) and max_edit <= 0.100001
assert parameter_ids_after_image == parameter_ids_after_clip
assert arm_weights('joint') == (1., 1.)
assert [len(fit), len(cal), len(val)] == [48, 12, 4]
```
- [ ] Run the two new test files and record intended failures before implementation.
- [ ] Implement the signatures above using the V32-VC architecture/validation and V32 codec primitive; bypass temporal at T=1 and reset causal padding per cut. Keep the old task-specific model untouched.
- [ ] Run new tests plus `tests/test_v32vc_model.py tests/test_v32_codec.py`; require PASS.
- [ ] Commit Task1 with a scoped `feat-v33ap-shared-model` commit; update the durable execution ledger with test evidence.

## Task 2: Sources and differentiable frozen task supervision

**Files:** Create `adaptive_vcm/v33ap/sources.py`, `tasks.py`; test `tests/test_v33ap_sources.py`, `tests/test_v33ap_tasks.py`.

**Interfaces:**
- `SourceRecord`: `id`, `task`, `item`, `rgb`, `geometry`, `frame_recipe`, `support`, `targets`, `source_features`, `source_predictions`; clean RGB is THWC uint8, targets/features are CPU detached data, `geometry=None` for AR.
- `prepare_sources(ar_root: Path, od_root: Path, annotations: Path, heads, cfg: dict) -> dict` returns records indexed by namespaced ID, domain FIT/CAL/VAL memberships, source/annotation/supervision hashes and availability metadata. `read_source(task, item, cfg) -> tuple[rgb, geometry, frame_recipe]` reproduces historical canonical pixels with explicit short-clip metadata.
- `TaskHeads(device: str, *, action=None, detector=None)` defaults to frozen Swin/MobileNet and accepts injected fixtures. `source_targets(record: SourceRecord) -> dict` caches detached predictions/targets/features, using labels only from FIT for training targets.
- `frame_indices(t: int) -> tuple[int,...]` returns distinct first/middle/last indices; `losses(heads, reconstruction: Tensor, record: SourceRecord, arm: str) -> dict` returns scalar `ar`, `od`, `feature` and active-task audit. Feature hooks operate on Swin `features[1]` and detector FPN, and are always removed.

- [ ] Write failing source tests for same canonical AR/OD pixels as existing loaders, explicit short AR repetition, COCO T=1, correct COCO effective x/y scaling, unknown K400 folder/missing file rejection, and duplicate pixel hashes across namespaces/splits.
- [ ] Write task tests `test_joint_losses_reach_same_cnn`, `test_arm_masks_and_missing_ar_label`, `test_empty_pseudo_boxes_are_background_supervision`, `test_targets_detached_and_teacher_state_restored`. Assert finite nonzero CNN gradients from separate AR/OD objectives on editable inputs, all teacher parameter grads absent, BN buffers/modes unchanged, correct3/1 distinct frame indices,0.5 threshold/top20, and no action call/label on COCO.

```python
assert torch.isfinite(ar_gradient).all() and ar_gradient.abs().sum() > 0
assert torch.isfinite(od_gradient).all() and od_gradient.abs().sum() > 0
assert all(p.grad is None for p in teacher.parameters())
assert action_calls_on_coco == 0 and torch.isfinite(background_loss)
assert frame_indices(1) == (0,) and len(set(frame_indices(16))) == 3
```
- [ ] Run the new files and confirm intended failures.
- [ ] Implement source preparation using existing plans/geometry/support primitives, with AR+predicted OD protection union and no GT support. Cache source Swin probabilities, pooled features and pseudo boxes; use source-to-received KL and detector loss mode with fixed BN/dropout and restored state. Fixed per-step RNG is supplied by training.
- [ ] Run new tests plus `tests/test_v32rf_bridge.py tests/test_v32vc_training.py`; require PASS and commit `feat-v33ap-joint-supervision`.

## Task 3: Paired training, proxy diagnostics and strict shared checkpoint

**Files:** Create `adaptive_vcm/v33ap/calibration.py`, `train.py`, `checkpoint.py`; test `tests/test_v33ap_training.py`, `tests/test_v33ap_checkpoint.py`.

**Interfaces:**
- `calibrate_domain(records: dict, cal_ids: list[str], task: str, cfg: dict, device: str, out: Path) -> dict` builds10 fixed entries from actual identity/bounded-blur codec trials only. `diagnose(model, records, val_ids, calibration, heads, cfg, out) -> dict` measures actual/proxy reconstruction, feature and rate mismatch without updates.
- `training_step(model, virtual, heads, record, codec, arm, cfg) -> dict` returns differentiable `loss`, component losses, actual/estimated bpp and actual codec seconds. `temporal_loss(y, x, cuts) -> Tensor` handles singleton/all-cut clips with finite zero.
- `train.run(args)` consumes `--arm joint|ar_only|od_only --ar-root --od-root --annotations --device --out`; prepares/calibrates both domains, deterministically executes1920 optimizer steps, writes history and `shared_last.pth`, then VAL diagnostics. Source targets remain fixed across arms.
- `make_checkpoint(model, metadata) -> dict`; `load_checkpoint(path: Path, device='cpu') -> LearnedJointPreprocessor`; `LearnedJointPreprocessor.pixels(rgb, support, codec) -> ndarray`; `assert_source(identifier, rgb)` rejects training/calibration/validation overlap by ID or pixels.

- [ ] Write failing tests `test_actual_forward_proxy_backward`, `test_all_arms_same_exposure_and_initialization`, `test_domain_loss_and_temporal_mask`, `test_calibration_train_only_and_no_retarget`, `test_diagnostics_do_not_update`, and strict checkpoint corruption/overlap tests. Use a tiny supplied recipe only through test dependency injection; production `settings()` stays locked.
- [ ] Use an actual x264/x26516×16 fixture: compare bridge forward pixels to independently decoded packet, verify finite CNN gradients and exact paid bpp. Test zero-weight task terms and feature masks without dropping shared rate/distortion exposure.

```python
assert np.array_equal(forward_rgb, independently_decoded_rgb)
assert actual_bpp == pytest.approx(8 * packet.coded_bytes / 256)
assert joint_order == ar_only_order == od_only_order
assert epoch_steps == 960 and len(history) == 2
assert state_after_diagnostics == state_before_diagnostics
```
- [ ] Run the new test files and record intended failures.
- [ ] Implement calibration with existing VirtualCodec/fit_rate/DCT primitives; training with real_forward/substitute and Task2 losses; fixed per-step RNG, norm clipping1, source no_grad targets and domain-specific fixed proxies. Save metadata/state/source/labels/model hashes and complete LAST history; load with weights_only and reject mismatched recipe/state/provenance.
- [ ] Run new tests plus `tests/test_v32rf_training.py tests/test_v32rf_bridge.py tests/test_v32vc_training.py`; require PASS and commit `feat-v33ap-real-codec-training`.

## Task 4: Shared receiver, dual guards and paired AR/OD evaluation

**Files:** Create `adaptive_vcm/v33ap/receiver.py`, `selection.py`, `evaluate.py`; test `tests/test_v33ap_receiver.py`, `tests/test_v33ap_evaluate.py`.

**Interfaces:**
- `SharedReceiver(action, detector).predict(packet, *, tasks: tuple[str,...], indices: tuple[int,...]|None=None) -> dict` returns packet/RGB hashes, unique coded bytes, frame-indexed OD predictions, optional clip AR probabilities and per-head latency. Default indices are Task2 sampled frames; callers may request all frames. Only received RGB is accepted; COCO calls OD only.
- `choose_stream(rgb, support, model, codec, heads, cfg, *, task: str, retarget: bool=False) -> dict` returns independently decoded identity/CNN packets, selected packet, forced packet, candidate/guard/cost audits. On clips all admissible candidates satisfy AR and all reliable sampled-frame OD guards; audit unreliable frames, reject CNN if none has reliable boxes or predictions are nonfinite.
- `evaluate.run(args)` consumes shared checkpoint, `--task ar|od`, roots/annotations, `--split dev|test --count --transfer --out`; uses DEV16/TEST32 or fast TEST8 and scores identity/forced/guarded plus OD raw/CNN-retarget ablations. Task/geometry/model mismatches fail before scoring. Save auxiliary OD on Kinetics with mAP=null.

- [ ] Write failing receiver tests for same decoded packet at both heads, unique byte accounting, cache separation for equal bytes/different geometry and requested indices, frozen heads, real unreplicated single-frame COCO and missing-task metrics=null. Assert latency covers all evaluated backbones and CUDA measurements synchronize when applicable.
- [ ] Write guard tests for independent AR and OD rejection, no-object source identity fallback, per-task slack values, insufficient1% savings, and source/GT fields absent from receiver/selector calls. Check fallback does not conceal forced-CNN results.
- [ ] Write actual-codec evaluation tests with fake teachers/metrics: complete2×5 grid; paid elementary/header/grid bytes and one shared packet hash; COCO inverse mapping; AR retarget off; exact raw/CNN retarget serialization; flat/insufficient-overlap BD=null.

```python
assert received['ar_packet_sha256'] == received['od_packet_sha256']
assert received['coded_bytes'] == packet.coded_bytes
assert selected_name_on_ar_rejection == selected_name_on_od_rejection == 'identity'
assert coco_result['ar_accuracy'] is None and kinetics_result['od_map'] is None
assert invalid_curve_result is None
```
- [ ] Run the new files and confirm intended failures.
- [ ] Implement using lower-level ar_guard/detection_distance, existing V32 packet/grid/inverse and metrics primitives. Score GT only after label-free selection and received predictions; report separate Swin/R3D/R(2+1)D and COCO mAP/AP50 results, actual/proxy diagnostics, timing and source scope.
- [ ] Run new tests plus `tests/test_v32_selection.py tests/test_v32_evaluate.py tests/test_v32rf_evaluate.py tests/test_v32at_receiver.py`; require PASS and commit `feat-v33ap-paired-evaluation`.

## Task 5: Public Kaggle payloads, final review and submission

**Files:** Create `scripts/v33ap_kaggle.py`, `tests/test_v33ap_payload.py`, `docs/V33_AP.md`, `docs/V33_AP_EXECUTION.md`; add no credential/account content to the repository.

**Interfaces:**
- `job_spec(arm: str, account: str, origin: dict) -> dict` validates the three arms and both datasets; names `v33ap-joint-s303251`, `v33ap-ar-only-s303251`, `v33ap-od-only-s303251`, public=True.
- `prepare(arm: str, account: str, out: Path) -> dict` builds but never submits source snapshot, notebook/metadata/job/origin and local validation receipts. Reuse bounded pack/safe_extract and serialized source byte limit1000000; keep the source manifest exact.
- Runtime mounts both datasets, records installed versions/model hashes, runs Task3 training and Task4 DEV16/TEST32/fast TEST8 for both tasks. It saves status/traceback/progress and evidence archive even on failure.
- Submission is an explicit separate orchestration step using existing `pool_environment`/`KaggleClient.call`, quota-only preflight and push receipts. Do not invoke `preflight_account`, since it polls all owner job statuses. Never embed/dump pool keys.

- [ ] Write failing payload tests for exactly three arms, both mounted datasets, public flag, same source/seed/scope across arms, finite bounded unpacking and size, no checkpoint credential input, runtime commands/exception archive, duplicate-submit prevention and ambiguous push receipt handling.

```python
assert set(jobs) == {'joint', 'ar_only', 'od_only'}
assert len({job['origin_commit'] for job in jobs.values()}) == 1
assert all(not metadata['is_private'] for metadata in payload_metadata)
assert all(size < 1_000_000 for size in serialized_sizes)
assert push_calls_for_previously_accepted_handle == 0
```
- [ ] Run the payload file and record intended failures.
- [ ] Implement immutable source packing/runtime and campaign documentation. A receipt already proving accepted handle/version prevents repush. A timeout with unclear acceptance is recorded as ambiguous, not retried blindly on another account; resolve only acceptance metadata or report it.
- [ ] Run all `tests/test_v33ap_*.py` and the Task1–4 regression set with workspace TEMP/TMP. Save exact commands, exit status, logs and local limitations.
- [ ] Request one fresh whole-branch review on the most capable available model, providing approved spec/plan, complete diff and evidence. Review is read-only; assess material findings, perform one fix pass and rerun impacted checks. No per-task subagents are used under the preserved native method.
- [ ] Commit final source and docs; prepare all payloads from the same clean committed snapshot. Compile notebook cells, validate config/source hashes, bounded extraction, secret absence and serialized size; save three complete local validation receipts.
- [ ] Query verified GPU quota of configured accounts only now; assign eligible slots/accounts to joint/ar_only/od_only without cancelling or monitoring prior jobs. Push the three authorized public notebooks, proceeding to another account only for a definitive rejection before acceptance. Preserve per-arm push/accepted metadata.
- [ ] Record accepted handles/versions/source hashes in execution ledger and report links. If any arm remains unaccepted, report it accurately with reason; never label the campaign fully submitted without all three receipts. Stop after acceptance, with no automation/status/log polling.

## Self-review and handoff

All spec sections map to Tasks1–5: shared model/protocol1; data/task masks/pseudo labels2; actual/proxy gradient/calibration/checkpoint3; shared receiver/selection/retarget/GT/BD/latency4; immutable public multi-account payloads/evidence/submission5. Review Focus tests are assigned above; signatures and namespace conventions are consistent across tasks. The plan does not claim a BD-rate gain or a GPU integration result from CPU tests.

The existing native execution method is preserved. This written plan is ready for user review; implementation starts after that review, followed by `superpowers:executing-plans`. Kaggle publishing authorization has already been given and does not need a separate consent request.
