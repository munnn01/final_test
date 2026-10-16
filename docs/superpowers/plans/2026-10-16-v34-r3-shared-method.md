# V34 R3 Shared Method Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship one task-conditioned real-codec filtering framework with separate AR/OD/OT checkpoints and evaluations, improving the OD/OT training targets and selection while preserving frozen AR and all R2 joint runs.

**Architecture:** Reuse the unchanged nine-condition `adaptive_vcm.v34.model.Controller` for every task: CNN, GRU(16,24), MLP, 13×5 action/risk/rate outputs. New R3 adapters own OD COCO error components, OT preprocessed context metrics, prefix rollout, calibration and constrained selection; AR delegates to the exact frozen R1 behavior. Separate schema, outputs and source snapshots prevent mixing R1/R2/R3 artifacts.

**Tech Stack:** Python, PyTorch/torchvision, NumPy/SciPy, OpenCV, pycocotools, supervision==0.27.0, pinned TrackEval, FFmpeg x264/x265, pytest, existing redacting Kaggle client.

**Spec:** `docs/superpowers/specs/2026-10-16-v34-r3-od-ot-isolation-design.md` (approved shared-method clarification; design commit `7ccc3358439335c88a8febe4e5e0b98b13549b71`).

## Global Constraints

- One method/topology; separate checkpoints and results per `ar`, `od`, `ot`. Reject `ar_od_ot` in R3. Reject AR training/refit, including warm training.
- Preserve existing tracked R1/R2 code/config/scripts and all frozen AR model/config/calibration bytes. No module-global monkey-patching and no new OD/OT-only neural output heads.
- QP `{30,35,40,45}`, x264/x265, unchanged 13 action coefficients/order, preset/GOP/fps/sizes from the corresponding task recipe. Identity + three nonidentity deployed probes.
- OD surrogate: COCOeval matching, IoU `.50:.05:.95`, category-aware, area=all, maxDets=100, score cutoff `.10`; separate normalized FN/FP/localization increments. Final cohort COCO mAP remains unchanged.
- OT GT targets: official MOT-preprocessed causal context increments of `1-HOTA`, `1-DetA`, `1-AssA`. Source-reference metrics are explicitly separate. Continuous transactional tracker prefixes, no future frames and no per-GOP tracker reset.
- CAL never updates weights. TRAIN-only scales. Count independent images/sequences, not codec/QP/window rows. Fewer than two OT CAL sequences blocks promotion, not exploratory diagnostics; no formal coverage guarantee.
- Predeclared component grids: OD `{0,.01,.025,.05}`; OT `{0,.005,.01,.025}`. Equal-valued vectors within each group; select on DEV, never historical TEST.
- Keep raw QP points, cubic BD and separately labelled PCHIP sensitivity. Never force monotonicity. Historical TEST is exploratory; a confirmation label requires audited, previously unused cohorts.
- Three fixed seeds 303401/303402/303403; private Kaggle notebooks/datasets only. Accounts: `thuha1205`, `qktttttttttt`, `huolgggnuyen`, `baoancut`, `trmnguyn111`. Never print credentials from the existing pool.
- Do not cancel/restart R2 jobs or change the R2 worktree HEAD `c445ef3c690383e6a593c24043fca6d8a3f0cdaa` used by its dispatcher.

## Review Focus

1. Cross-task or cross-revision checkpoint loading where tensor shapes match: reject by task, metric-slot schema, recipe, model/prefix/input hashes (Task 1/4 tests).
2. Ignored/crowd or empty scenes causing misleading zero damage: use official COCO/MOT preprocessing and explicitly test undefined bins/empty predictions (Task 2 tests).
3. Resume from an earlier prefix or partial writes silently changing the stream: content-check completed rows, replay causal state and reject conflicting lineage (Task 3/6 tests).
4. Missing/nonfinite calibration, a single CAL sequence, or all rejected candidates: identity fallback and honest promotion status, never a hidden relaxed guard (Task 4/5 tests).
5. AR running through a new entrypoint with old weights but altered proposal/tie-break/bytes: compare actual decisions, packets and predictions with native AR, not just file hashes (Task 1/5/7 tests).

## Execution setup and verification conventions

Implementation needs a checkout separate from the R2 dispatcher's current worktree. Prefer a native managed worktree at the design/plan branch. The app task root is not itself a Git repository; if native setup cannot target the nested repository, document that result and use a Git worktree at `D:\STUDY\LAB\bao_1\v34_r3_build` on `v34-r3-shared-method`, based on the committed design/plan branch. Do not move or reset the R2 checkout. The user reviews this setup together with the execution choice.

Use a per-plan ledger under the executing skill's ignored workspace. Commit each completed task; append exact tests/results and any rulings with the same work call. Final review is fresh-context, then one verified fix pass. Existing user authorization covers pushing the finished verified code to `https://github.com/munnn01/final_test`; do not push a partial implementation as finished.

`PY` means `D:\STUDY\AI\envs\ten_env\python.exe`. Before tests, set `PYTHONPATH=D:\STUDY\LAB\bao_1\tmp\v34r2_deps`, `V34_TRACKEVAL=D:\STUDY\LAB\bao_1\tmp\v34r2_TrackEval`, and OMP/MKL/OPENBLAS threads to 2. Set TEMP/TMP to a fresh workspace directory for each suite. Invoke `PY -m pytest <files> -q --basetemp=<fresh workspace path>`; capture long output and read the result. Run commands from the R3 checkout. Never recursively delete an unchecked path.

Each task follows its RED → implementation → GREEN steps. An import failure is an acceptable first RED for a new module; later tests must fail for the specified behavior. A test passing without the change requires checking whether it exercises the new behavior.

### Task 1: Common model, version contracts and frozen AR adapter

**Files:** Create `adaptive_vcm/v34r3/{__init__,protocol,model,checkpoint,ar}.py`, `configs/v34_r3_qp4.json`, `tests/test_v34r3_protocol.py`, `tests/test_v34r3_ar.py`. No R1/R2 edits.

**Interfaces:**
- `protocol.settings(path=None) -> dict`: schema `adaptive-vcm-v34-r3-1`, revision `r3-qp4-shared-method`, common grid and per-task recipes. `recipe(cfg, task, stage) -> dict` rejects joint mode and non-evaluation AR stages.
- `model.Controller`: alias the unchanged R1 class; construct with `actions=13, conditions=9`. No replacement forward.
- `protocol.component_names(task) -> tuple[str,str,str]`: OD `fn,fp,localization`; OT `hota,deta,assa`; AR retains native risk semantics through its adapter. Keep task and component masks separate.
- `checkpoint.save(path, model, metadata)` and `load(path, *, task, recipe, task_models, device='cpu')`: JSON metadata/state hashes, weights_only loading, exact task/metric/recipe checks even with matching tensors.
- `ar.load_reference(manifest, device='cpu') -> (cfg, models)` and `ar.choose_packet(*args, **kwargs)`: verify pinned config and three native checkpoints, delegate to native AR loader/selector. No checkpoint conversion.
- `protocol.baseline_manifest(root, base_commit) -> dict` and `verify_baseline(root, manifest)`: hash all tracked baseline Python/config/project dependency files, not only the two V34 folders. Runtime validates a packaged manifest without needing a Git executable for hashing.

- [ ] **Step 1 — RED:** Write `test_controller_is_native_ar_topology`, asserting identical state keys/shapes, exact forward outputs on cloned weights/input, and logits/risk/rate shapes `(2,13)/(2,13,3)/(2,13)`. Write `test_task_and_component_masks_are_distinct`, `test_joint_and_ar_retraining_are_rejected`, and `test_shape_compatible_wrong_task_or_metric_checkpoint_is_rejected`. Assert changing a frozen recipe/calibration hash is rejected.
- [ ] **Step 2 — Run RED:** `PY -m pytest tests/test_v34r3_protocol.py tests/test_v34r3_ar.py -q`. Expected: new-module/contract failures; record the output.
- [ ] **Step 3 — Implement** the interfaces above. AR manifest contains native file hashes, seeds, settings and expected task-model hashes; the new config references the old AR config instead of rewriting it. Native delegation preserves proposal order and tie-break semantics. Add deterministic fixture comparison `test_ar_adapter_preserves_native_selection_and_decoded_prediction` using the same anchor/probe callbacks for both calls.
- [ ] **Step 4 — GREEN:** Run the two task files and existing `tests/test_v34_learning.py tests/test_v34_preprocessing.py`. Expected: all pass, AR tensors and decision audits match, no tracked baseline file change.
- [ ] **Step 5 — Commit:** `feat: add shared R3 contracts with frozen AR compatibility`; ledger tests and baseline manifest path.

### Task 2: Official OD/OT component targets and source-only scoring

**Files:** Create `adaptive_vcm/v34r3/{coco_targets,mot_targets,tasks}.py`, `tests/test_v34r3_targets.py`.

**Interfaces:**
- `coco_targets.components(predictions, annotations, image, geometry, *, score_cutoff=.10) -> np.ndarray(3)`: inverse-map detection boxes, use COCOeval per-image matching/ignore flags, compute normalized FN/FP and matched localization error averaged over the ten IoUs. Never compute the final mAP as average image AP.
- `mot_targets.read_context(path, start, stop) -> list[dict]`: retain IDs, classes, zero_marked, boxes for every annotation class, with explicit frame numbering.
- `mot_targets.components(package, tracks, annotations, geometry, *, benchmark, sequence) -> dict`: construct raw MOT context, call pinned `MOTChallenge2DBox.get_preprocessed_seq_data(...,'pedestrian')`, then official HOTA; return `hota,deta,assa` costs plus labelled diagnostic quality.
- `mot_targets.source_components(package, tracks, source_tracks) -> np.ndarray(3)`: same causal frames, source-reference HOTA-family consistency; no GT/distractor annotation access and no GT-accuracy label.
- `tasks.Heads` wraps frozen R2 task inference/support behavior without monkey-patching it. `prepare(source, task, source_prefix)` is label-free; `source_score(rgb, source, *, prefix, package=None) -> dict` returns costs/predictions/probe. Separate `gt_score(..., *, annotations, context, package)` exists only in the offline path. Source OD cost retains miss/extra/localization/confidence components; preserve R2 scalar costs as diagnostics.

- [ ] **Step 1 — RED:** Write `test_coco_components_fn_fp_wrong_class_and_localization` (perfect `[0,0,0]`, miss positive FN, extra positive FP, wrong class both); `test_coco_crowd_ignore_and_empty_scenes` (crowd-overlap prediction ignored, empty/empty zero, empty GT with false positive penalized). Write `test_mot_distractor_zero_marked_and_association_components` comparing an identical context against the pinned dataset API, with an ID switch reducing AssA and a matched MOT20 class-6 distractor removed. Use a sentinel whose annotation access raises in `test_source_score_never_reads_annotations`.
- [ ] **Step 2 — Run RED:** `PY -m pytest tests/test_v34r3_targets.py -q`. Expected: missing new target functions or explicit behavior failures; no skipped TrackEval checks in the release environment.
- [ ] **Step 3 — Implement** exact Task 2 signatures. Use official matching/preprocessing, not a renamed positives-only or custom matching substitute. Compute source and GT fields separately and mark all target arrays detached/native before caching. Empty undefined final COCO bins remain undefined through the unchanged evaluator.
- [ ] **Step 4 — GREEN:** Run Task 2 tests with pinned TrackEval and `tests/test_v34r2_learning.py`. Expected: all pass, source-only tests cannot access sentinel labels, context units stay [0,1].
- [ ] **Step 5 — Commit:** `feat: add inference-aligned OD and preprocessed OT targets`; ledger tested distractor/empty cases.

### Task 3: Causal measurement, cohort manifests and restartable prefix rollout

**Files:** Create `adaptive_vcm/v34r3/{sources,measurement}.py`, `tests/test_v34r3_measurement.py`. Reuse unchanged R2 `Support`, `Prefix`, source-copy whitelist and real-codec primitives.

**Interfaces:**
- `sources.load_groups(args, recipe) -> (groups, meta)` supports existing OD/MOT sources and a user-supplied pinned cohort manifest. Original MOT classes must remain available offline; no GT on heldout source objects. `validate_cohorts(groups, history) -> manifest` detects duplicate pixels/IDs/cameras and marks historical versus auditable confirmation status.
- `measurement.measure(record, source, maps, codec, heads, recipe, *, prefix=None, annotations=None, gt_context=None, package=None, prefix_variant='identity') -> row`: 13 measured packets, three GT components, separate measured source vector, legacy diagnostics, common starting-prefix hash. Does not commit arbitrary GT-selected winners.
- `measurement.measure_stage(groups, meta, cfg, out, origin, heads, task, *, variant, warm_models=None, parent=None, package=None, scope=None) -> cache`: identity or frozen-warm rollout. Row key `(source_id,codec,qp,prefix_variant)`; lineage includes source/model/map/codec/parent/warm-policy hashes.
- `measurement.validate_cache(cache)` and `resume(parent, expected)` content-check completed rows; replay only committed identity/warm actions to reconstruct continuous prefixes. Incomplete rows are remeasured, contradictory rows rejected. Sealed R2 costs cannot be relabelled R3 targets.

- [ ] **Step 1 — RED:** Write `test_candidates_share_frozen_prefix_and_probe_order_does_not_change_scores`, `test_resume_replays_selected_stream_and_matches_uninterrupted_packet_hashes`, `test_stale_parent_and_conflicting_row_are_rejected`, `test_sequence_camera_and_pixel_split_leakage_is_rejected`, and `test_only_train_cal_can_measure_gt_targets`. Across two 16-frame windows, assert committed generation increments once, tracker IDs persist, and no future frame enters source history. Include scene-cut/context-cap/re-entry behavior using the existing support and prefix boundaries.
- [ ] **Step 2 — Run RED:** `PY -m pytest tests/test_v34r3_measurement.py -q`. Expected: new measurement contract failures.
- [ ] **Step 3 — Implement** Task 3. Persist partial cache only after a complete condition row and a parent-state receipt, so interruption cannot mark half a bank complete. Identity stage commits identity; rollout stage commits a source-only action from a frozen TRAIN warm checkpoint. Never choose a rollout prefix from GT costs. Record warm hash and action chain. For OD there is only identity-prefix data and no temporal rollout duplication.
- [ ] **Step 4 — GREEN:** Run Task 3 tests and existing `tests/test_v34r2_support.py tests/test_v34r2_continuation.py`. Expected: interrupted/replayed and uninterrupted streams agree, baseline source unchanged.
- [ ] **Step 5 — Commit:** `feat: add restartable R3 measurement and causal rollout`; ledger source/prefix lineage.

### Task 4: Shared-controller learning, grouped calibration and risk-constrained selection

**Files:** Create `adaptive_vcm/v34r3/{learning,selection}.py`, extend R3 checkpoint metadata, create `tests/test_v34r3_learning.py tests/test_v34r3_selection.py`.

**Interfaces:**
- `learning.fit_scales(rows) -> np.ndarray(3)`: TRAIN-only 75th percentile absolute GT-component increments, floor `1e-3`.
- `learning.train(model, rows, inputs, recipe, *, device) -> (history,scales)`: native Controller; soft-target CE + 4×scaled masked risk regression + 4×rate MSE, Adam(lr=.001), gradient norm 5, 30 epochs by recipe. No extra head; final OT gives identity/rollout variants equal aggregate weight. Warm controller sees TRAIN identity data only.
- `learning.calibrate(rows,predictions,coverage=.9) -> dict`: CAL-only empirical group-max residuals per action/component, source-group count and prefix policy hash. Never count 8 conditions as 8 sources; explicitly no formal coverage guarantee. CAL does not change model parameters.
- `selection.shortlist(prediction, recipe, kind, *, fixed=None, rng=None) -> list[int]`: exactly three distinct nonidentity indices; CNN predicted logits with deterministic index ties; fixed/random controls equal budget. `learning.fixed_shortlist(rows, recipe) -> list[int]` uses TRAIN only.
- `selection.choose(candidates, encode, source_score, *, task, recipe, prediction, calibration, prefix=None) -> (action,packet,result,audit)`: all four real probes, component-wise predicted-risk bounds + measured source deltas + byte budget, then fewest bytes and smallest action index. Override identity risk/delta to zero and identity rate to one; commit exactly its selected probe. No annotations/GT parameter. `risk_gate=False` is the explicit source-only ablation and warm policy.

- [ ] **Step 1 — RED:** Write `test_train_scales_and_shortlist_never_fit_cal_or_test`, `test_ot_identity_rollout_have_equal_training_weight`, `test_cal_counts_sequences_and_keeps_weights_unchanged`, `test_gt_component_semantics_reject_shape_compatible_wrong_metadata`, `test_risk_gate_rejects_source_feasible_but_high_risk_candidate`, `test_bad_or_missing_calibration_falls_back_to_identity`, and `test_ties_probe_permutations_and_all_rejections_are_deterministic`. Assert a cheap candidate with one bad association component cannot pass using a good detection component.
- [ ] **Step 2 — Run RED:** `PY -m pytest tests/test_v34r3_learning.py tests/test_v34r3_selection.py -q`. Expected: missing learning/selection or behavioral failures.
- [ ] **Step 3 — Implement** interfaces and diagnostics (per-head errors, shortlist frequency/entropy, identity/fallback fraction). Use component-mask metadata, not task-mask positions, for OD/OT risk regression. Missing/nonfinite/mismatched CAL bounds fail closed, without relaxing limits. Mark OT CAL group count <2 exploratory and ineligible for promotion. Keep risk units distinct from full-cohort mAP/HOTA.
- [ ] **Step 4 — GREEN:** Run Task 4 files and Task 1 topology/checkpoint tests. Expected: all pass, Controller remains native 13×5, no frozen task-head parameter acquires a gradient.
- [ ] **Step 5 — Commit:** `feat: train common R3 controllers and gate measured candidates`; ledger calibration scope and fallback behavior.

### Task 5: Staged runner, separate evaluations and truthful curve reports

**Files:** Create `adaptive_vcm/v34r3/{runner,evaluation,metrics}.py`, extend `ar.py`, create `tests/test_v34r3_runner.py tests/test_v34r3_metrics.py`.

**Interfaces:**
- `runner.run_records(groups, meta, cfg, out, origin, heads, task, package=None, *, stage='all', parent=None, scope=None)`: stages `measure-identity`, `train-warm`, `measure-rollout`, `train-final`, `evaluate`, `all`. OD skips temporal rollout; AR accepts only `evaluate` through native frozen-reference adapter. Each stage writes immutable settings/parent/scope receipts and checkpoints with explicit role WARM/FINAL; final evaluation cannot accidentally use WARM.
- `evaluation.evaluate(..., *, splits=('dev','test'),codecs=None)` performs label-free packet selection before heldout annotation reads and independent receiver inference. Runs CNN/fixed/random plus identity on the same source/QP/codec and per-arm continuous OT state.
- `ar.evaluate_reference(groups, heads, models, cfg, meta, out, trackeval_path)` calls unchanged `v34.evaluation.evaluate(groups,heads,models,cfg,meta,out,trackeval_path)` after manifest validation. Reused reports have `execution='reused_frozen_reference'` and content hashes; fresh evaluation has a distinct receipt.
- `metrics.compare(anchor, adaptive, *, task) -> dict`: retain raw points and the historical cubic/PCHIP values. Invalid monotonicity/overlap/flat/duplicate quality clears acceptance; cubic interpolation overshoot, negative log-rate derivative in the integration interval, or >5 percentage-point cubic/PCHIP disagreement marks `interpolation_sensitive`/unreliable. The 5-point diagnostic threshold is predeclared engineering policy, not evidence of physical accuracy. AR native reporting remains untouched.
- `evaluation.promotion(report, cohort_manifest, calibration) -> dict`: require declared independent data status, OT >=2 CAL groups, per-QP gap >=−0.5 points, byte improvement and DEV support for the gate. Compare CNN/fixed/random honestly. Invalid BD does not become valid through reordering/filtering. Confirmation remains false without an audited confirmation cohort.

- [ ] **Step 1 — RED:** Write `test_od_full_grid_three_seed_fixture_smoke` (QP4×2 codecs×13 measured actions, all three final checkpoints), `test_ot_two_window_rollout_resume_and_official_metrics`, `test_heldout_gt_sentinel_is_read_only_after_selection_commit`, `test_ar_common_entrypoint_matches_native_actions_packets_predictions`, `test_nonmonotone_and_cubic_unstable_curves_clear_reliability`, `test_one_cal_sequence_or_identity_only_cannot_confirm_improvement`, and `test_final_evaluation_rejects_warm_checkpoint`.
- [ ] **Step 2 — Run RED:** `PY -m pytest tests/test_v34r3_runner.py tests/test_v34r3_metrics.py -q`. Expected: missing runner/metric APIs or explicit pipeline failures; annotate fixture-only quality evidence.
- [ ] **Step 3 — Implement** stage orchestration, CLI, complete stream scoring via unchanged COCO/TrackEval helpers and reports: bytes including header; quality by task; DetA/AssA/IDF1/MOTA/FN/FP/IDs; probes and all sender/guard/receiver work; used checkpoint hashes; CAL group counts; raw component violations. Grid tuning reads DEV only and freezes recipe before confirmation evaluation. Reject fresh-confirmation labels for known historical IDs/hashes.
- [ ] **Step 4 — GREEN:** Run Task 5 plus all R3 tests. Expected: complete real-codec fixture streams, exact AR compatibility, no skipped official target tests, invalid curves reported without invented gain.
- [ ] **Step 5 — Commit:** `feat: run and evaluate separate tasks through shared R3 framework`; ledger fixture and reporting limits.

### Task 6: Private Kaggle payloads and durable staged execution

**Files:** Create `scripts/v34r3_kaggle.py`, `scripts/v34r3_workflow.py`, `tests/test_v34r3_payload.py tests/test_v34r3_workflow.py`, `docs/V34_R3.md`.

**Interfaces:**
- `v34r3_kaggle.job_spec(task,account,origin,*,stage,parent=None,scope=None,ar_reference=None,cohort_manifest=None) -> dict`; `prepare(...,out)` only builds a source-only notebook; `validate_payload(directory)` verifies source/runtime/metadata hashes and reconstructs the expected notebook.
- AR payloads are evaluation-only with content-pinned native checkpoint/config parents or labelled verified-reference imports. OD/OT parents use R3 artifacts only. All metadata has `private=True` and `is_private=True`; dataset versions, evaluator commit and public data revisions are pinned.
- `v34r3_workflow.step(plan,state,client) -> state`: reuses the existing redacting `KaggleClient`, checks source HEAD/config hashes, allowed accounts and active jobs, submits ready stages at most once, verifies/downloads completed stage receipts, and records recoverable timeout parents. No public probes and no modification of the running R2 dispatcher.
- Workflow states: prepared → submitted/running → complete_verified or failed_attention. Warm and final dependencies are task-specific; evaluation shards are codec/split bounded. Never inherit partial target tables without prefix/source hashes. Stage outputs stay under a new R3 workflow root.

- [ ] **Step 1 — RED:** Write `test_private_payload_and_runtime_are_pinned`, `test_ar_training_and_joint_submission_are_rejected`, `test_foreign_task_parent_or_edited_notebook_is_rejected`, `test_dispatch_is_idempotent_and_does_not_cancel_r2`, `test_timeout_resumes_verified_rows_without_retraining_ar`, `test_wrong_or_incomplete_stage_receipt_is_not_complete`, and `test_state_and_logs_redact_credentials`. Compile every notebook cell and assert source <1,000,000 serialized bytes.
- [ ] **Step 2 — Run RED:** `PY -m pytest tests/test_v34r3_payload.py tests/test_v34r3_workflow.py -q`. Expected: missing new payload/workflow contracts.
- [ ] **Step 3 — Implement** narrowly scoped R3 payload/runtime and workflow using existing archive/private-parent primitives without editing old launchers. Existing MOT20 cohort can run only as exploratory. A manifest with independent additional CAL/confirmation sequences must be provenance-checked before promotion; do not silently split one sequence or treat MOT17 detector variants as distinct cameras. If only the historical data is available, run diagnostics and explicitly block confirmation/promotion rather than stop all technical evaluation.
- [ ] **Step 4 — GREEN:** Run Task 6 files and `tests/test_v34r2_payload.py tests/test_v34r2_continuation_payload.py tests/test_v34r2_workflow.py`. Expected: private, reproducible, idempotent R3 jobs and unchanged R2 behavior. Use simulated clients in tests; real submissions wait for Task 7 verification.
- [ ] **Step 5 — Commit:** `feat: package private restartable R3 task experiments`; ledger supported stages/accounts/data limits.

### Task 7: Regression, fresh review, publish and launch authorized separate tasks

**Files:** Extend `docs/V34_R3.md` with measured verification receipts; add `docs/V34_R3_REVIEW.md`. No model weights, credentials, downloaded cohorts or mutable run outputs in Git.

**Interfaces:** release receipt records baseline/source/config hashes, all task tests, native AR comparison, smoke scope, final review and commit. Operational outputs live under `output/v34_r3_shared_method_20261016` (a unique suffix if already populated). Keep code success separate from dataset-dependent quality results.

- [ ] **Step 1 — RED where needed:** Add integration tests for any missing specification requirement discovered in coverage review; watch them fail before fixing. Existing passing checks need no artificial RED. Verify every test named by Tasks 1–6 exists and was run.
- [ ] **Step 2 — Full verification:** Run all new R3 tests plus the existing V34 and V34r2 tests identified with `rg --files tests -g 'test_v34*.py'`. Expected: all pass, no skipped official evaluator tests, baseline hash manifest unchanged. Run real-codec full-grid OD smoke and continuous OT prefix smoke; verify AR entrypoint parity separately. Record exact commands/counts/environment and fixture limitations.
- [ ] **Step 3 — Fresh review:** Use requesting-code-review on the full branch against c445ef3c with the spec, plan, ledger and the five Review Focus cases. For Native execution, the executing-plans skill explicitly requires one fresh reviewer on the most capable available model; dispatch only that reviewer, not implementation agents. Re-grade findings, fix critical/important issues with RED→GREEN and one green regression run; ledger all rulings/deferred minors. Do not claim real-data gains from this review.
- [ ] **Step 4 — Publish verified code:** Commit release evidence, inspect remote state, and push finished code to `munnn01/final_test` without force. If another commit has landed, preserve it and integrate with verification rather than overwrite it. Verify remote SHA. Keep the R2 checkout HEAD and dispatcher state unchanged.
- [ ] **Step 5 — Prepare/launch runs:** Resolve native AR reference files and all source/cohort hashes first. Prepare private OD/OT stage jobs on the previously authorized accounts with available capacity, evaluate/import AR only as frozen reference, and submit through the new idempotent workflow. Preserve all existing jobs. Validate each returned handle and stage status; report actual running/completed/error state, pending quality evidence, and which data is only exploratory. Continuous staged execution may continue from its dedicated private state; no new recurring app automation is implied.

## Plan self-review and approval handoff

Coverage: spec §1/4/10 → Tasks 1/5/7; §2/3/6/7 → Tasks 2/3; §5/8 → Tasks 3/4/5; §9 → Tasks 3/5/6; §11 → Tasks 6/7. The five Review Focus risks each have named behavior tests. All produced signatures are defined in their owning task; later tasks consume those names. No R1/R2 file change is scheduled. An unavailable independent cohort limits scientific confirmation, not code/fixture execution.

Execution approach is pending the user's plan review. Recommended: **Native**, because the seven tasks share tightly connected target/cache/prefix/checkpoint contracts and there is one fresh whole-branch review before publish. Subagent-driven execution is available if the user prefers per-task implementer/reviewer gates. Implementation starts after this written plan and execution method are approved.
