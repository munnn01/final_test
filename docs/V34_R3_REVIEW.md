# R3 release review and verification

Scope: one native V34 controller architecture, task-specific OD/OT training targets, checkpoints and evaluation; frozen AR reference; unchanged R2 joint pipeline. No real-data R3 quality gain is established by software tests.

## Single independent review

The final fresh-context reviewer (`r3_final_review`, GPT-6 Astra) reviewed the branch against `c445ef3c690383e6a593c24043fca6d8a3f0cdaa`, the approved design, implementation plan and decision ledger. Verdict: **with fixes**; four Important findings, no Critical findings. The implementer retained all four grades and performed one test-driven fix pass. There is no second reviewer pass.

| Important finding | Correction and regression |
| --- | --- |
| Identity measurement timeout could prepare a new payload while its recovery dataset was still processing. Existing prepared payloads could name a stale parent. | Wait for both dependency and recovery parents, compare the complete expected job including task, stage, parent, scope and source before submission. Processing→ready regression verifies no premature preparation/submission and rejects a stale payload. |
| Shape/recipe-compatible checkpoints did not enforce expected experiment origin, measurement caches or input lineage; FINAL calibration bindings were insufficiently checked. | Persist training lineage and input artifacts through DEV/TEST parents; verify expected execution origin, actual input bytes, current TRAIN/CAL source/annotation manifests, checkpoint cache/input/origin fields, and FINAL calibration state/cache/prefix bindings. Wrong and missing lineage and tampered calibration regressions reject loading. |
| OT offline risk diagnostics subtracted the always-identity stream after adaptive prefixes diverged. | Capture the identity probe on each arm's current frozen prefix during source-only selection; score it after decisions commit. Use this delta for risk errors and violations. Retain always-identity stream delta under a distinct field. A two-GOP real transactional-prefix regression reproduces the former doubling error. |
| Local MOT17/additional independent CAL sequences were blocked by the old four-sequence MOT20 planner, and detector suffixes were lost during final evaluation. | Add an R3-only local MOT17/MOT20 planner; canonicalize camera groups only for leakage/grouping, preserve original sequence names for official TrackEval files. Load two independent MOT17 CAL cameras and score a separate MOT17 heldout sequence with the pinned evaluator; reject detector variants across splits. R2 planner stays unchanged. |

One fixture initially used `ids` instead of the tracker's actual `track_ids`/`frame_number` contract; the fixture was corrected without changing production serialization.

## Evidence and limitations

The final review fix regression passed **30 tests in 220.07 seconds**: `pytest tests/test_v34r3_release.py tests/test_v34r3_protocol.py tests/test_v34r3_workflow.py tests/test_v34r3_runner.py -q`. The initial review regressions (parent waiting, absent lineage interface, MOT17 loader and own-prefix OT damage) were observed failing before their fixes. This green run includes real x264/x265 QP4 OD smoke with all three controller seeds, continuous OT warm/rollout/replay and official metrics, heldout-label sentinels, and native AR action/packet/decoded prediction parity. Fixtures establish code behavior only. The first full-suite attempt was stopped when the Important review findings arrived; it is not counted as a passing suite.

A subsequent full-suite attempt exposed a harness error: placing pytest temporary directories inside the checkout let an existing V31 "no Git repository" fixture discover the parent repository. The error was reproduced separately, then all **50 V31 protocol tests passed** with temporary storage outside Git. No V31/AR/R2 code or tests were changed. That interrupted full-suite run is also excluded from passing counts.

The next serial run completed with 1,074 passed and 16 TrackEval import-path failures. Older V32 tests use `output/v32_tests/trackeval_official`; the harness had pointed V34 tests to `tmp/v34r2_TrackEval`. Both checkouts were clean at the identical official commit, and the loader correctly rejected mixing cached modules from different roots. The conflict was reproduced directly. With the canonical former path, a mixed V32/AROT/V34/R3 MOT17 regression passed **4 tests in 11.58 seconds**. Complete verification is repeated across four isolated local pytest processes using that same canonical evaluator and separate temporary roots; this changes verification execution only.

**Final complete verification: 1,090 passed, 0 failures, 0 skipped, 870.61 seconds wall time**, across four isolated local pytest processes (298, 278, 241 and 273 tests). Every collected test file ran exactly once; JUnit counts match the 1,090-test collection manifest. All official evaluator cases ran. The four review findings are resolved by the regression tests and this green suite. Logs, JUnit XML, collection and environment receipts are retained under `output/v34_r3_shared_method_20261016/verification` in the local parent workspace; mutable experiment outputs are excluded from Git.

Environment: Windows, Python `D:/STUDY/AI/envs/ten_env/python.exe`; dependency overlay `D:/STUDY/LAB/bao_1/tmp/v34r2_deps`; TrackEval commit `12c8791b303e0a0b50f753af204249e622d0281a`.

AR was imported from `dngbolm/v34-ar-r1-qp4-cef864eb` as `reused_frozen_reference`, with its original six reports. It was neither retrained nor presented as fresh R3 heldout evaluation. Config SHA256: `221115b5cb53c078e4c982b922abecdfe84d9343ec2128f8eca605eed1e54f1a`. Checkpoint SHA256 values, in seed order 303401/303402/303403:

- `6f76f8a14ca0e2a4f8812f32ca18597f80dd899cf6fc6ed6cc5e7602787b8d7e`
- `990c896c5cf8354360cd90adb4792a153510bdcd9f591e44a160c9a32f12c840`
- `c4e142d50f3ba9e9d2cd4125e2f64c4e87ba2d110064457c601978ef60109433`

The baseline source manifest pins 277 files; manifest SHA256 `0900d29a0d7c2f4fcb517d4a8565c9242c3b524d1b45b7761ede07a7a47c2e8a`. The separate R2 checkout remains at `c445ef3c690383e6a593c24043fca6d8a3f0cdaa`. Its jobs completed naturally before R3 launch, without cancellation or restart.

Default historical cohorts remain exploratory. OT has one CAL sequence and cannot pass the independent-confirmation promotion gate. Additional local MOT17 mounts are supported; the automated Kaggle downloader remains pinned to MOT20. A supplied history manifest is an audit input, not external proof of unused data.

## Rulings and costs if wrong

1. Use an approved sibling CLI worktree after the app worktree tool rejected the non-repository app root. Cost if wrong: source drift could interrupt or invalidate R2; verify its checkout separately.
2. Use seed 303401 as one frozen warm rollout policy shared by final seeds. Cost if wrong: policy dependence remains; seeds describe optimization variation, not independent experiments.
3. Bind CAL to immutable recipe and model state; record DEV policy overrides separately. Cost if wrong: bounds could be applied to the wrong model or policy; load/selection checks must fail closed.
4. Memoize frozen inference on identical decoded pixels with separate sender/receiver caches and arm-owned trackers. Cost if wrong: nondeterminism could change decisions, or representative timing could be mistaken for wall time; labels and deterministic evaluation are required.
5. Lock one DEV policy across both codecs and seeds; allow codec-sharded TEST under that exact lock. Cost if wrong: codec-only DEV selection would bias cross-codec comparisons.
6. Transfer parent trees in private hash-pinned bounded archives; automatically continue only confirmed measurement timeouts making verified row progress. Cost if wrong: a parent may be unavailable or require manual inspection; never silently rerun AR or blindly resubmit ambiguous writes.
7. Pin source geometry/metadata, current annotations/maps, decoded replay pixels and native AR artifact bytes. Cost if wrong: stale caches are rejected and must be measured again; no live production R3 cache migration is implied.
8. Reviewer declined real-data OD/OT rate–quality judgment. Ruling: software completion authorizes the prescribed experiment, not an improvement claim. Cost if wrong: mistaken scientific conclusions; require measured QP curves, controls and promotion evidence.
9. Reviewer declined live Kaggle quota/readiness/runtime judgment. Ruling: independently inspect server quota/status and persist submission evidence; report running/complete/error literally. Cost if wrong: GPU quota could be spent on an incomplete stage; verified partial recovery and no blind retry limit this cost.
10. Reviewer declined component-wise superiority or causal contribution of the new targets/rollout. Ruling: retain the unchanged R2 comparison and prescribed fixed/random/source-only/identity-prefix controls; do not claim isolated causal benefit without the relevant ablation. Cost if wrong: attributed gains may come from another component or tuning.
11. Place verification temporary storage outside the Git checkout so "no repository" fixtures are independent of the enclosing worktree. Cost if wrong: environmental false failures can hide the actual regression state; preserve the RED/GREEN harness receipts and rerun the complete suite.
12. Standardize verification on the clean official TrackEval checkout used by existing V32 tests, at the same required commit. Keep the loader's module-origin check and all AR/R2 code intact. Cost if wrong: mixing evaluator implementations could invalidate metrics; a mixed-version regression and full collected suite must pass under the canonical path.

## Deferred minors

- DEV repeats source-only policies for each GT limit even though that limit is ignored when `risk_gate=False`. Deduplicating would save evaluation work; it is deferred to preserve the agreed single Important-fix pass.
