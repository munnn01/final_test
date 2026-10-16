# V34 R2 continuation implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline, with one final independent review.

**Goal:** Continue the two timed-out R2 experiments using their saved measurements/checkpoints and sessions smaller than twelve hours.

**Architecture:** Keep the original scientific settings and action selection unchanged. Add execution-only codec/domain/split scopes to evaluation and a separate continuation entry point. Complete the joint measurement cache by replaying only identity prefixes for already measured tracking windows, train once, then evaluate complete sequences in separate jobs. A durable local workflow transports recovered/trained state through private datasets and dispatches downstream jobs after verifying completed artifacts.

**Spec:** User request to run the timed-out parts again, following the result audit and proposal to save/reload state across sessions. The five explicitly authorized accounts remain the allowed deployment destinations. AR/OD completed runs remain untouched.

## Global constraints

- Original QPs 30/35/40/45, two codecs, thirteen profiles, three seeds, source cohorts, frozen heads, selection constraints and DEV diagnostics remain unchanged.
- No temporal shards: every evaluation job starts at the beginning of the complete sequence, retaining continuous source/receiver tracker histories.
- Execution scope never mutates checkpoint settings or training/calibration data.
- Cache/checkpoint, source pixel, task-head, origin and inherited artifact checks precede reuse.
- No TEST GT enters selection. No changes to BD metrics in this continuation.
- Parent kernels are pinned; source/artifact bytes are checked. No credentials appear in submitted payloads.
- Use private continuation notebooks. The rejected public CPU probe is abandoned.
- Preserve durable submit intents; no blind retry of an ambiguous submission.

## Task 1: Scoped evaluation

- [x] Write and run a failing real-codec test comparing full and scoped evaluation packets, actions, continuous prefixes and official OT metrics.
- [x] Add keyword-only split/codec/domain execution scopes; use scoped settings only for report aggregation.
- [x] Validate invalid scope values and keep default behavior unchanged.

## Task 2: Resume measurements and training

- [x] Write failing tests for partial-cache integrity, duplicate/missing source conditions and incompatible inherited provenance.
- [x] Add continuation entry point: recover OT models without retraining; finish only missing joint measurement rows and regenerate deterministic controller inputs.
- [x] Restore causal identity target prefixes by re-encoding/scoring cached anchors, verifying packet and prefix hashes; never replay a candidate as a selected prefix.
- [x] Record inherited and executing provenance separately; write a stage-complete manifest after verified outputs.

## Task 3: Kaggle packaging and staged dispatch

- [x] Prepare scoped payloads with pinned kernel inputs; validate embedded source/artifact manifests and compile notebook cells.
- [x] Test stage readiness, idempotent receipts, failed/unknown dependency states and account allowlist behavior.
- [x] Run focused continuation tests and the existing R2 suite, then obtain one independent final review.
- [x] Commit/push the verified code to the previously authorized GitHub repository (initial continuation commit `1e10a1e9`; transport fix follows).
- [ ] Preflight the requested accounts, submit OT scopes and joint measurement continuation, verify accepted versions/status and start the durable downstream workflow.

## Review focus

Same source pixels with changed task heads; mixed cache origins; identity prefix replay after an occlusion/cut; codec-scoped reporting accidentally omitting conditions; downstream jobs starting before controller artifacts exist.

## Progress and rulings

- Root cause: OT used ~5.13h for TRAIN/CAL then ~45min per full DEV window across codecs/seeds/diagnostics; joint exhausted its session during CAL. Rerunning the monolithic entry point repeats the failure.
- Ruling: reevaluate each complete OT sequence from frame one using the saved trained models, because the interrupted job did not persist mutable tracker state or the full DEV diagnostic table. This repeats evaluation work, not TRAIN/CAL measurements or training, and preserves the original experiment.
- Ruling: keep optional execution scopes separate from the immutable experiment configuration; do not pretend a subset is a complete benchmark.
- Verification: execution-scope RED 5 failures -> GREEN 5; resume RED 3 failures -> GREEN 8 total; payload RED 6 -> GREEN 6; dispatch RED 3 -> GREEN 3; additional cohort/privacy RED 2 -> expanded suite GREEN 54/54 (128.40s).
- Final independent review: approved with one recovery-path finding. The reviewer graded the missing `version` in a pre-push intent as Minor; the implementer regraded it Important because interrupted submission recovery is a central requirement. A targeted test reproduced the restart `KeyError`; one fix pass normalizes legacy intents to `version=None` and rejects a receipt for another handle. Targeted GREEN: 1/1. No other review findings; no additional review cycle.
- Final post-fix verification: all 55 R2 tests passed in 122.36s; `git diff --check` clean; R1 implementation/configuration/launcher unchanged from the published baseline.
- Live deployment finding: Kaggle's CLI accepted all five notebooks but warned that the cancelled parent kernels were not valid input sources. Four OT jobs failed immediately at inherited-state validation; no model evaluation ran. The first workflow stopped on those terminal errors. Preserve those receipts and relaunch from a new workflow directory using verified recovered files in private datasets owned by each execution account. Use the same dataset transport for trained joint artifacts.
- Deployment regression RED: rejected-input warning falsely classified as submitted, and recovered-state dataset transport unavailable (2 failures). GREEN: 6/6 workflow tests after the targeted fix. This is a live integration repair, not a new scientific method or another review cycle.
- Transport repair verification: all 57 R2 tests passed in 131.35s. Only workflow transport/receipt handling changed; the tested scientific implementation remains the `1e10a1e9` implementation.
