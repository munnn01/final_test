# V32-AT execution ledger

Plan: docs/superpowers/plans/2026-10-09-v32-ar-tracking.md
Spec: docs/superpowers/specs/2026-10-09-v32-ar-tracking.md
Base: fc36e38; existing dedicated checkout D:/STUDY/LAB/bao_1/adaptive_v32_build.

User approved implementation after viewing the AR + tracking image and written plan. Preserve native execution and public Kaggle/any available configured account authorization. Do not monitor after submission.

Ruling: Reuse the existing dedicated non-main experimental checkout, consistent with the previous V32 execution preference; no additional worktree or implementation approval is needed. Cost if wrong: relocate the four commits into another isolated checkout.
Ruling: Use native Windows ledger/test commands because the prior Git Bash bookkeeping helper failed on this host. This file is the durable execution record. Cost if wrong: translate the ledger into helper-managed bookkeeping later.
Pre-flight: protocol/frozen loader and candidate switch feed AR; shared receiver feeds AR and joint tracking; joint factories preserve old tracking defaults; both adapters feed the two-task payload builder. No interface conflicts found.

Task1 RED: four intended failures (new package/switch missing), one historical-default test passed. Initial tmp_path setup error was corrected by creating the workspace-local outputs parent; rerun had no setup errors.
Runtime recovered from the original packed AR notebook without executing notebook code; bundle/files hashes match the pinned immutable origin.
Task1 validation: original checkpoint loads; source and checkpoint tamper rejected; default retarget behavior retained. Windows codec tests use workspace TEMP/TMP; default host temp chmod was denied. Historical injected candidate factories required the old call shape when the new flag is omitted; corrected without changing the opt-out behavior.
Task2 validation: shared decoded RGB, frozen heads, independent per-arm tracker states, monotonic ranges and short windows verified. AR adapter actual-codec paired grid has70 rows with three backbones and zero grids. Initial Task2 RED process completed after implementation because import was delayed; it passed and is not recorded as RED evidence.
Task3 RED: two intended missing-module failures. GREEN: joint AR/association rejection, missing-person fallback, support union and real codec/pinned TrackEval17-frame fixture pass; frame17 preserves ID1 and AR metrics remain null.
Task4 RED: two intended missing-module failures. GREEN: public ar/ot-only plans, bounded packed transport, notebook compilation and size checks pass.
Integrated verification: outputs/test_at.cmd with all V32AT tests and V32 tracking/codec/selection/evaluation/RF evaluation/bridge regressions:71 passed in253.11s, exit0. Earlier RF training regression tests also passed after workspace TEMP fix.
Final whole implementation review dispatched read-only while the complete integration suite ran. No submissions yet.
Task commits:575d5dd (contract/loader),864eb34 (receiver/AR),137763a (joint tracking),65b0c4c (public payloads/config/docs). Integrated71-test suite was green before these source commits.
Final review: no Critical; Important receiver cache geometry and missing secondary AR latency; both enter one RED→GREEN fix pass. Minor (deferred): separate action forward/saliency call counters in joint audit; wall-time already includes these calls.
Review exclusions assessed: executor has independently run the71-test suite; actual Kaggle execution/performance remains future work, with acceptance verification required before reporting submission; quota checked through authorized API; original AR short-video protocol and historical common-prefix association guard retained as explicitly required; GT download hashes do not enter support/selection and parsed GT remains final scoring only; broad old OD/RF behavior/full-benchmark and per-track action recognition have no new requested scope.
Preliminary payload validation: AR971250 serialized bytes, OT917826, both public and source65b0c4c; not submitted. Regenerate from final fixed source in fresh directories.
Final RED: equal-byte/different-shape receiver test reproduced3 output frames for a1-frame input; timing test reproduced only`ar` latency rather than all three backbone names. Both fixes implemented in one pass; integrated regression suite rerunning.
Final: fixed cache geometry — test_equal_pixel_bytes_different_geometry_do_not_share_head_cache RED→GREEN; fixed missing secondary AR timing — test_ar_adapter_scores_same_packets_complete_grid_without_retarget RED→GREEN. Full integrated suite72/72, exit0,93.42s. No additional review requested, consistent with the one-review/one-fix-pass workflow.
Ruling: Keep the experimental branch and checkout after Kaggle deployment, following the requested deployment scope without an unsolicited repository merge/remote push. Cost if wrong: integrate the recorded commits into the desired branch later.
