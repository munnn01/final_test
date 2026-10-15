# R2 implementation review

A fresh reviewer inspected the complete change from `cef864eb` to `2ada451`.
The review was read-only and included a 160-frame source/candidate tracking
check across the 64-frame context cap. No critical finding was reported.

The implementation addresses all five important findings:

1. OD bootstrap draws now use the same curve validity rules as the main
   report, with requested/valid counts and a validity fraction.
2. Invalid BD curves clear dependent acceptance flags and BD guards.
3. Cache validation checks the action bank, origin, frozen head hashes,
   source frame counts, active masks, packet hashes and common prefixes.
4. Local measurement verifies source-file hashes; training/evaluation reject
   inconsistent settings, experiment or frozen heads.
5. COCO evaluation preserves undefined size bins when detections are empty.

The small numerical single-map AR finding was treated as a correctness
requirement because the support API promises exact legacy maps. The sole map
is now returned directly, avoiding rounding from recomposition. The legacy
AR/R1 implementation and configurations remain unchanged.

Fourteen reproductions failed before the fixes. The targeted run then passed
20 tests (one HOTA test skipped in that targeted invocation because its
environment variable was absent); the release verification supplies pinned
TrackEval and runs all relevant R1/R2 tests. A further source-hash test checks
that a valid manifest is accepted and a stale file hash is rejected.

The reviewer declined to establish real-data gains, the quality of selected
hyperparameters, hosted completion time/resources, or realistic MOT occlusion
quality from fixture tests. These remain empirical questions for the prepared
experiments. The reported AR map rounding discrepancy was fixed directly;
there is no remaining reliance on whether it changes an encoded packet.

No Kaggle training run is implied by code verification or notebook preparation.

Release verification after all fixes: **103 tests passed, no skips**, comprising
35 R2 tests and 68 V34 regressions. A separate complete four-QP/two-codec OD
fixture run measured 208 packets, trained all three seeds for one epoch each,
and evaluated 80 held-out rows across ten arms. The entire older repository
test suite was not run, and the fixture results are not real-data quality evidence.
