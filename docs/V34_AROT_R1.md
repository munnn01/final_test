# V34 AR+OT repair R1

User requested repair after the V34 AR+OT pilot selected identity for every
held-out condition. This recipe preserves the standard codec and adaptive
region/band filtering. Original V34 and its public results remain reproducible.

## Evidence and bounded repair plan

1. Official TrackEval HOTA is in percent; the generic curve helper expects a
   fraction. Normalize once. Suppress BD fits on nonmonotone tracking curves;
   retain same-QP differences and official raw metrics.
2. AR and MOT clips had the same six controller conditions despite different
   label/pseudo-label losses. Add source-domain conditions, normalize the risk
   training residual by its task tolerance, and keep separate empirical CAL
   residual tables by domain. CAL is not an independent coverage guarantee:
   codec/QP rows and MOT windows are correlated.
3. Add a deployment arm that checks a bounded CNN shortlist against the real
   decoded anchor using source-only Swin feature/KL and track consistency.
   Predeclared limits are 0.025 AR and 0.05 OT excess cost, and 1.01 measured
   byte ratio. It uses no held-out labels or IDs and makes no accuracy guarantee.
   Preserve the calibrated-only arm as an ablation. Log every tried action,
   guard cost, rejection, encoding and inference time.
4. Use frozen ResNet50 FPN for this AR+OT recipe, 960-pixel canonical MOT video
   and detector transform, and 300 detections per image. Keep the cheaper
   320-pixel support transform on K400. Record clean-source continuous tracking
   alongside codec anchors to expose a weak detector independently of coding.
5. Prove regressions and codec/checkpoint/payload behavior locally, review the
   complete change once, then submit one public three-seed Kaggle notebook on
   an eligible account. Do not monitor completion automatically.

Ruling: the existing request to repair and authorization for public Kaggle
execution cover this bounded follow-up. Native implementation continues without
another design/permission round. Risk: this is an exploratory repair, not evidence
that the method improves either task. Historical cohorts were already inspected;
new results are diagnostic rather than independent confirmation.

## Verification ledger

Regression RED observed for all seven initial repair contracts (one temporary
directory setup issue was corrected and that contract rerun RED). GREEN:
31 V34 tests passed in 27.59 seconds, including real H.264/H.265 and official
TrackEval. The end-to-end fixture also exercises clean continuous tracking,
label-free guard selection, the calibrated ablation and shared decoded stream.

Native real ResNet50 FPN / Swin3D-T architecture smoke passed with random
weights. Detector inputs were 960x960 for MOT and 320x320 for K400; models
remained frozen. This proves API/shape compatibility, not accuracy.

Fresh whole-change review found one Important issue: rejected shortlist
candidates' preprocessing/decode time was omitted. A new assertion failed
with KeyError `deployed_candidate_work`, then passed after adding per-action
work, CNN inference time, deduplicated guard time, sender total including source
references, and benchmark total including the transfer evaluator. No other
Important/Critical findings were reported.

Ruling: do not repeat the known unrelated full legacy suite Windows codec pipe
stall. No full-repository pass is claimed. Evidence is preserved in
`../output/v34_validation/full_suite_limitation.json` and fresh repair test/smoke
logs in `../output/v34_r1_validation/` (paths relative to repository root).

Public source-only payload passed immutable validation and credential-byte scan
(341 source members; 621,719 serialized notebook source bytes). Source commit
`df1fa0dd712e5d6c7046a24b18cdd1a306a2de59` is pinned in the payload.
Fresh account preflight found 27.14 GPU hours and terminal previous notebooks.

Kaggle accepted version 1:
https://www.kaggle.com/code/htran123456/v34-ar-ot-r1-df1fa0dd

Submission evidence is saved in `../output/v34_r1_kaggle/ar_ot/submission.json`.
No completion monitoring was started. The three seeds run the same exploratory
historical cohorts; results must not be described as independent confirmation.
Task improvement remains unverified until the new outputs are available.

Ruling: preserve the existing native branch and checkout for continuing this
research; no Git remote push, merge or cleanup is needed for Kaggle execution.
