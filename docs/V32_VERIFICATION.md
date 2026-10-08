# V32 verification — 2026-10-08

Verified implementation snapshot: `ab1021ebf6d70d869efcff57b919ea0ec1e4e8af`,
standalone branch `v32-roi-retargeting`. Later documentation commits do not alter
the Python/config source verified here. No AR/OD task gain is claimed.

## Local verification

- Final inherited suite: **777 passed in3620.64s (1:00:20)**; source and Git HEAD frozen throughout.
- Core V32 regression set: 50 passes, plus the added stream-resolution-change
  regression passed separately. Covers geometry, paid wire recipe, receiver
  bounds, arm completeness, source/provenance refusal and actual paired OD CLI.
- Tracking module: 17 passes, including official pinned TrackEval distractor
  preprocessing, actual two-codec packets, complete50-point analytic fixture,
  persistent IDs through frame17, and model-state digest.
- Kaggle payload: 21 passes after the output-directory boundary regression.
- Six fresh real x264/x265 smoke packets at this snapshot independently redecoded
  exactly. A64×64 synthetic source retargeted to28×28 and reconstructed to64×64.
  This validates transport, not task quality or BD-rate gains.
- Existing original AR and OD V30-B checkpoints loaded through the unchanged
  original runtime provenance checks; three proposals preserve fully protected
  pixels. V32 introduces no training, backward or STE.

Logs and synthetic evidence are outside the checkout at:

- `D:/STUDY/LAB/bao_1/output/v32_tests/full_frozen_ab1021e.log`
- `D:/STUDY/LAB/bao_1/output/v32_tests/payload_output_green.log`
- `D:/STUDY/LAB/bao_1/output/v32_transport_final_ab1021e/summary.json`

An earlier full-suite run was invalidated by code changing during measurement:
727 passes/2 failures. The frozen-source rerun above replaces it. No invalid run
is presented as successful verification.

## Review

Fresh original-scope and subsequent OT-scope reviews found four material issues
in total. Receiver frame/geometry aliases and unbounded RGB capture, omitted
whole-arm completeness, and the Kaggle/CLI empty-output boundary now have
RED→GREEN regressions. No Critical issue or actionable Minor remains deferred.
No duplicate re-review of already fixed findings was requested.

## Actual private Kaggle MOT20 screen

Version1 completed:
https://www.kaggle.com/code/vtk269/v32-mot20-hf-f32-s640-s303201

Server metadata confirms private/T4/Internet. Runtime uses CUDA,
torch2.11.0+cu128 and torchvision0.26.0+cu128; installed packages, container
digest, weights enums and actual model-state digests are archived.
All57 runtime Python file hashes match the committed archive. Original commit
and ephemeral runtime Git commit are recorded separately.

- Hugging Face `Lekim89/MOT20` revision
  `5fcaa0ef66906b63195a44672ec260d033545ef4`.
- MOT20-01 frames1–32 of429, source1920×1080,25fps; letterbox640, two windows16.
- TrackEval `12c8791b303e0a0b50f753af204249e622d0281a`, official MOT20 preprocessing.
- 50 complete paired arm/codec/QP points;100 window rows and selected packet
  byte lengths/SHA256 verified after download.
- All arms choose identity: selected byte saving0%; HOTA/IDF1/MOTA equal anchor.
  All160 edited probes save>=1% bytes but fail anchor association preservation;
  146 also exceed detector relative-distance0.03. Thresholds were not tuned on
  this result. Candidate savings are not accepted selected-stream savings.
- GT is read only after selection and predictions, with distractor/ignored rows
  retained for official preprocessing. A32-frame single-sequence screen does not
  confirm the full MOT20 benchmark or the BD-rate target.

Report and machine evidence:

- `D:/STUDY/LAB/bao_1/output/v32_mot20_kaggle/KET_QUA_VI.md`
- `D:/STUDY/LAB/bao_1/output/v32_mot20_kaggle/tracking_points.csv`
- `D:/STUDY/LAB/bao_1/output/v32_mot20_kaggle/artifact_inspection.json`
- `D:/STUDY/LAB/bao_1/output/v32_mot20_kaggle/guard_summary.json`

Scope rulings and rationale are in `V32_EXECUTION_LEDGER.md`: preserve the separate
checkout, use fixed ROI plus optionally frozen V30-B, pay reconstruction metadata,
and treat OT guard as conditional on the common anchor prefix. Actual per-arm
independent tracking persists across windows. No new expert training or bit-exact
VVC paper reproduction is asserted; statistical target confirmation remains false.
