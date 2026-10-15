# V34-R1: separate AR, OD, OT with four QPs — 2026-10-15

User requested a fresh V34-R1 rerun with separate AR, OD, and OT experiments.
The user explicitly selected QP **30, 35, 40, 45**, removing QP 50. This applies
to real-codec measurement, TRAIN, CAL, DEV, and TEST in the new recipe.

`configs/v34_r1_qp4.json` copies the R1 settings, changing only the revision
and QP grid. Original five-QP configs and combined AR+OD / AR+OT experiments
remain available. This is a fresh training experiment, not evaluation of the
old five-QP checkpoints or removal of a point from their final curves.

Each task has its own source cohorts, measurements, controller inputs, three
new controllers (seeds 303401/303402/303403), domain calibration, and results.
The supervision mask is one-hot for the chosen task. Frozen detection still
provides encoder region maps for AR. OT uses frozen detections and ByteTrack;
its single-task run has no AR consistency objective or AR evaluation.

The unchanged bank contains 13 actions, including the identity anchor. Every
TRAIN/CAL source is processed by all 13 actions through real x264 and x265 at
the four QPs, with medium preset and the original closed-GOP settings. Actual
paid bytes and frozen task costs form detached, fixed supervision. No gradient
crosses the codec or tracker. CAL sources receive no optimizer updates.

AR and OD retain 48 TRAIN, 16 CAL, 16 DEV, 64 TEST sources each. Each has 512
TRAIN+CAL measurement rows and 6,656 candidate encodes. OT retains the R1
MOT20-02 TRAIN/CAL, MOT20-03 DEV, MOT20-01 TEST sequences, 256 frames each,
and 16-frame contiguous windows: 12 TRAIN windows, 4 CAL, 16 DEV, 16 TEST.
Its TRAIN+CAL table has 128 rows and 1,664 candidate encodes.

The source-only guard retains R1 limits and shortlist. For OD, this path uses
source detection/feature consistency and does not access held-out GT boxes.
GT is used by training costs and final external scoring. OT continues official
TrackEval HOTA/IDF1/MOTA and a clean-source tracking baseline.

Verification includes task-only masks and label-free costs, the full four-QP
measurement/training grid with both real codecs and three native checkpoints,
task-only held-out evaluation including official tracking metrics, immutable
public payload validation, and the existing V34 regressions. The known legacy
whole-repository Windows codec-pipe stall is outside this change; no complete
legacy-suite pass is claimed.

Fresh verification: 68 relevant tests passed in five bounded invocations:
7 recipe/mask/payload tests (21.52 s), the OD full measured-grid training test
(67.77 s), AR and OT full measured-grid training tests (138.11 s), 3 actual-codec
held-out evaluation tests including official TrackEval (15.16 s), and all 55
existing V34 regression tests (40.34 s). No skipped test in these final runs.
An earlier combined measurement run was interrupted after its first test;
all three task cases subsequently completed successfully in bounded runs.
Independent source review found no Critical or Important issue. Its noted
evaluation coverage gap was closed by the three held-out evaluation tests.

Public Kaggle execution is authorized by the request to rerun these experiments
and the existing project's execution workflow. Prepare exact committed source,
scan payloads for credentials, verify account quota and terminal owned jobs,
then submit each immutable task notebook once with durable receipts. Do not
create a recurring monitor. Acceptance or a RUNNING state does not establish
completed training or improved task performance. These historical cohorts
remain exploratory and are not an independent confirmation study.
