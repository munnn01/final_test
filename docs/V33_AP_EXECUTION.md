# V33-AP execution ledger

Plan: docs/superpowers/plans/2026-10-10-v33-joint-adaptive-preprocessing.md
Spec: docs/superpowers/specs/2026-10-10-v33-joint-adaptive-preprocessing-design.md
Base: 1e7e1ed; native implementation approved by the user.

Ruling: Reuse the existing dedicated experimental checkout/branch and native Windows bookkeeping, as approved in the plan and previous campaign; no new worktree or Bash scratch helper — preserves the requested execution method — cost if wrong: move the recorded commits to another checkout and translate the ledger.

Pre-flight: Task1 model/protocol feed Task2 sources and Task3 training; Task2 record/head interfaces feed Task3 training and Task4 receiver; Task3 strict checkpoint feeds Task4 evaluation; Task3/4 CLI and metadata feed Task5 payload. No conflicting interfaces. Historical V32 selector requires two AR teachers, so Task4 reuses only lower-level guard primitives.

Task1: completed. Baseline RF bridge/VC model:22 passed. RED:8 intended missing-module failures. GREEN:31 passed,6.38s,exit0 (new model/protocol + VC model/real codec). Corrected a test fixture: tanh(10) saturates in float32, so the gradient check uses bias0.5 while the bound check retains10. No product change was needed for this fixture correction.

Task2: completed. RED:5 missing-module failures. GREEN:new sources/tasks + RF bridge/VC training,21 passed,16.32s,exit0. Separate AR and OD gradients reach the same head; teachers have no gradients, BN/modes restored, empty pseudo boxes finite, COCO does not invoke action. Canonical pixels and explicit repetition/effective letterbox scales verified.

Task3: completed. RED:4 missing-module failures. GREEN:new training/checkpoint + RF training/bridge/VC training,38 passed,17.28s,exit0. Both actual codecs checked against independent packets and paid bpp; proxy backward reaches CNN. 960 unique source/codec/QP exposures per epoch; validation diagnostics preserve state. Rehashed split tampering and state corruption rejected.

Task4: completed. RED:4 missing-module failures. Timing RED:missing last_times. GREEN:34 passed,45.53s,exit0 (new receiver/evaluate/codec + V32 selection/evaluate/RF evaluate/AT receiver). Full real 2-codec/5-QP/7-family OD grid:70 rows, exact paid accounting, independently decoded grids. Byte-exact standard streams retained while timing encode, independent decode and inverse separately, eliminating the unnecessary encoder-local RGB decode in the new path only.

Task5 implementation: payload RED:2 missing-module failures; GREEN:3 passed,0.58s,exit0 including tamper rejection, bounded transport, duplicate/ambiguous intents and runtime compile. Final snapshot preparation/submission remain pending final review. Receiver cache bug found during author verification: equal RGB with different packet identity retained old per-head hash; test_equal_rgb_different_wire_reports_current_packet failed then passed after attaching current wire hashes.

Whole implementation verification: outputs/v33ap_whole_tests.log —112 passed,61.97s,exit0. Command: outputs\\v33ap_test.cmd tests/test_v33ap_model.py tests/test_v33ap_protocol.py tests/test_v33ap_sources.py tests/test_v33ap_tasks.py tests/test_v33ap_training.py tests/test_v33ap_checkpoint.py tests/test_v33ap_receiver.py tests/test_v33ap_evaluate.py tests/test_v33ap_codec.py tests/test_v33ap_payload.py tests/test_v32vc_model.py tests/test_v32_codec.py tests/test_v32rf_bridge.py tests/test_v32vc_training.py tests/test_v32rf_training.py tests/test_v32_selection.py tests/test_v32_evaluate.py tests/test_v32rf_evaluate.py tests/test_v32at_receiver.py -q --basetemp outputs/temp/v33ap-whole. git diff --check clean. CPU fixture/real-FFmpeg checks only; no pretrained GPU performance claim.

Fresh final review: base1e7e1ed through5842951, read-only whole implementation against the approved spec/plan. Two Important findings: nondeterministic ZIP timestamps/set representation prevented cross-process payload validation; separate evaluation invocations did not audit duplicate DEV/TEST pixels together. No Critical or Minor findings; Declined-to-judge empty. One fix pass, no second review.

Review fixes: deterministic sorted outer ZIP members with fixed timestamps and stable REQUIRED representation. The committed source is repacked as ZIP_STORED, preserving every member byte and manifest hash while leaving adequate notebook size margin. Freeze both domains' DEV16/TEST32 pixel/ID cohort before training; reject duplicates across TRAIN/DEV/TEST and require membership/pixel agreement during evaluation. The intended same-ID TEST8 transfer remains allowed. Held-out pixels are used for provenance only, with no held-out supervision or parameter update.

Review RED:3 failed/6 passed (timestamp rebuild and missing cohort helpers). Targeted GREEN:14 passed,13.94s,exit0. Final regression:115 passed,84.08s,exit0; outputs/v33ap_review_whole_tests.log. Same whole-suite command above, with --basetemp outputs/temp/v33ap-review-whole. Deferred Minor findings: none. Rulings remain the one approved native-checkout ruling above. Preserve branch v32-real-forward and the existing checkout; no integration or cleanup requested.

Additional user request completed: old V32AT OT+AR receipts and archives collected to D:/STUDY/LAB/bao_1/output/v32at_collected_20261010. Report RESULTS_OT_AR.md records OT identity80/80 and BD-rate0%, plus AR Swin TEST32 BD-rate+1.30% H.264/-0.51% H.265. These are previous campaign results, not V33-AP results.
