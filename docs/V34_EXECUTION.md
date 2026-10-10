# V34 execution ledger

Plan: `docs/superpowers/plans/2026-10-10-v34-adaptive-controller.md`.

Ruling: User explicitly requested immediate implementation of the linked detailed research direction. Preserve prior native execution preference and public Kaggle authorization; prepare concrete spec/plan and proceed without repeating design authorization.

Ruling: Existing checkout is clean on a dedicated development branch. Work in place on new branch `v34-adaptive-controller`, with isolated v34 modules; do not create an additional worktree or alter V33 behavior.

Ruling: Initial action bank implements six region/band coefficients with a finite search. Residual/proxy fine-tuning and JRDB joint-label benchmark require positive evidence/access and remain future experiments; deployed notebooks report exploratory multi-domain scope explicitly.

Pre-flight: Task1 map/action/model interfaces feed Task2; Task2 calibration/checkpoint feed runners; runner utilities feed OT; payload invokes runners. Both experiment jobs share an immutable recipe and seed list. No interface conflicts found at plan start.

Task1 complete: RED missing package, GREEN 4 tests; commit 6bf6723. Identity Q8, real edits, causal scene-cut history and controller gradients verified.

Task2 complete: RED missing learning/checkpoint, GREEN initial3 tests; commit7d1d7ac. Added three-seed native checkpoint and mixed-length training regressions; final learning5 tests pass. No gradient crosses codec or association.

Task3 complete: RED missing sources/runner, GREEN4 tests with actual x264/x265; commit111ba03. Real codecs require workspace TEMP in this Windows sandbox. Shared receiver predictions and costs retained, CAL frozen before held-out evaluation.

Task4 complete: RED missing tracking, GREEN ByteTrack3 tests; commitad59752. Corrected bidirectional identity surrogate after GT-merging regression. Additional official pinned TrackEval smoke returns HOTA100/IDF1=100 on a synthetic perfect-track fixture; no benchmark claim.

Task5: RED missing payload builder; GREEN public/recipe/compilation/no-secret/idempotent timeout3 tests. One fresh whole-branch review found mixed T stacking (P1) and discarded secondary source scores (P2). Both fixed and verified independently. All21 V34 tests pass, including real codecs and TrackEval.

Verification ruling: Initial bare pytest baseline had70 failures/214 fixture errors due Windows sandbox TEMP; workspace TEMP removes that problem. A whole-suite attempt during active source edits invalidated a V31 manifest test; the same V31 test passes independently after stabilization. Cancel that changing-code attempt and run the final full suite only after source is frozen. Do not label the transient manifest mismatch a V34 regression.

Kaggle preflight: dngbolm and htran123456 each report30 GPU hours and no active/unknown owned job. Other accounts with redacted private notebook placeholders or active/unknown jobs were excluded. Recheck exact final slug before submission. Two source-only public jobs remain authorized; no completion monitoring.

Scope ruling: First pilot includes AR paired bootstrap at same QP; OD CI and OT sequence CI remain null and are explicitly reported. OT uses one sequence per held-out split and window-local training surrogate versus continuous deployed ByteTrack. Single-GOP buffering and the actual-byte guard require an anchor encode plus a proposed encode; deployment cost is recorded, without claiming single-encode operation.

Final V34 verification: unchanged commit6ca686b passed all21 V34 tests again in151.76 seconds, including actual H.264/H.265 and pinned official TrackEval. Random-weight Swin/OD architecture/API smoke passed; this is not an accuracy result. Both packed public payloads were unpacked and checked against the account-pool secrets; no secret bytes found, and all source/hash/recipe checks passed.

Full-suite limitation: final native run stalled at39% in legacy `test_train_motion.py::test_real_codec_train_caches_source_and_fits_final_spatial_pixels`, inside a Windows subprocess pipe-reader join. There was no live encoder child when inspected. Stopped the verified test process and preserved `output/v34_validation/native_full.txt`; the full repository suite has no final passing result. Submit the isolated exploratory V34 pilots on the strength of the relevant tests, actual-codec checks and completed code review, with this limitation reported explicitly.

Final account refresh: fresh quota and owned inventory checked before each upload. Terminal status evidence was reused only for identical notebook references and lastRunTime; changed versions would require a fresh status check. Both accounts remained eligible with30 GPU hours. Completion monitoring is left to the user; no automation was created.

Submission complete: Kaggle accepted public version1 of `dngbolm/v34-ar-od-6ca686b7` and `htran123456/v34-ar-ot-6ca686b7`. Each immutable notebook contains code commit6ca686b and three seeds. Durable individual receipts are in `output/v34_kaggle/{ar_od,ar_ot}/submission.json`. Acceptance was verified from the push response and one owner-list lookup; no worker status/log/completion polling followed. Benchmark results remain pending and no performance improvement is claimed.
