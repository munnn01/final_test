# V34-R1 independent Video Swin validation — 2026-10-11

The purpose is to test whether the locked adaptive preprocessor and real-codec
source guard improve AR rate–accuracy on videos not used in pipeline development.
This is an evaluation extension. The V34-R1 model, calibration, filter bank,
source guard, task heads, codec settings and controller conditions remain fixed.

## Important audit correction

All 64 AR clips in the previous R1 TEST belong to the official Kinetics-400 TRAIN
split. The project's hash-based TEST naming did not establish independence from
the pretrained backbone's training data. Old results are exploratory. This new
study admits only official K400 validation YouTube IDs, removes any official
TRAIN overlap and excludes video IDs and canonical pixel hashes found in local
experiment history. Independence here means independence from adaptive pipeline
development and official backbone TRAIN. The pretrained backbone may itself have
been selected using its published validation performance.

## Frozen design

- Source: public `jizeyong/kinetics400-rgb256-f10-val-data`, using provider JPEG
  frames. Membership and labels are verified against CVDF's official K400 val
  CSV; official train CSV is used only for exclusion. The provider's original
  decoding/resampling cadence is not independently verified. This is an
  independent frame-source evaluation, not a claim to reproduce original MP4
  sampling or official full-video/multi-view accuracy.
- Target: 1,024 unique available eligible YouTube videos; seeded uniform hash
  ranking (`20261011`) chosen before running task models. Four deterministic
  disjoint shards of 256 videos. Insufficient or undecodable selected data fails
  the run; do not shrink the cohort or replace videos based on model results.
  Revision 2 additionally requires at least 32 provider `img_*.jpg` files before
  hash ranking. Count every available official-val folder and retain its count
  and any short-source exclusions in `source_inventory.json`.
- Inputs: 16 distinct center frames, stride 2 in the provider frame sequence,
  canonical RGB 128×128, INTER_AREA. No padded short clips. Record frame names,
  frame-file SHA-256, pixel SHA-256 and complete global selection on each shard.
- Models: original LAST controllers for seeds 303401/303402/303403, source commit
  `df1fa0dd712e5d6c7046a24b18cdd1a306a2de59`, verified byte/state/metadata hashes.
  No training, recalibration, threshold fitting or seed selection on new data.
- Same source-only AR+OT guard and encoder support as R1; no new MOT GT test in
  this AR validation. Retain anchor, fixed, forced, calibrated and selected arms.
- Standard real x264/x265; medium; QP 30/35/40/45/50; 16-frame closed GOP,
  no B frames or scene cuts; coded FPS 25. Include the common paid header.
- Primary endpoint: Swin3D-T rate–top-1-accuracy, preselected controller seed
  303401, both codecs. Other seeds are robustness checks; R3D-18 on the same
  packet is a secondary transfer check. No tuning against TEST labels.
- Global inference: merge all four completed matching shards; resample source
  videos, sharing each draw across QPs. 5,000 paired-video bootstrap draws, seed
  20261012. Use two-sided 97.5% percentile intervals for the two primary codec
  endpoints (Bonferroni familywise 95%); descriptive 95% for secondary results.
  Report invalid/finite fits; require at least 99% finite draws for confirmation.
- Predeclared confirmation: primary passes the existing V34 point screen on
  both codecs (canonical and PCHIP BD-rate below −10%, nonnegative BD-quality,
  no same-QP loss worse than 0.5 pp, nonincreasing same-QP paid rate), and both
  primary BD-rate interval upper bounds lie below zero. A shard cannot confirm
  the target. No claim that the Guard guarantees correctness on each video.
- Report all raw QP accuracies/rates, confidence intervals, corrected/damaged
  video counts, choice frequencies and deployment candidate work/timing.
  Report failure or degradation without changing this design after viewing it.

## Implementation and validation

`adaptive_vcm/v34/holdout.py` loads frozen checkpoints and reuses the unchanged
R1 evaluation path. It adds official-val sampling, provider-frame loading,
historical overlap checks, state invariance checks and a paired global merger.
`scripts/v34_holdout_kaggle.py` prepares four public immutable payloads. Payloads
contain code, exclusion/annotation/protocol manifests and the three tiny frozen
controllers; never account credentials or the large task-model weights.
To fit Kaggle's notebook source limit, the transport retains every committed
`adaptive_vcm` Python file, every config and `pyproject.toml`; docs, tests and
local orchestration scripts are omitted. Kept files are exact committed bytes
and verified individually against the source manifest and original R1 core.

Tests cover official TRAIN exclusion, history exclusion, disjoint deterministic
shards, failure on insufficient/short data, pixel overlap, paired-grid mismatch,
exact row-to-selected-cohort binding, shard identity/count, frozen state hashes,
annotation/exclusion hashes, rejection of changing the 5,000 bootstrap draws,
and a numerical paired bootstrap with real five-QP binary-accuracy curves,
and a real x264/x265 holdout run with synthetic external task heads checking that
GT labels cannot reach source support/Guard. Frozen heads and checkpoint metadata
are checked against the original R1 output before submission.
Local validation before revision 2: all V34 tests passed (52 passed, 2 skipped), including real
x264/x265 integration. Windows codec temporary files were placed in the writable
workspace. The skipped tests require external tracking assets not used here.

GPU runs use already authorized public Kaggle notebooks and eligible accounts
from the existing pool. Submit once with durable intent/acceptance receipts;
do not start an automatic monitor. Independent results will be collected when
the user requests a check.

The merger requires the locally frozen `protocol.json`, rather than accepting
the returned notebook's protocol as authoritative. Example after collection:

```powershell
python -m adaptive_vcm.v34.holdout merge --protocol <frozen-protocol.json> --paths <shard0-experiment> <shard1-experiment> <shard2-experiment> <shard3-experiment> --out <independent-analysis>
```

## Revision 2: source eligibility repair — 2026-10-11

All four initial sessions ended with `short frame source; no padding/replacement
allowed` while loading selected clips, before `evaluation.evaluate` and before
any AR accuracy/BD-rate was generated. Original selection admitted official-val
folders without checking whether the provider had at least 32 JPEG frames.

The repair prospectively defines eligible videos as official-val IDs present in
the provider with at least 32 JPEG frames, then applies the same history exclusions
and hash ranking. This changes the sampling population to sufficiently long frame
sources; it does not replicate the original 1,024 selected IDs. All counts and
exclusions are retained, including clips excluded solely for insufficient frames.
The new protocol is separately frozen as revision 2 before running task scoring.
No task accuracy, model preference, or GT-dependent filtering determines eligibility.
The original 16-frame stride-2 recipe, 1,024-video target, four shards, checkpoints,
codecs, QPs, guard thresholds and inference criteria remain fixed. Decode failure
after selection still aborts the run rather than replacing the selected video.

The regression test exercises real provider-like folders with 31, 32 and 40 files:
31 cannot enter the candidate set; 32 and 40 can; insufficient eligible data still
fails rather than reducing the target. The real-codec frozen integration test is
also rerun with the explicit 32-frame eligibility declaration.
Revision 2 local verification: all V34 tests passed (53 passed, 2 external
TrackEval tests skipped), including the frame-count regression and real-codec
frozen evaluation path. This verifies the repair locally; a remote `RUNNING`
status by itself is not evidence that task evaluation has started or finished.
The broader 957-test repository suite was also invoked with a 180-second bound.
It timed out during `test_actual_byte_training_and_isolated_component_evaluation[od]`
in `test_rateaware.py`; no named failure was observed before timeout. The partial
run is not reported as a full-suite pass.
