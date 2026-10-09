# V32-RF: real-forward preprocessing experiment

The user approved implementation after the concrete improvement assessment. Keep the completed V32-VC experiment at origin `28cc97db6708c87346c025107a8a7ea485c21163` as the immutable warm-start and reference. Reuse the existing isolated checkout on branch `v32-real-forward`. Execute natively with a fresh whole-branch reviewer. Public Kaggle submission and use of available pool accounts are already authorized; do not follow running jobs after submission, per the user's preference.

## Objective and scope

Implement a controlled next experiment that distinguishes a poor proxy gradient, an ineffective CNN, and a selector that rejects CNN candidates. Deliver selector diagnostics, held-out TRAIN proxy diagnostics, paired proxy/real-forward finetuning, feature consistency, explicit codec execution conditions, forced-CNN evaluation, strict provenance, and public AR/OD Kaggle jobs. Adaptive scales, ROI-QP optimization, learned proxy networks, multi-task training, AV1 and a learned receiver remain subsequent experiments; they are not prerequisites for this controlled comparison.

## Locked experiment

- Same 64 TRAIN sources as the verified prior plain CNN; preserve all pixel and supervision hashes. FIT48 unchanged. Deterministically rank the original CAL16 by SHA256 of `303231/id`: first four become PROXYVAL4, remaining twelve CAL12. Refit the fixed scalar/rate proxy on CAL12 with the new codec settings; PROXYVAL4 enters neither fitting nor optimization. The original warm CNN used FIT48, so PROXYVAL4 was not used for its weight updates. Prior proxy calibration used CAL16, but its parameters are not reused.
- Warm-start from the prior **plain LAST** checkpoint through its complete immutable original source runtime and original strict loader. Never rewrite the checkpoint's provenance or relax V32-VC validation.
- Two fresh copies of that warm state, arms `proxy` and `real`, seed303231, width12, Adam1e-4, two epochs of 480 steps each, identical shuffled source/codec/QP schedules and random seeds, clip norm1, LAST. No retargeting during finetuning, to isolate codec-forward behavior. Fixed retargeting is evaluated afterward.
- Loss: supervised frozen task loss +0.01 estimated paid bpp +10 RGB MSE +0.1 frozen feature cosine distance. Feature target is detached source information: AR layer2 pooled to (2,4,4), OD transformed-image FPN maps pooled to (4,4). Source features cached per source; teacher parameters, buffers, and training flags remain frozen/restored. Rate gradient remains surrogate in both arms.
- `real` reconstruction: `proxy + (real_decode - proxy).detach()`, mathematically equal to real decode in forward and proxy Jacobian in backward. Real encode uses the same uint8 conversion as inference. Actual bytes and codec latency are recorded; no identity-STE claim and no claimed exact codec derivative.
- Primary real codec: x264/x265, medium, QP30/35/40/45/50, fixed GOP16 for AR/GOP1 for OD, bframes0, scenecut off, closed GOP, 25fps. Receiver uses only received packet bytes. Transfer evaluation uses preset fast, the same GOP, TEST8; no V30-B arm there because its strict preset contract is retained. All compared transfer arms share the actual settings.
- CAL12 reuses the three-step scalar search (.5,1,2) and nonnegative affine rate fit. PROXYVAL4 diagnostics before/after report pixel MSE/PSNR, feature mismatch, paid-rate error, and measured rate directional agreement for a normalized surrogate-rate descent perturbation. Diagnostic work never modifies weights, calibration, guards or checkpoints.

## Evaluation

Preserve historical V32 behavior when extension hooks are absent. Add warm CNN, proxy CNN, real CNN, raw retarget, proxy+retarget, real+retarget guarded arms, plus **warm/proxy/real forced plain CNN arms**. Forced arms are explicitly diagnostic and bypass semantic/rate selection; their quality is independently scored. Generate these explicit candidates even when the OD source teacher has no reliable foreground. The existing historical pool in that case stays identity-only.

Primary DEV32 and TEST32: 16 arms including anchor and historical V30-B, two codecs, five QPs =10,240 rows/task. Fast-preset TEST8 transfer:14 arms without V30-B/hybrid =1,120 rows/task. Record elementary/header/grid bytes, packet hashes/CRC, full candidate/selection audits, CNN/grid preparation and all codec probe costs. Replay guards and classify CNN rejection into guard failure, insufficient anchor savings, or another eligible smaller candidate. No checkpoint/parameter selection using DEV/TEST.

Checkpoints validate task/arm, state/meta hashes, fixed recipe, exact FIT48/CAL12/PROXYVAL4 partition and source hashes, warm origin/state, calibration completeness, supervised/feature target hashes, and two finite complete epoch histories. Proxy and real pairs share initial state/data/calibration/recipe provenance. Evaluated source IDs and pixels must not overlap any of the 64 TRAIN sources.

## Verification and limitations

Tests prove real-forward values and surrogate Jacobian, finite actual-codec CNN gradients, feature input gradients with unchanged teachers, recipe/partition/state rejection, forced evaluation versus guarded selection, backward compatibility, explicit FFmpeg flags, and immutable/public/credential-free payloads. A fresh reviewer checks the complete branch before submission. Frozen source snapshots and all checkpoint attachments are exact bytes; credentials stay only in local account environment.

Kaggle archives include warm/runtime provenance, training and diagnostics, all evaluations and packet evidence, model/environment hashes, complete/failed status and traceback even on failure. Submit only after local tests and review. Report submission as submission, not improvement. Accuracy gains, codec-gradient fidelity, realtime operation, multi-seed confidence and a -10% BD target require actual returned results.
