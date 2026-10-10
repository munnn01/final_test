# V33-AP: shared adaptive preprocessing for OD and AR

Source RGB + QP/codec + source-predicted support → one bounded residual CNN → standard x264/x265 → paid packet → independent standard decoder → the same received RGB for frozen Video Swin and detector. No NeRV, tracking or RGB restoration network. COCO remains a single image; Kinetics auxiliary OD has no mAP ground truth.

Three public jobs (`joint`, `ar_only`, `od_only`) share seed303251, width12, initial identity, source exposure, cached support and two epochs/1920 steps. Their task loss masks differ. Actual codec reconstruction is the forward value; a CAL-only fixed VirtualCodec supplies surrogate derivatives. This is not an exact gradient of H.264/H.265.

Each domain uses TRAIN64 → FIT48/CAL12/PROXYVAL4. DEV16/TEST32 plus fast-preset TEST8 are scoring only. Historical partitions make this an exploratory pilot. AR uses16 stride2 frames at128 square (explicit short-video repetition), followed by official Swin spatial transform; this is not official32-frame/multi-clip K400 evaluation. OD uses320 letterbox and exact inverse x/y scaling. Teacher weights/statistics stay frozen; source targets are detached. LAST checkpoint only.

Evaluation saves identity, forced CNN and guarded CNN for h264/h265 at QP30/35/40/45/50, medium,25fps, bframes0, scenecut off, closed GOP16(AR)/1(OD). Guarded clips must pass both AR and reliable sampled-frame OD preservation plus1% savings; unknown OD evidence falls back to identity. Retarget factor2/4 is an OD-only separate ablation with paid serialized inverse grids. Receiver byte count is unique, and both heads carry the same packet hash. Accuracy is separate for Swin/R3D/R(2+1)D; COCO mAP and AP50 remain separate. Flat/insufficient-overlap BD is null, never an extrapolated gain.

Native entry points:

```text
python -m adaptive_vcm.v33ap.train --arm joint --ar-root AR --od-root COCO_VAL --annotations INSTANCES --device cuda --out TRAIN_OUT
python -m adaptive_vcm.v33ap.evaluate --task ar --root AR --split test --count 32 --checkpoint TRAIN_OUT/shared_last.pth --device cuda --out AR_OUT
python -m adaptive_vcm.v33ap.evaluate --task od --root COCO_VAL --annotations INSTANCES --split test --count 32 --checkpoint TRAIN_OUT/shared_last.pth --device cuda --out OD_OUT
python -m scripts.v33ap_kaggle --arm joint --account OWNER --out PAYLOAD
```

The payload builder requires a clean committed snapshot; it does not submit. Submission uses the authorized external credential pool through subprocess environment, writes durable intents/receipts, and prevents duplicate or ambiguous retry. Notebooks mount both approved datasets, train one shared checkpoint and score both tasks, including transfer. Runtime status/progress/traceback/manifests/checkpoint hashes survive in an evidence archive on failure. Quota is checked only for submission; acceptance is verified once. No completion monitoring is scheduled.

CPU tests inject small frozen teachers and run real FFmpeg streams. They establish mechanics, gradients, geometry and accounting; pretrained GPU memory/runtime/accuracy and BD-rate must be established by Kaggle outputs. Hardware is recorded so accounts with different GPUs are not compared as algorithmic speed gains.

Approved design and execution plan are in `docs/superpowers/specs/2026-10-10-v33-joint-adaptive-preprocessing-design.md` and `docs/superpowers/plans/2026-10-10-v33-joint-adaptive-preprocessing.md`. Validation, decisions, review and submission receipts are recorded in `docs/V33_AP_EXECUTION.md`.

Before fitting, canonical pixel hashes of both domains' DEV16/TEST32 are checked together against TRAIN and each other. These held-out pixels/labels never enter loss or calibration. The frozen cohort is saved with the checkpoint and verified again at each scoring call; the same-ID TEST8 transfer subset is explicitly permitted. Transport preserves every committed file byte, stores the inner ZIP without compression and uses a deterministic outer LZMA container, providing ample room below Kaggle's serialized-source limit.
