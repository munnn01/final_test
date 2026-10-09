# V32-AT: Action Recognition and Object Tracking

The user selected only AR and object tracking, removing standalone OD from the new campaign. The detailed joint-pipeline design was presented before that instruction: [AR_TRACKING_PIPELINE_VI.md](D:/STUDY/LAB/bao_1/output/v32_task_split/AR_TRACKING_PIPELINE_VI.md). Preserve native execution, existing public Kaggle/available-pool authorization, and the user's instruction not to monitor after submission. Historical OD code, immutable experiments and detector components are reference/library material, not a third active task.

## Scope and architecture

Active tasks are exactly `ar` and `ot`. Video Swin-T is a frozen classifier after standard decoding/inverse retarget, not a new encoder Transformer or RGB restoration head. The tracking head uses the existing frozen detector plus Hungarian IoU reference tracker. Reuse existing weights and codec infrastructure; this campaign does not train a new network, change the tracking algorithm, or add action classification per track ID.

Provide a shared receiver operating on already decoded packet RGB, with explicit action and tracking outputs. In the MOT20 joint run, a packet is encoded/decoded once per candidate and its selected reconstruction is shared by the Swin and tracking heads. Preserve tracker state over codec windows, separately for each sequence/codec/QP/arm. AR operates on actual received clip frames; tracking keeps all ordered frames at source FPS. Keep the unpadded short final codec window. Record the actual AR window length rather than silently duplicating frames.

The AR evaluation on Kinetics uses frozen RF-real preprocessing/controls with all retarget candidates disabled, Swin as primary evaluator and R3D/R(2+1)D as secondary evaluators. Disabling retarget must affect both historical and extension candidate builders. Keep a forced-CNN diagnostic alongside guarded selection. On MOT20, fixed ROI/retarget candidates remain optional and must pass both the AR semantic guard and the existing tracking association guard. Protect the union of predicted person ROI, motion and source AR saliency; never use GT for ROI or selection. Identity remains fallback.

## Immutable models and execution

Frozen RF-real AR checkpoint: `output/v32rf_kaggle_packed1/ar/extracted_v1/v32rf-ar-fit48-test32-s303231/training/real_last.pth`, 40169 bytes, SHA256 `12f05e4a2647890f33d27bc0a95b495398f4e1fc2ada1ce1beee017a984cb272`. Load it with the unmodified strict loader from its source snapshot `50a22e57ad68bba91ad9d1ffc4caec5140ede259`, archive SHA256 `26659e08959250b8dfb297950998f4ca3024006d9e2faefaf2185600d3c19270`. Do not rehash checkpoint provenance to accept new runtime code. Verify every original source file, state hash, metadata, fit/cal/validation membership and overlap through the original loader.

Use frozen torchvision `swin3d_t`, `Swin3D_T_Weights.KINETICS400_V1`, official transform. Tracking keeps the existing detector models/thresholds/reference tracker/TrackEval pin. Codec settings for compared new arms: x264/x265, medium, QP30/35/40/45/50, GOP16, bframes0, scenecut off, closed GOP. Kinetics keeps its original 16-frame, stride2, 128-square evaluation protocol; MOT20 uses contiguous frames at seqinfo FPS, 640-square letterbox, no temporal subsampling. These are separate benchmark protocols, not identical source videos. All paired arms within a job share source geometry and codec settings.

## Runs and evidence

Create exactly two public Kaggle jobs: `ar` Kinetics DEV32/TEST32 inference evaluation, and `ot` MOT20-01 first128 frames with joint Swin output. Use existing data mount/Hugging Face revision and locally configured account pool. This is an exploratory screen, not full MOT20 or a new independent Kinetics TEST set. Do not refit or select a checkpoint using these splits.

Kinetics supplies AR labels; MOT20 supplies tracking GT. The joint MOT20 run saves Swin probability/top-class/window outputs without fabricating AR accuracy or AR BD-rate there. Tracking reports official HOTA, IDF1, MOTA and ID switches. Each labeled task reports its own BD curves; undefined/flat-overlap curves remain null. Ground truth is read only after label-free selection and receiver predictions. Retarget inverse derives grids solely from packet bytes. Count elementary/header/grid bytes once, normalize by canonical source pixels, and report all candidate-search/receiver costs separately.

Save immutable source/model/environment/config hashes, full frame/source scopes, packets, selection/dual-guard audits, per-task output files and failure tracebacks/status. The new CLI, manifests and Kaggle plan reject standalone `od`. Existing historical APIs remain compatible when new hooks are omitted. Payloads contain no credentials and fit the previously verified Kaggle serialization-size limit. After submission, verify accepted handles/version once and stop; do not set up or perform progress monitoring.

## Acceptance

Tests demonstrate OD rejection, no AR retarget construction, frozen model/source validation, same packet/RGB used by both heads, no duplicate byte counting, tracking state isolation/continuity, correct source-coordinate mapping, independent dual-guard rejection, short-window behavior, truthful missing-AR-label reporting, and credential-free/public bounded payloads. Run the relevant V32 codec/selection/tracking/RF/Swin regressions and a fresh whole-branch review before submission. Report submission separately from results. No BD-rate gain is assumed from sharing a receiver.
