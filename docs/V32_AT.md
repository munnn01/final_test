# V32AT: AR and object tracking

This campaign has two active tasks: action recognition and object tracking.
The detector is an internal tracking component; no standalone OD job is launched.

AR path: source clip → frozen RF-real CNN preprocessing or controls → x264/x265
→ independent packet decode → frozen Video Swin-T, R3D and R(2+1)D classification.
Retarget construction is disabled for every AR candidate. Kinetics DEV32 and
TEST32 use 16 frames at stride2, 128-square, with original FIT/CAL pixel exclusion.
Guarded RF and forced RF are separate comparisons; forced RF is diagnostic.
Historical `retarget` summary arm remains a controls-only alias on this path.

Joint MOT20 path: source person boxes + motion + source Swin saliency → fixed
ROI simplification/retarget proposals → real codec → independent packet decode
and inverse map → the same reconstructed RGB for frozen Swin and detector →
Hungarian IoU reference tracker. Candidate acceptance requires both tracking
association preservation and the existing AR anchor-relative semantic guard.
Missing reliable source person evidence falls back to identity. No GT enters
candidate selection. TrackEval ground truth is staged after all predictions.

The receiver never encodes or decodes. It caches frozen-head predictions, while
each sequence/codec/QP/arm retains independent mutable tracking state. Frames
remain ordered and contiguous across codec windows; the final window is unpadded.
Tracking coordinates map back to the original sequence dimensions.
Prediction cache keys include decoded shape, dtype and pixel hash. AR rows record
latency for Swin, R3D and R(2+1)D plus total prediction work. Reused packet predictions
are marked explicitly and charged zero inference work for the repeated row.

Common conditions: x264/x265 medium, QP30/35/40/45/50, closed GOP16, no B frames,
scenecut disabled. Rates include the whole paid packet and ROI metadata.
MOT20-01 first128 frames, 640-square, uses seqinfo FPS. MOT20 has no action
labels: Swin whole-window predictions are saved, while AR accuracy and AR BD-rate
are explicitly null. The reference tracker is not ByteTrack and no action-per-ID
labels are claimed. Results are exploratory, not full-benchmark claims.

The immutable RF runtime and checkpoint are loaded through their original strict
loader. Exact source, bundle and checkpoint hashes must match before model import.
No checkpoint metadata is rewritten. `configs/v32at.json` sets the new campaign
seed without changing the original runtime's training configuration.

Entry points: `python -m adaptive_vcm.v32at.ar` and
`python -m adaptive_vcm.v32at.tracking`. Public payload preparation uses
`python -m scripts.v32at_kaggle --task ar|ot ...` from a clean committed snapshot.
Exactly two Kaggle jobs are submitted, with no monitoring after acceptance.
