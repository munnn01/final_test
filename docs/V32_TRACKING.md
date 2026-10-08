# V32 MOT20 tracking screen

This is fixed ROI preprocessing evaluated through frozen Faster R-CNN detectors
and a deterministic Hungarian IoU reference tracker. It does not train a model,
use a V30-B checkpoint, implement SORT/ByteTrack, or claim state of the art.
The encoder teacher is COCO MobileNet Faster R-CNN; the independent evaluator
is COCO ResNet50 Faster R-CNN. Both use person label 1, score >= 0.25. Matching
uses IoU >= 0.3 and retains IDs through at most two missing frames. There is no
motion or appearance model and no interpolated output for missing detections.

```bash
git clone https://github.com/JonathonLuiten/TrackEval.git /kaggle/working/TrackEval
git -C /kaggle/working/TrackEval checkout 12c8791b303e0a0b50f753af204249e622d0281a
python -m adaptive_vcm.v32.tracking \
  --root /kaggle/working/MOT20/train --sequences MOT20-01 \
  --max-frames 32 --size 640 --out /kaggle/working/v32_tracking_results \
  --trackeval-root /kaggle/working/TrackEval --device cuda
```

The verified Hugging Face mirror is [Lekim89/MOT20](https://huggingface.co/datasets/Lekim89/MOT20/tree/5fcaa0ef66906b63195a44672ec260d033545ef4)
at revision `5fcaa0ef66906b63195a44672ec260d033545ef4`. Its train sequences
are MOT20-01, 02, 03, and 05. Download `seqinfo.ini`, `gt/gt.txt`, and exactly
the requested prefix `img1/000001.jpg` onward, keeping that directory layout.
The runner validates every selected image and refuses gaps. It reads the
contiguous prefix from frame 1 at the rational FPS in seqinfo (25), with no
temporal subsampling or padding. A shorter last codec window is kept shorter.
Frames are letterboxed to the requested even square size; exact effective
resize scales and padding are recorded, and predictions are mapped back to
original MOT image coordinates before scoring.

Each point uses actual paid VC32 packets for H.264 and H.265 at QP 30, 35, 40,
45, and 50. Anchor, controls, simplification, retarget, and V32 are reported
separately. Codec windows have at most 16 frames. The evaluator receives
canonical reconstructed frames and keeps one tracker state over the entire
selected sequence, including across codec boundaries.

ROI masks use only frozen source-predicted person boxes. They remain THW for
ROI transforms; the historical OD controls receive their HW union. Optional
`--motion-support` adds the existing source-RGB flow support. Ground truth is
opened only after label-free selection and evaluator predictions finish.
It never supplies ROI masks or candidate selection evidence.

Every candidate guard calls the existing OD `relative_guard` independently
on each clean source / reconstructed anchor / reconstructed candidate frame.
The maximum relative distance must be <= 0.03; every per-frame decision is
recorded. Same-frame anchor detections are matched to candidate detections
using Hungarian IoU matching, and every anchor association link must retain
equal candidate IDs at its two endpoints. Probe trackers clone a common
anchor prefix; the committed anchor state advances between windows. Final
independent evaluator tracks are formed from each arm's actual chosen frames.
Unknown/empty teacher evidence fails closed to the anchor.

Metrics come from the official pinned [TrackEval MOTChallenge adapter](https://github.com/JonathonLuiten/TrackEval/blob/12c8791b303e0a0b50f753af204249e622d0281a/trackeval/datasets/mot_challenge_2d_box.py)
with `BENCHMARK=MOT20` and `DO_PREPROC=True`. All GT classes and flags for the
selected frames are preserved in staged GT, including distractor and ignored
rows; only frame scope and staged seqLength are shortened. This lets official
preprocessing remove distractor matches and excluded ground truth. The
historical pin uses removed NumPy aliases; the runner restores only absent
`np.float`, `np.int`, and `np.bool` aliases without changing metric algorithms.
Missing/incompatible TrackEval fails before expensive probes.

Outputs include source/sequence/config/code hashes, source seqinfo snapshots,
exact frame scope, original and staged GT hashes, every selected packet,
per-window rows and candidate/guard/probe-cost audits, MOT-format predictions,
and official HOTA, IDF1, MOTA, ID switches, false positives, and false negatives.
HOTA/IDF1/MOTA are percentages; ID switches and errors are counts. Rate includes
wire header, grid metadata, and elementary stream, divided by canonical source
T*H*W pixels. All arms share that denominator. Existing nonempty outputs are
rejected to avoid overwriting evidence.

BD curves are exploratory. Undefined curves, including flat quality or no
common quality interval, are represented by null. `target_confirmed` remains
false. A truncated sequence screen is not the full MOT20 benchmark; no
bootstrap confidence is reported from a single short sequence. Dense crowds,
COCO detector transfer, scale reduction, and last-box association can limit
absolute tracking quality. Real dataset quality requires the Kaggle run; local
analytic fixtures verify codec, continuity, and official metric interoperability.

The association guard is conditional on the common decoded-anchor prefix. It
does not guarantee teacher associations for every concatenated selected stream.
The persistent independent per-arm evaluation measures those actual streams.

Prepare the self-contained private Kaggle payload from a clean committed checkout:

```bash
python scripts/v32_mot20_kaggle.py prepare --account vtk269 \
  --slug v32-mot20-hf-f32-s640-s303201 \
  --out ../output/v32_mot20_kaggle \
  --sequences MOT20-01 --max-frames 32 --size 640
```

The notebook contains a verified committed source archive and immutable Hugging
Face/TrackEval revisions. It downloads selected data inside Kaggle. Credentials
stay in the local submission environment. Runtime diagnostics live in the parent
job output; tracking receives a fresh `tracking` child. The final archive includes
both. Errors save diagnostics and are re-raised so a failed kernel cannot appear
successful. Installed dependency versions are recorded in the job output.
