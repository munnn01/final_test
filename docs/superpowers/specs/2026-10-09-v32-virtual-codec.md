# V32-VC design

User-approved direction: CNN is the learned preprocessing block; a virtual codec supplies training gradients; standard H.264/H.265 supplies final bitstreams; optional fixed ROI retargeting follows preprocessing. User explicitly requested implementation on 2026-10-09 after reviewing this architecture.

## Scope and authority

Implement an experimental V32-VC extension beside immutable V32/V30-B. Native execution in this session. Preserve original checkpoints, protocol and results. No automatic claim of improvement. The request to implement supersedes repeated conversational approval handoffs; record concrete design and plan before code. Existing private Kaggle source/checkpoint upload and AR/OD execution permissions persist.

## Preprocessor

Small conditional residual CNN: RGB, semantic protection, motion, QP/51 and codec ID feed two 3x3 spatial convolutions of width 12. AR adds one depthwise 3x1x1 temporal convolution, reset at scene cuts. A single RGB head predicts a tanh residual bounded by 0.1, multiplied by (1-P)*(1-0.75*M). Zero-initialize the RGB head for identity initialization. Restore P=1 exactly before coding. No gate, mixture head, profile bank, learned decoder or postprocessor.

## Virtual codec

Parameter-free tensor operations: BT.601 YCbCr, 4:2:0 chroma sampling, 8x8 orthonormal DCT, scalar quantization, inverse DCT and previous reconstructed-frame prediction for AR. Reset predictor at scene cuts; OD is intra only. Add uniform quantization noise during training; round during diagnostic inference. Gaussian coefficient entropy is an approximate rate, not a transmitted bitstream. Scalar quantizer/rate calibration is fitted using real codecs exclusively on a source-disjoint subset of TRAIN, then frozen. Motion search and B frames are outside this simplified surrogate; report fidelity and transfer limitations.

## Retargeting and gradients

Reuse V32 source-only ROI grids (factor 2/4, margin 0.05), wire format and independent receiver. Differentiable patch resampling follows identical fixed grid coordinates for training and inverse reconstruction; grid construction and factor decisions have no gradient. Train two independent CNNs on the same FIT sources and schedule: plain-only and joint (plain/factor2/factor4 round-robin). Evaluate joint CNN plain-only separately to distinguish training from runtime geometry. Hard protection is a pre-codec constraint, not a guarantee after quantization or resampling.

## Training

64 source pilot per task, deterministic source-level TRAIN split: 48 FIT and 16 CAL. Seed 303221. Two codecs, QP 30/35/40/45/50. Four fixed epochs, Adam lr 1e-4, batch 1, gradient clipping 1.0, LAST checkpoint only. Loss = task + 0.01*estimated total bpp + 10*source reconstruction MSE. AR task loss is mean cross entropy from frozen R3D18 and MC3-18 on FIT labels. OD task loss is the sum of frozen MobileNet Faster R-CNN classifier/box/RPN training losses on FIT ground truth, with batch statistics/dropout fixed. Only CNN optimizer parameters update. Ground truth never constructs ROI or chooses a transmitted candidate. Independent R(2+1)D18 and ResNet50 evaluators are excluded from optimization and guard selection.

CAL fits positive affine rate mappings per codec/QP using real elementary byte counts for identity and deterministic bounded pixel edits; retargeted examples are included where geometry permits. Quantizer step scale is selected from a fixed small set by reconstruction MSE, independently per codec/QP. Record reconstructed-pixel error, rate error and edit-delta sign agreement. Calibration is diagnostic; actual codec comparison decides success, even if surrogate training loss decreases.

## Evaluation

Extend an explicit separate evaluation CLI, reusing V32 infrastructure without changing its old CLI behavior. Paired arms: existing anchor/controls/V30-B/simplification/retarget/hybrid/V32 plus VC plain, joint CNN without retarget, raw retarget-only and joint CNN with retarget. All candidates measured with real H.264/H.265 medium at five QPs; source-based bitrate denominator, full paid wire bytes and all probe costs. No equal-K claim. Guard thresholds remain 0.1 AR/0.03 OD with 1% savings; failure uses identity/control fallback.

First DEV32 pilot; do not tune on the already observed DEV32. A locked TEST32 comparison follows when runtime budget allows, labeled exploratory, with cubic/PCHIP overlap-only BD-rate, per-QP quality deltas and paired uncertainty where feasible. New CNN uses 48 FIT sources versus historical V30-B's 32 TRAIN sources; disclose this rather than attributing every difference exclusively to architecture.

## Provenance and failure behavior

Store source IDs and pixel hashes, FIT/CAL partition, code/config hashes, seed, calibration, history and exact model state digest. Reject missing/incompatible/nonfinite checkpoint data, mismatched task/protocol and evaluation overlap with either FIT or CAL by ID or pixels. Source archive must be committed and credential-free. Empty output directories prevent mixing runs. Kaggle mounting uses bounded known dataset paths, no whole-input recursive scan.

## Required verification

Gradient reaches CNN through frozen task network, fixed virtual codec and both retarget directions; teacher weights/statistics remain unchanged. Quantizer round is absent from training gradient path. DCT inversion, YUV geometry, cuts, constant/short clips, bounded output, exact protected RGB, malformed state and TRAIN/evaluation leakage tested. Real codec receiver smoke for both codecs and mapped geometry. Targeted inherited V32 transport/selection/baseline tests before fresh final review. Cloud results carry source/environment/checkpoint hashes and complete paired grids; no improvement claim from synthetic tests.

## References

- Zhao et al., A Preprocessing Framework for Video Machine Vision under Compression: https://arxiv.org/html/2512.15331v1 (virtual codec/AR and tracking; OD extension here is our experiment).
- Rozek et al., ROI-based retargeting: https://link.springer.com/article/10.1186/s13640-025-00682-3 (fixed geometric branch already implemented in V32).
