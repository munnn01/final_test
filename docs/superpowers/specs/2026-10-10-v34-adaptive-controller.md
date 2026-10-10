# V34 adaptive filtering controller

Implements the research direction approved by the user with "triển khai ý tưởng này luôn, có thể dùng nhiều nick kaggle". Native execution preference carries forward. The research report is `D:/STUDY/LAB/bao_1/output/research_adaptive_multitask_20261010/RESEARCH_DIRECTION.md`.

One geometry-preserving 8-bit RGB output goes through a real H.264/H.265 packet and independent decoder. Frozen Swin AR and detector/ByteTrack OT consume the same received RGB. No NeRV or receiver restoration.

Initial implementation uses a bounded finite action set of six region/band coefficients, source-only object/context/background maps with causal history and scene-cut reset, and a compact temporal controller. Actual codec candidates provide cached rate and per-task damage supervision. Missing task labels remain masked and provenance distinguishes GT from source consistency. Controller training does not differentiate through codec or association. Residual/proxy fine-tuning is a later experiment gated by evidence, not part of this initial version.

Two public Kaggle experiments: `ar_od` uses Kinetics/COCO; `ar_ot` uses Kinetics plus MOT20 sequence-disjoint TRAIN/CAL/DEV/TEST. MOT20 AR reports source consistency/predictions, never action accuracy. JRDB-Act remains a later joint-label validation, not silently claimed by these jobs. Each notebook trains seeds 303401/303402/303403 from the same real-codec cache and evaluates anchor/fixed filter/controller on both codecs. Counts are exploratory and recorded. TRAIN=48 and CAL=16 per image/video domain; DEV=16 and TEST=64. MOT sequence TRAIN/CAL=02, DEV=03, TEST=01, first256 frames, contiguous16-frame windows. CAL windows follow TRAIN windows and never tune model weights.

Quality selection allows gains at comparable bits, predicts per-task damage and rate, and calibrates error bounds on CAL. TEST labels never enter controller/support/selection. Identity remains feasible; forced policy and oracle are labeled separately. Actual packet bytes, model/code/source hashes, Q8 change rate, timing, selection rate and raw curves are stored. Basic BD only on valid overlap; report uncertainty or null for degenerate curves.

Dependency: `supervision==0.27.0` ByteTrack adapter, actual per-frame low-confidence detections, pinned TrackEval checkout. Checkpoint metadata must be JSON/native types with weights_only=True roundtrip. Source-only payload, no credentials or pretrained checkpoints included. Existing authorization covers public Kaggle code and eligible pool accounts; no monitoring requested.

Acceptance: real-codec integration, nonidentity Q8 action, scene-cut/no-future tests, missing-label and split leakage tests, deterministic candidates/serialization, same packet receiver checks, official OT metric integration and notebook payload validation. Passing tests establishes implementation correctness, not task improvement.
