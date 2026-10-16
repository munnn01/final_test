# Continuing the interrupted R2 experiments

The 16 October OT and joint jobs exhausted Kaggle's 43,200-second session. Their saved artifacts are reused under the original QP4/three-seed protocol. This change does not modify the R1 implementation or any task-quality/BD definition.

## Execution boundaries

- OT: load the three existing controllers. Evaluate DEV and TEST separately for each codec (four jobs).
- Joint: reuse the saved 4,248 condition rows; regenerate controller inputs, restore identity target prefixes, measure the 104 missing conditions (13 CAL windows), then train the original three controllers once.
- Joint evaluation: split by domain and codec; MOT DEV/TEST are separate jobs. All three source constraints still apply to every joint video packet.

Every OT evaluation starts at frame one of its full sequence. The interrupted run did not persist the mutable tracker or complete DEV exhaustive diagnostics. Reevaluating the complete sequence preserves causal tracker state and full diagnostics; it does not repeat TRAIN/CAL measurement or OT training. No temporal fragments are merged into an artificial sequence.

`evaluation.evaluate(..., splits=..., codecs=..., domains=...)` changes execution scope only. Model settings and calibration remain intact. Scoped report aggregation uses only executed codecs, while the manifest retains the complete experiment settings and ordered checkpoint hashes. A scope's `stage_complete.json` is not a full benchmark completion marker.

## Inherited state

The continuation verifies inherited artifact bytes, cache integrity, task-head hashes, full source-cohort pixel hashes, action bank and settings. Scientific source files must match the inherited snapshot; the runner/evaluator orchestration files are the explicit exceptions for this continuation. Both training origin and executing origin are retained. Cached tracking anchors are re-encoded and must match packet/decoded/prefix hashes before their identity prefix can advance.

The interrupted parent kernel versions `/1` remain provenance references. Kaggle rejected those cancelled notebooks as input sources, even while reporting a successful notebook push. The workflow therefore copies the locally recovered state into private, content-checked datasets owned by each execution account. The trained joint state uses the same transport. No raw images/video or credentials enter those state datasets. Dataset version 1 is never overwritten by this workflow. Rejected input-attachment warnings stop dispatch even if the CLI reports a successful push.

## Run the durable workflow

From the clean committed repository:

```powershell
python -m scripts.v34r2_continue_workflow `
  --pool <local-account-pool.json> `
  --results-root <downloaded-v34-r2-results-directory> `
  --out <continuation-workflow-directory> --run
```

Without `--run`, dispatch/check one bounded iteration. Every submission/upload writes its intent before the remote call. Ambiguous outcomes are never blindly retried. A failed dependency cannot start its descendants. The workflow verifies stage manifests before merging non-overlapping output scopes and writing the final completion marker.

This dispatcher runs locally; the host must remain running to dispatch downstream stages. Submitted GPU work itself runs on Kaggle. An `attention.json` or `needs_attention` state requires diagnosis before restarting; existing accepted notebooks are preserved.

Use a new workflow directory for the dataset-based relaunch after the initial rejected-input attempt. Its plan records the original parent references, recovered file paths/hashes, new immutable source commit and each private dataset receipt separately.
