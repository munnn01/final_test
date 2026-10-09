# V33-AP execution ledger

Plan: docs/superpowers/plans/2026-10-10-v33-joint-adaptive-preprocessing.md
Spec: docs/superpowers/specs/2026-10-10-v33-joint-adaptive-preprocessing-design.md
Base: 1e7e1ed; native implementation approved by the user.

Ruling: Reuse the existing dedicated experimental checkout/branch and native Windows bookkeeping, as approved in the plan and previous campaign; no new worktree or Bash scratch helper — preserves the requested execution method — cost if wrong: move the recorded commits to another checkout and translate the ledger.

Pre-flight: Task1 model/protocol feed Task2 sources and Task3 training; Task2 record/head interfaces feed Task3 training and Task4 receiver; Task3 strict checkpoint feeds Task4 evaluation; Task3/4 CLI and metadata feed Task5 payload. No conflicting interfaces. Historical V32 selector requires two AR teachers, so Task4 reuses only lower-level guard primitives.

Task1: completed. Baseline RF bridge/VC model:22 passed. RED:8 intended missing-module failures. GREEN:31 passed,6.38s,exit0 (new model/protocol + VC model/real codec). Corrected a test fixture: tanh(10) saturates in float32, so the gradient check uses bias0.5 while the bound check retains10. No product change was needed for this fixture correction.

Task2: completed. RED:5 missing-module failures. GREEN:new sources/tasks + RF bridge/VC training,21 passed,16.32s,exit0. Separate AR and OD gradients reach the same head; teachers have no gradients, BN/modes restored, empty pseudo boxes finite, COCO does not invoke action. Canonical pixels and explicit repetition/effective letterbox scales verified.

Task3: completed. RED:4 missing-module failures. GREEN:new training/checkpoint + RF training/bridge/VC training,38 passed,17.28s,exit0. Both actual codecs checked against independent packets and paid bpp; proxy backward reaches CNN. 960 unique source/codec/QP exposures per epoch; validation diagnostics preserve state. Rehashed split tampering and state corruption rejected.
