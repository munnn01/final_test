# Final independent review

Reviewed range:28cc97db6708c87346c025107a8a7ea485c21163..92b968bfa32d8a3515bbecb1238cd43830f61eb2. A fresh senior reviewer on GPT-6-astra reviewed the whole branch read-only. No Critical findings. Exact actual-forward/proxy-backward values, frozen teacher hooks, original warm provenance, forced unknown-foreground candidates and same-stream restoration pairing were judged coherent.

Important finding: RF checkpoint loading omitted mandatory codec execution and real measured bitrate checks. The original loader accepted missing or contradictory GOP/preset metadata and missing actual-bitrate evidence. The fix requires exact primary execution settings; positive finite measured bpp and codec time for each real epoch; absent measured rate and zero actual-codec time in each proxy epoch; identical execution settings across pairs. Final regression verification is recorded in V32_RF_EXECUTION.md.

Deferred minor: payload failure-archive testing inspects runtime source strings rather than executing a controlled failing runtime and opening the resulting archive. Catchable failures have status/traceback/finally archive code; the behavior test remains future work. No observed runtime defect was reported.

Final rulings on the reviewer's declined scope:

- Performance, useful surrogate gradients, BD-rate gains and GPU runtime await returned evidence. No improvement is claimed; cost: the experiments may be inconclusive.
- Frozen Swin features plus a trainable RGB head and the same 16-frame evaluation preserve the controlled budget and source recipe. Full backbone finetuning and official 32-frame multi-clip evaluation are subsequent work; cost: this may miss a stronger receiver or produce different absolute accuracy from the published benchmark.
- Python finally archives catchable failures; an uncatchable platform kill or quota termination can prevent the final archive. Cost: partial evidence may need collection from surviving output files.
- Adaptive scales, ROI-QP, learned proxies, multitask training and AV1 remain deferred until the first controlled comparison returns evidence. Cost: potential gains from those directions are not measured in this run.

The experimental branch is retained locally and deployed as frozen public Kaggle notebooks under existing authorization; no merge or GitHub push is part of the task.
