# Evidence behind the R2 experimental design

This is a bounded literature-informed design, not a reproduction of these
papers or a prediction of their reported gains. See [the executable recipe](V34_R2.md)
for the exact implementation, ablations and empirical limitations.

- **OD context:** AccMPEG shows that task importance extends beyond object
  boxes and uses offline task-loss supervision for an importance predictor.
  This motivates testing a soft context halo. Its modified x264 block-QP
  control and source-reference F1 evaluation differ from R2's fixed-QP
  filters and COCO mAP. [Du et al., MLSys 2022](https://proceedings.mlsys.org/paper_files/paper/2022/file/853f7b3615411c82a2ae439ab8c4c96e-Paper.pdf).
- **OT support:** ByteTrack recovers low-confidence detections through
  association. R2 therefore protects associated low-score boxes rather than
  accepting every low-score detection. [Zhang et al., ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf).
  Simply lowering thresholds can increase false positives; temporal validity
  needs its own ablation. [Vaquero et al., ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09351.pdf).
- **Tracking targets:** HOTA measures both detection and association using
  trajectory-wide information. R2's finite-context HOTA is an offline target;
  final evaluation uses complete continuous streams and reports DetA/AssA.
  A local ID-switch cost is not renamed HOTA. [Luiten et al., IJCV 2021](https://link.springer.com/article/10.1007/s11263-020-01375-2).
- **Task-aware supervision:** Frozen task networks and rate/task objectives
  have precedent in task-aware video coding. The cited method controls a
  neural codec; R2 instead measures real x264/x265 outputs and keeps the codec
  outside backpropagation. [Ge et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ge_Task-Aware_Encoder_Control_for_Deep_Video_Compression_CVPR_2024_paper.html).
- **Source versus GT:** Annotation-free optimization motivates retaining
  source-derived targets, but consistency with a frozen source model does not
  establish GT accuracy. R2 stores the two measurements separately.
  [Windsheimer et al., ICIP 2024, author version](https://arxiv.org/html/2406.07938v1).
- **Rate/quality reporting:** Raw operating points, monotonicity, overlap and
  interpolation sensitivity matter for Bjøntegaard results. R2 applies the
  same validity rules to primary comparisons and every bootstrap draw.
  [Herglotz et al., IEEE TIP 2024, author version](https://arxiv.org/pdf/2304.12852).

The 64-frame context, halo size, decay, score thresholds and ranking
temperature are locked engineering hypotheses. Their quality must be tested
on TRAIN/CAL/DEV and then assessed on an untouched confirmation dataset.
Keeping AR/R1 code and checkpoints unchanged is an implementation isolation
requirement; these papers do not guarantee that a newly trained joint stream
will preserve AR accuracy.
