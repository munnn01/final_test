# V34 R3: phương pháp chung, checkpoint và đánh giá riêng AR/OD/OT

Ngày: 16/10/2026. Trạng thái: **đặc tả để người dùng duyệt; chưa triển khai R3**.

## 1. Phạm vi đã được duyệt

Làm rõ mới nhất của người dùng: **“Cùng thuật toán/kiến trúc, checkpoint riêng mỗi task; giữ nguyên AR”**. Yêu cầu trước đó giữ nguyên pipeline ghép R2 vẫn áp dụng. Lựa chọn này thay thế cách diễn đạt ban đầu chỉ có hai experiment OD/OT trong R3.

Một framework thực hiện cùng chuỗi: nguồn → support maps theo task → cùng 13 profile lọc → codec thật → CNN+GRU dự đoán action/risk/rate → probe/guard → chọn packet → đánh giá task. Framework có experiment `ar`, `od`, `ot`; mỗi experiment có checkpoint, cache/calibration và kết quả riêng. Mỗi lần chỉ kích hoạt một task. Không gộp loss hoặc metric ba task thành một điểm duy nhất, không dùng một checkpoint đa tác vụ và không yêu cầu một packet đồng thời phục vụ ba task.

AR dùng recipe và checkpoint đã khóa, không huấn luyện lại và không đổi quyết định chọn packet. Target/metric adapters và các ràng buộc có cấu hình theo task; việc có task-specific loss/threshold không biến chúng thành ba phương pháp preprocessing không liên quan. Phần cải tiến được huấn luyện mới trong đợt này chỉ áp dụng OD/OT. Không gọi kết quả AR cũ là kết quả một AR controller được huấn luyện bằng targets R3.

Giữ nguyên toàn bộ experiment ghép `ar_od_ot` của R2, gồm preprocessing, controller, các packet đã chọn và job Kaggle đang chạy. Launcher mới từ chối `ar_od_ot`, từ chối TRAIN/CAL lại AR và không thay routing mặc định của script R1/R2. Đưa thay đổi này vào bitstream ghép chung nằm ngoài đặc tả.

Mục tiêu thực nghiệm là cải thiện trade-off số byte–COCO mAP của OD và số byte–official HOTA của OT. R3 không có mục tiêu ép đường OT thành đơn điệu hoặc tạo bằng được một BD-rate hữu hạn.

## 2. Chẩn đoán có bằng chứng

Mốc mã đối chứng là `c445ef3c690383e6a593c24043fca6d8a3f0cdaa`. Các artifact ở `output/v34_r2_continue_20261016/workflow_private/artifacts` là kết quả lịch sử, không phải tập xác nhận độc lập cho R3.

### 2.1. Vì sao OT/H.264 bị từ chối BD-rate

Trên TEST ghép R2, đường identity đã có đảo chiều:

| QP | bpp | HOTA | DetA | AssA |
|---|---:|---:|---:|---:|
| 30 | 0.06051829 | 37.1566 | 37.4806 | 37.8948 |
| 35 | 0.03424910 | 37.5082 | 35.8376 | 40.5089 |

Ít bit hơn đi kèm DetA thấp hơn nhưng AssA cao hơn. Bộ kiểm tra `tracking_curve` từ chối tính BD-rate cho cặp đường không đơn điệu. Đây là lý do trực tiếp đã xác minh; chưa quy được thay đổi association cho một ngưỡng/detection cụ thể. OT/H.265 TEST và OT/H.264 DEV có các đường hợp lệ trong cùng giao thức, nên không thể nói mọi đường OT đều không hợp lệ.

### 2.2. Điểm yếu cần thử sửa

- `v34r2/evaluation.py::shortlist` xếp hạng bằng logits và source feasibility. `choose` quyết định theo byte thật và source-consistency guard; calibrated GT risk không tham gia quyết định cuối. Đây là hành vi thiết kế R2, không phải lỗi quên nối một hàm đã được chứng minh hữu ích.
- Trong cache TRAIN ghép, 2.791/7.920 ứng viên OD nonidentity qua source guard vẫn vượt `damage_tolerance` của GT training-loss target. CAL có 986/2.774 trường hợp. Đây là thống kê trên ứng viên và surrogate, không phải tỷ lệ giảm mAP của packet triển khai.
- OD dùng detector training loss làm GT target; phép đánh giá cuối là COCO mAP từ inference. Chúng không đồng nhất.
- OT dùng finite-context HOTA chỉ có pedestrian positives làm GT target, nhưng source guard dùng local FN/FP/ID surrogate và TEST dùng full-stream official MOT preprocessing. Cần tách rõ cả ba phép đo.
- Trong cache ghép, cả 458 ứng viên OT qua source guard ở TRAIN và 410 ở CAL đều nằm trong GT tolerance hiện tại `.05` (tối đa 5 điểm context HOTA). Vì thế điều kiện GT này không loại thêm ứng viên nào trong chính tập đó. Đây là dấu hiệu cần kiểm độ phân biệt của target/tolerance, không phải bằng chứng rằng chúng an toàn ở full-sequence HOTA.
- TRAIN/CAL OT đo ứng viên trên identity prefix; lúc triển khai dùng selected prefix. R3 cần kiểm tra và giảm sự lệch trạng thái này.
- OT standalone DEV dùng cùng shortlist có thứ tự `(6, 11, 4)` ở toàn bộ 64 điều kiện của mỗi codec cho seed 303401. Shortlist đã chứa ứng viên tối ưu theo source guard trong toàn bộ chẩn đoán DEV này. Vì thế tăng ngân sách shortlist đơn thuần chưa có căn cứ; cần kiểm tra target/guard và đóng góp thật của CNN.
- Giao thức OT hiện tại có một sequence TRAIN và một sequence CAL, mỗi phần 256 frame. Các window hoặc tám điều kiện codec×QP không phải sequence độc lập.

## 3. Cơ sở phương pháp và giới hạn chuyển giao

- [HOTA, IJCV 2021](https://arxiv.org/html/2009.07736v2): HOTA xét detection và association trên trajectory. Dùng DetA/AssA để chẩn đoán và làm target phụ; không gọi metric cục bộ là full-sequence HOTA.
- [ByteTrack, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf): low-score detections có ích khi được liên kết với track; không bảo vệ mọi box thấp hoặc hạ threshold tùy ý. R2 đã áp dụng nguyên tắc này, R3 không nhận đó là đóng góp mới.
- [AccMPEG, MLSys 2022](https://proceedings.mlsys.org/paper_files/paper/2022/file/853f7b3615411c82a2ae439ab8c4c96e-Paper.pdf): ngữ cảnh ngoài box có thể quan trọng với tác vụ. Giữ soft context của R2 trong thí nghiệm đầu; paper không bảo đảm COCO mAP cho bộ 13 filter hiện tại.
- [Task-Aware Encoder Control, CVPR 2024](https://arxiv.org/html/2404.04848v1): động cơ học từ rate và task objective với task network cố định. Neural-codec control của bài khác x264/x265 và không cung cấp mức gain dự báo cho R3.
- [The Bjøntegaard Bible, IEEE TIP 2024](https://arxiv.org/pdf/2304.12852): cần xét raw points, phần giao chất lượng và độ nhạy nội suy. Không bỏ QP, làm trơn/reorder chất lượng hoặc đổi metric sau khi thấy TEST để tạo BD-rate thuận lợi.

Các thay đổi dưới đây là giả thuyết thực nghiệm của dự án, không phải recipe đã được các paper trên xác nhận cho V34.

## 4. Kiến trúc cô lập

Thêm namespace `adaptive_vcm.v34r3`, cấu hình điều phối chung `configs/v34_r3_qp4.json` và launcher `scripts/v34r3_kaggle.py` dùng `--experiment ar|od|ot`. Cấu hình có task recipe riêng; recipe AR trỏ tới config AR cũ và manifest checkpoint đã khóa, không truyền config R3 thay cho config mà checkpoint AR yêu cầu. Không sửa hành vi hoặc source của `adaptive_vcm/v34`, `adaptive_vcm/v34r2`, các config/script R1/R2.

Chỉ tái sử dụng các tiện ích bất biến của bản đối chứng: 13 action với cùng thứ tự/hệ số, filter giữ hình học, codec thật, frozen task-model weights, đọc nguồn, ghi packet và official evaluators. Không monkey-patch module globals của R1/R2. Nếu một hàm cũ không có điểm mở phù hợp, adapter R3 sở hữu luồng gọi mới; không sửa hàm cũ để phục vụ R3.

**Kiến trúc mạng chung:** giữ nguyên topology `v34.model.Controller`: spatial CNN, GRU 24 hidden units, MLP và 13×5 đầu ra gồm logits, ba risk slots, rate. Cùng 9 điều kiện codec/QP/task mask/budget/domain như recipe AR QP4 có domain condition. OD/OT không bổ sung một backbone, feasible head hoặc một cặp GT/source risk head chỉ dành cho hai task. Source costs được đo trực tiếp trên decoded probes, không cần một learned source-risk head.

AR nạp nguyên state dict cũ bằng loader/provenance cũ. OD/OT huấn luyện ba controller riêng với seeds 303401/303402/303403 theo cùng topology. Không chia sẻ tham số hoặc optimizer với AR/joint R2. Ba risk slots có ý nghĩa do metric adapter đăng ký: AR giữ nguyên ý nghĩa và mask cũ; OD/OT dùng ba thành phần nêu ở mục 6–7. `task_mask` chọn task và `component_mask` chọn thành phần target là hai khái niệm riêng, không đổi task mask để đánh dấu metric slots.

**Thuật toán chung:** condition task → tạo support → dự đoán/xếp hạng → thử cùng ngân sách identity+3 → áp ràng buộc → chọn packet thực. Metric/support/constraint adapters là các tham số theo task của thuật toán này. Các chi tiết AR đang ảnh hưởng kết quả, gồm proposal order, tie-break và trạng thái bật/tắt gate, giữ nguyên recipe hiện tại. Compatibility adapter AR phải gọi hành vi đã khóa, không được thay bằng một selector mới rồi giả định checkpoint cũ đủ giữ kết quả. Vì vậy tuyên bố khoa học là một framework task-conditioned với kiến trúc chung và các recipe theo task, không phải cả ba task đã chạy cùng một recipe R3 mới hoặc có trọng số giống nhau.

Các đơn vị chức năng:

1. **Protocol/provenance:** kiểm cấu hình, experiment, cohort, checkpoint, prefix và cache; từ chối trộn R2/R3.
2. **Targets:** đo GT targets và source-reference costs riêng; mọi bảng đo codec là stop-gradient.
3. **Temporal measurement:** tạo và replay prefix theo policy đã khóa; candidate probes là transaction độc lập.
4. **Learning/calibration:** huấn luyện ranking/risk/rate; CAL không cập nhật trọng số.
5. **Selection:** cùng giao diện action/risk/rate, các ràng buộc task-specific; OD/OT thử risk gate đã kiểm định cùng source costs thực, AR giữ recipe guard cũ. Không nhận GT.
6. **Evaluation/launcher:** task adapter AR accuracy / OD COCO / OT TrackEval sau khi chọn packet, checkpoint và báo cáo riêng. AR chạy evaluation-only hoặc nhập lại artifact đã kiểm provenance với nhãn reused frozen reference; không gắn nhãn fresh run cho kết quả nhập lại.

## 5. Cấu hình cố định và thứ tự ablation

Giữ QP `{30,35,40,45}`, x264/x265, preset/GOP/fps/độ phân giải và 13 profile của recipe task tương ứng. Ngân sách triển khai vẫn là identity + 3 ứng viên nonidentity; các đối chứng mới OD/OT dùng cùng ngân sách TRAIN-fixed/random. AR giữ nguyên bộ đối chứng đã khóa và không phải chạy lại TRAIN để tạo một fixed shortlist mới. Giữ frozen task models/tracker và không tinh chỉnh threshold của receiver trong đợt này.

Đợt đầu dùng **đúng support maps R2** để cô lập tác dụng target, prefix và selection. Không đồng thời đổi halo, action coefficients, backbone, retargeting hoặc neural codec. Cải tiến support/mask là đợt ablation sau, chỉ mở nếu các chẩn đoán hiện tại chỉ ra đó là nút thắt và được người dùng duyệt phạm vi bổ sung.

Thứ tự thử:

1. Đối chứng R2 gọi runner/checkpoint R2 đã khóa để kiểm provenance và tái lập packet/action trên fixture và một cohort chẩn đoán. Không nạp checkpoint R2 có head khác vào Controller R3; đối chứng được ghi rõ là R2. Song song đối chiếu AR qua entrypoint chung với entrypoint AR cũ.
2. Thay targets, giữ identity-prefix và selection R2 làm đối chứng.
3. Thêm đa dạng prefix TRAIN, giữ target mới.
4. Thêm risk gate đã kiểm định; so với source-only gate cùng ngân sách probe.

Chỉ ghép các thành phần có lợi trên DEV. Cache packets/predictions có thể được chia sẻ giữa các chẩn đoán cùng pixel/prefix/codec, nhưng không dùng lại một target hoặc tracker state không đúng nguồn gốc.

## 6. OD: targets sát inference hơn

Giữ detector training-loss increment như target phụ/đối chứng. Target chính thử nghiệm được tạo từ **inference detections của frozen detector** và GT TRAIN/CAL, với các thành phần tách biệt:

- Miss/FN: số GT không được ghép đúng lớp và IoU.
- FP: detection không ghép được, gồm sai lớp.
- Localization: phần thiếu IoU của cặp ghép đúng lớp.

Sử dụng per-image matching và `gtIgnore`/`dtIgnore` của COCOeval trên lưới IoU `.50,.55,...,.95`, category-aware, `area=all`, `maxDets=100`, với prediction score cutoff `.10` được ghi trong config cho surrogate này. Không tự viết một thuật toán matching khác rồi coi là COCO matching. FN chỉ đếm GT không ignored chưa ghép; FP chỉ đếm detection không ignored chưa ghép; localization dùng các cặp hợp lệ, loại crowd/ignored. FN và FP chuẩn hóa theo `max(số GT không ignored,1)`; localization chuẩn hóa theo số cặp, bằng 0 nếu không có cặp, trong khi FN/FP vẫn phạt các vật thể không ghép được. Lấy trung bình theo lưới IoU, lưu từng thành phần và increment so với identity cùng điều kiện. Empty-GT/empty-prediction phải có định nghĩa và test riêng. Annotation gốc phục vụ crowd/ignore được truyền tường minh chỉ vào luồng đo GT offline.

Các giá trị này là **inference-error surrogates**, không phải COCO mAP và không phải bảo đảm giữ mAP. Không thay phép chấm cuối bằng trung bình AP từng ảnh. COCO mAP@[.50:.95], AP50/AP75 và AP-small/medium/large vẫn chấm trên cohort đầy đủ.

Source-reference costs không dùng GT: dùng dự đoán nguồn làm reference, giữ các thành phần miss/extra/localization/confidence có tên và đơn vị riêng. Không đồng nhất false positive của nguồn với một đối tượng thật. Guard mới được đối chiếu với guard R2 bằng tỷ lệ vi phạm GT target trên CAL/DEV và mAP của cohort.

## 7. OT: tách detection/association và giảm lệch prefix

GT target R3 gồm increment của `1-HOTA`, `1-DetA`, `1-AssA` trên context causal liên tiếp. Số đo nằm trong [0,1]; một delta 0.005 tương ứng 0.5 điểm của **context metric đó**, không phải 0.5 điểm full-sequence HOTA được bảo đảm.

GT context đi qua MOTChallenge preprocessing của evaluator đã pin, gồm xử lý distractor/ignored annotations phù hợp. Không tiếp tục dùng positives-only rồi gắn nhãn là official MOT target. TRAIN/CAL có quyền đọc GT; selector runtime không có. Phép so với clean source tracks được ghi là source-reference tracking consistency, không phải GT HOTA.

Source-only guard bổ sung các delta riêng về detection và association so với identity candidate trên **cùng frozen current prefix** và cùng context. Một thành phần tốt lên không được che việc thành phần khác vượt ngưỡng. Giữ local tracking cost R2 làm đối chứng và diagnostic.

Prefix measurement có hai lượt TRAIN được đánh dấu rõ:

1. Cache identity-prefix như đối chứng; huấn luyện warm controller TRAIN-only.
2. Đóng băng warm controller và policy source-only; rollout liên tục trên TRAIN, không dùng GT để commit prefix; đo lại 13 ứng viên tại các prefix đã rollout. Mọi ứng viên ở cùng một điều kiện bắt đầu từ cùng snapshot. Controller cuối học trên hỗn hợp identity-prefix và rollout-prefix với trọng số hai loại bằng nhau.

Warm policy/checkpoint hash và toàn bộ chuỗi commit action/prefix được lưu. CAL dùng policy warm đã khóa để đo targets; sau khi calibration được khóa, DEV chạy policy cuối với lịch sử thật. Đây là biện pháp giảm lệch trạng thái, không phải chứng minh loại bỏ hoàn toàn off-policy shift.

Không reset tracker mỗi GOP; không dùng future frames. Scene cut/reset, hết context, mất track và re-entry qua ranh GOP phải có test. Đánh giá cuối vẫn chạy official TrackEval trên toàn đoạn liên tục đã đăng ký của mỗi sequence, không trung bình HOTA của các window.

## 8. Học, kiểm định risk và chọn packet

Controller OD/OT dùng cùng logits, ba risk slots và rate của kiến trúc AR. Logits phục vụ ranking; feasibility được tính từ risk bounds và phép đo guard, không có một neural feasibility head riêng. Risk slots học GT-component increments; source costs đo trực tiếp và lưu riêng, không trộn vào GT target. Scale của từng thành phần chỉ fit trên TRAIN OD/OT; không refit scale AR. Metadata nêu rõ thứ tự, đơn vị và scale của từng output; checkpoint không tương thích phải bị từ chối.

TRAIN ranking ưu tiên ứng viên ít byte trong tập thỏa đồng thời các giới hạn thành phần. Dùng soft ranking như R2 làm cơ sở; chưa thêm một loss pairwise hoặc một backbone khác trong đợt đầu. Lưu mức đa dạng shortlist, tỷ lệ identity, entropy, sai số từng head và chênh lệch với TRAIN-fixed/random để phát hiện policy gần như hằng.

CAL residual được tổng hợp theo nguồn độc lập: ảnh OD và sequence OT; các row QP/codec/window cùng nguồn không được tính là mẫu độc lập. Báo số group hiệu dụng. Dùng empirical residual bounds với phạm vi áp dụng được ghi rõ; không quảng bá bảo đảm conformal/coverage 90% khi có quá ít sequence hoặc có distribution shift. Bound không hữu hạn, thiếu miền/thành phần hoặc thiếu provenance dẫn tới identity fallback.

Quy trình runtime OD/OT trong framework chung (AR dùng recipe tương thích đã khóa):

1. CNN đề xuất ba ứng viên nonidentity từ input nguồn, codec và QP.
2. Encode/decode identity và ba ứng viên; tính toàn bộ chi phí triển khai.
3. Tính source costs trên packet thật. Đối chiếu dự báo risk đã hiệu chỉnh CAL với giới hạn tương ứng, không truy cập GT.
4. Chọn packet ít byte nhất trong tập qua cả risk gate, source guard và budget; tie-break cố định theo action index. Identity luôn là phương án dự phòng hợp lệ.
5. Commit đúng một selected tracker probe; receiver độc lập nhận đúng packet được chọn. Probe bị loại không được đổi prefix hoặc thứ tự kết quả ứng viên khác.

Risk gate chỉ được nâng thành cấu hình chính khi giảm vi phạm trên DEV và cải thiện trade-off so với source-only gate. Nếu nó chỉ trả về identity hoặc không vượt đối chứng, báo đúng thất bại/thỏa hiệp, không gọi đó là cải tiến tiết kiệm bit.

Giới hạn thành phần được chọn bằng grid đã khai báo trước chạy DEV: OD normalized error increments `{0,.01,.025,.05}`; OT context/source-reference quality drops `{0,.005,.01,.025}`. Chỉ thử các vector dùng cùng một mức trong mỗi nhóm để giới hạn số tổ hợp. Khóa một cấu hình cho mỗi task bằng DEV rồi mới chấm cohort xác nhận. Các giá trị này là lựa chọn kỹ thuật, không phải hằng số từ paper hoặc từ TEST đã xem.

## 9. Dữ liệu, đánh giá và điều kiện chấp nhận

- R2 lịch sử giữ nguyên để đối chiếu. TEST đã được xem chỉ dùng báo cáo exploratory, không làm nguồn chọn threshold hoặc tuyên bố xác nhận độc lập cho R3.
- Manifest R3 phải liệt kê split, source IDs, pixel hashes, sequence/camera identity và lịch sử sử dụng. OD tăng đa dạng ảnh khi dữ liệu sẵn có; OT cần thêm sequence độc lập. Các bản detector khác nhau của cùng sequence MOT17 phải cùng group/split.
- Trước bất kỳ tuyên bố xác nhận nào, chọn và khóa một cohort chưa tham gia quyết định thiết kế, sau khi kiểm lịch sử manifest. Nếu không có cohort như vậy, báo exploratory; không tự gắn nhãn untouched.
- R3 có thể kiểm kỹ thuật trên cohort lịch sử, nhưng không được gọi một CAL sequence là bằng chứng coverage. Quyết định cấu hình chính phải có ít nhất hai CAL sequence độc lập và báo rõ vẫn chưa có bảo đảm thống kê mạnh.
- Chấm ba seed, cả hai codec, đúng bốn QP. Không chọn seed tốt nhất để báo chung. Bootstrap OD theo nguồn độc lập; chỉ tính OT sequence CI khi có đủ sequence, không bootstrap frame như quan sát độc lập.
- Báo byte/bpp gồm header, raw same-QP metrics, worst-QP gap, DetA/AssA/IDF1/MOTA/FN/FP/IDs cho OT; tổng sender/guard/receiver work và tỷ lệ fallback.
- Giữ cubic BD lịch sử và PCHIP sensitivity với nhãn riêng. Kiểm nonmonotonicity, overlap, duplicate/flat points và chênh lệch nội suy. Kết quả cực lớn/nhạy không được gắn nhãn reliable chỉ vì các điểm đơn điệu.
- Ngưỡng sàng lọc cuối tiếp tục dùng worst-QP quality gap không thấp hơn −0.5 điểm và tiết kiệm byte; đây là tiêu chí thực nghiệm theo cohort, không phải bảo đảm từ source guard. BD-rate chỉ có vai trò khi cặp đường và nội suy đạt kiểm tra hợp lệ. Nếu không, báo trade-off raw points và không kết luận mục tiêu BD đã đạt.
- CNN phải được so với TRAIN-fixed/random cùng ngân sách. Nếu không vượt fixed thì không khẳng định đóng góp riêng của controller. Identity bảo toàn baseline nhưng không đạt mục tiêu tiết kiệm.

## 10. Kiểm thử và bảo vệ AR/R2

Các kiểm tra bắt buộc trước khi chạy thật:

1. Launcher/framework chấp nhận evaluation `ar`/`od`/`ot` riêng, từ chối `ar_od_ot` và từ chối retrain/refit AR. Mạng OD/OT có cùng tên/shape tham số và forward topology với Controller AR domain-conditioned; task-checkpoint/target metadata sai phải bị từ chối dù tensor shapes trùng.
2. Hash manifest của toàn bộ file R1/R2/config/script liên quan AR/joint và checkpoint/config/calibration AR không thay đổi so với nguồn đối chứng đã khóa. Chạy AR qua entrypoint chung và entrypoint cũ, đối chiếu proposal order, action, packet/decoded hashes và dự đoán trên cùng fixture/môi trường. Nêu rõ phạm vi fixture; kiểm cohort thật nếu rerun và ghi nhãn reused reference nếu chỉ nhập artifact cũ. Không thay bằng chứng dự đoán bằng việc chỉ so hash mã.
3. Frozen task weight hashes khớp; không optimizer nào sở hữu tham số AR/task head.
4. OD matching đúng ở empty scenes, FP, FN, sai lớp, IoU, crowd/ignore và nhiều box chồng lấn; COCO evaluator cuối không đổi.
5. OT candidate isolation, probe-order invariance, causal prefix replay, scene reset, context cap và cross-GOP association; đối chiếu context preprocessing với official dataset API.
6. Không có GT trong runtime source/selector interface; annotation chỉ được đọc ở bước target offline hoặc scoring sau commit. Cố tình đưa sentinel GT để phát hiện rò rỉ trong test.
7. Calibration fit-split/grouping, scale TRAIN-only, missing/nonfinite risk fallback, identity invariant, mọi ứng viên bị loại vẫn cho kết quả xác định.
8. Fixture smoke hoàn chỉnh QP4×2 codec và đủ ba controller, bằng codec thật; không dùng fixture để tuyên bố gain chất lượng.
9. Kiểm nguồn đóng gói Kaggle, dữ liệu và output riêng; giữ notebook/dataset mới private. Không hủy, restart hoặc đổi source HEAD mà dispatcher R2 đang kiểm.

Không sửa artifact R2 cũ để khớp schema R3. Báo rõ kiểm tra nào đã chạy và phạm vi không được kiểm.

## 11. Triển khai và điểm dừng

Hiện mới có chẩn đoán và đặc tả. Sau khi người dùng duyệt **bản đặc tả này**, viết implementation plan rồi để người dùng chọn cách thực hiện theo workflow. Sau khi code, kiểm thử và review đạt, đẩy lên repository `munnn01/final_test` theo ủy quyền trước đó. Chuẩn bị/chạy OD/OT riêng; AR chỉ evaluation/provenance check bằng recipe và checkpoint đã khóa, tuyệt đối không retrain. Dùng tài khoản đã được phép và tính tới quota/job đang dùng account.

Không tự khởi chạy R3 trước khi có code đã kiểm và plan đã duyệt. Không triển khai R3 vào pipeline ghép. Không hứa BD-rate sẽ hợp lệ hoặc chất lượng tăng trước số đo thật. Nếu DEV không ủng hộ risk gate/targets mới, giữ R2 làm đối chứng, báo kết quả và đề xuất bước tiếp theo có bằng chứng.
