# V32-VC: CNN preprocessing, virtual codec và retargeting

Nhánh thí nghiệm `v32-virtual-codec` triển khai yêu cầu ngày 2026-10-09. CNN là preprocessing trước encoder, không thay encoder/decoder chuẩn. Chưa có kết luận cải thiện AR/OD cho đến khi kết quả codec thật được thu về và kiểm tra.

## Luồng dữ liệu và gradient

```text
TRAIN: source -> residual CNN -> [fixed retarget] -> tensor virtual codec
                                            -> [inverse retarget] -> frozen task -> loss
Backprop: loss -> frozen task input -> virtual operations -> fixed resampling -> CNN
Evaluation: source -> frozen CNN -> [retarget] -> real H264/H265 bitstream + paid grid
                                            -> independent decoder -> [inverse] -> AR/OD
```

CNN width12, hai conv2D và một RGB head, thêm conv temporal depthwise causal cho AR. Head bắt đầu bằng identity, residual tối đa 0.1 trong RGB chuẩn hóa; P=1 giữ nguyên trước codec. Không còn 20 profile hay gate/expert/strength loss. Vùng ROI/motion lấy từ nguồn và frozen teacher, không lấy ground truth.

Virtual codec là phép toán tensor cố định: YUV420, DCT8, scalar quantization, inverse DCT, previous reconstructed-frame prediction. Train dùng uniform noise thay rounding; diagnostic eval dùng round. Không gọi FFmpeg trong minibatch và không dùng identity STE qua FFmpeg. Virtual codec không có neural weights; Gaussian entropy chỉ là rate surrogate, không phải byte bitstream.

64 nguồn TRAIN/task chia FIT48 và CAL16 ở cấp nguồn. CAL chọn scalar quantizer scale và fit affine rate theo byte codec thật, sau đó khóa. CNN plain và joint dùng cùng FIT, khởi tạo và lịch codec/QP; joint luân phiên plain/factor2/factor4. Bốn epoch, Adam1e-4, loss = task + .01*estimated_total_bpp + 10*MSE, LAST checkpoint. AR dùng CE của R3D18 và MC3-18; OD dùng classification/box/RPN loss của frozen MobileNet Faster R-CNN. Các task network không cập nhật trọng số hoặc batch statistics nhưng vẫn cho gradient tới ảnh đầu vào.

Các nhánh mới: `vc_plain`, `vc_joint_plain`, `vc_retarget_only` (nguồn nguyên bản, không simplification), `vc_joint`. Nhánh V32 cũ vẫn có candidate pool cũ. Guard 0.1 AR/0.03 OD và min saving1% giữ nguyên. Tất cả probe được đo; số probe không bằng nhau và không có claim K3. Header/lưới được tính vào rate theo pixel nguồn chuẩn hóa.

## Cách chạy

```powershell
python -m adaptive_vcm.v32vc.train --task ar --root <KineticsCleaned> --device cuda --out <empty_training_dir>
python -m adaptive_vcm.v32vc.train --task od --root <COCO_val2017> --annotations <instances_val2017.json> --device cuda --out <empty_training_dir>
python -m adaptive_vcm.v32vc.evaluate --task ar --root <KineticsCleaned> --split test --count 32 --bootstrap 0 --checkpoint <original_v30b_ar.pth> --v30-runtime <original_v30_runtime> --plain-checkpoint <training/plain_last.pth> --joint-checkpoint <training/joint_last.pth> --out <empty_evaluation_dir>
python -m scripts.v32vc_kaggle --task ar --checkpoint <original_v30b_ar.pth> --v30-runtime <original_v30_runtime> --out <empty_private_payload_dir>
```

OD evaluate bổ sung `--annotations`. Checkpoint loader kiểm tra task/arm, tensor và metadata SHA, TRAIN partition, code/config, cả FIT và CAL source ID/pixel hashes; không sửa provenance của V30-B để ép tương thích. Payload creator chỉ chuẩn bị notebook; thao tác submit được thực hiện riêng với quyền Kaggle đã cấp.

## Cách đọc kết quả

DEV32 là pilot; TEST32 giữ nguyên cấu hình, không chọn checkpoint theo DEV. R(2+1)D18 và ResNet50 là evaluator độc lập; R3D18 được ghi rõ là shared teacher. So sánh paired cùng nguồn, codec, QP và môi trường. New FIT48/CAL16 khác lịch TRAIN32 của checkpoint V30-B lịch sử: cần báo khác biệt này, không quy toàn bộ gain cho kiến trúc.

Giới hạn surrogate: chưa mô phỏng motion search/B frames, adaptive transform và entropy coder thật. DCT/entropy có thể cho gradient sai hướng; báo calibration MSE, rate error và edit sign agreement. Transfer được quyết định bằng encode/decode thật, Top-1/mAP, cubic/PCHIP BD-rate trên miền overlap và quality gap từng QP. TEST32 vẫn là mẫu nhỏ, bootstrap0 chưa có CI, không tự xác nhận mục tiêu -10%.

Tài liệu tham khảo: [Zhao et al. virtual codec](https://arxiv.org/html/2512.15331v1), [ROI retargeting](https://link.springer.com/article/10.1186/s13640-025-00682-3). Đây là biến thể đơn giản phỏng theo ý tưởng, không phải bản tái lập đầy đủ hai bài báo.
