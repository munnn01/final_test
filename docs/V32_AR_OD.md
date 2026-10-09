# V32 AR/OD: paired DEV32 trên Kaggle, 2026-10-09

Hai notebook riêng tư phiên bản1 đã COMPLETE trên T4:

- [Action Recognition](https://www.kaggle.com/code/vtk269/v32-ar-dev32-frozen-s303201): KineticsCleaned,32 clip DEV,16 khung hình128×128/clip. Teacher R3D18 và MC3-18; evaluator độc lập R(2+1)D18, kèm R3D18 để đối chiếu.
- [Object Detection](https://www.kaggle.com/code/vtk269/v32-od-dev32-frozen-s303201): COCO2017 val,32 ảnh DEV chuẩn hóa320×320. Teacher MobileNet; evaluator độc lập Faster R-CNN ResNet50; COCO mAP@[.5:.95].

Mỗi task có7 nhánh,5 QP30/35/40/45/50 và H.264/H.265. V30-B dùng checkpoint task tương ứng đã đóng băng, nạp qua runtime nguyên gốc commit59ebb3dc31fc8edd6e1fbb80ed7de508f979edd3. Không huấn luyện thêm hay điều chỉnh guard sau khi xem kết quả. Mọi nhánh được chạy trên cùng nguồn và môi trường.

| Evaluator | Codec | BD-rate cubic V32 so với V30-B (%) | PCHIP (%) | Giảm byte cùng QP, min–max (%) | Đổi chất lượng cùng QP, min–max (điểm %) |
| --- | --- | ---: | ---: | ---: | ---: |
| AR R(2+1)D18 | H.264 | +3.90 | +2.81 | 0.00–0.22 | −6.25–0.00 |
| AR R(2+1)D18 | H.265 | +1.14 | +1.08 | 0.00–0.16 | −3.125–0.00 |
| AR R3D18 | H.264 | −0.05 | −0.03 | 0.00–0.22 | 0.00–0.00 |
| AR R3D18 | H.265 | −0.06 | −0.06 | 0.00–0.16 | 0.00–0.00 |
| OD ResNet50 | H.264 | −4.31 | −4.06 | 0.52–4.21 | −1.76–+2.07 |
| OD ResNet50 | H.265 | −3.28 | −4.50 | 0.12–1.51 | −1.78–+3.03 |

BD-rate âm là giảm bitrate ở cùng chất lượng. Mức giảm byte ở cùng QP là phép đo khác, đã tính bitstream và header/grid ROI. Bpp chia cho pixel nguồn đã chuẩn hóa, không chia cho kích thước đã retarget.

AR chưa cải thiện trên evaluator độc lập: tiết kiệm thêm rất nhỏ, nhưng Top-1 giảm2/32 clip tại H.264 QP40 và1/32 clip tại H.265 QP40. R3D18 giữ Top-1; điều đó không đảm bảo R(2+1)D18 giữ chất lượng. Nhánh bổ sung được chọn4/320 trial, gồm3 simplification và1 hybrid retarget.

OD có tín hiệu BD-rate tốt hơn V30-B, nhưng chưa bảo toàn mAP ở mọi QP. Nhánh bổ sung được chọn83/320 trial:25 simplification,16 fixed retarget và42 hybrid retarget. Không tính mức tiết kiệm của candidate bị guard loại thành tiết kiệm của V32.

Đây là sàng lọc DEV32, bootstrap0, chưa có khoảng tin cậy; `target_confirmed=false`. Chưa đạt hay xác nhận mục tiêu−10%, chưa có tuyên bố trên tập TEST hoặc toàn bộ benchmark.

Kiểm chứng sau khi tải về: mỗi task đủ2240 hàng,320 audit; hash/kích thước/CRC và hình học của mọi selected packet hợp lệ;57 hash Python runtime khớp archive nguồn9039b0502852cc992e20e842b9b1d0c20e4b4a84. ID và hash pixel DEV không trùng TRAIN trong checkpoint. Giải mã độc lập thêm5 packet AR và4 packet OD trên máy local, gồm3 packet có grid ROI.

Mã lõi không thay đổi so với snapshot ab1021e đã chạy777 test pass. Phần mới chỉ là notebook thực thi một lần và tài liệu kết quả. Hai checkpoint cụ thể được người dùng cho phép gửi sang Kaggle riêng tư; payload không chứa credential.

Artifact trên máy: `../output/v32_ar_od_kaggle/KET_QUA_VI.md`, `rate_quality_curves.png`, `ar/points.csv`, `od/points.csv`, các comparison/inspection JSON và archive đầy đủ. Đường dẫn tuyệt đối: `D:/STUDY/LAB/bao_1/output/v32_ar_od_kaggle/`.
