# Đăng ký trước

Trạng thái: **bản nháp, chưa đóng băng.** Không chạy sweep chính trước khi mục 7 hoàn tất.

## 1. Câu hỏi

Một chuỗi *k* cổng verifier có bắt được lỗi tốt hơn một cổng không, và tốt hơn bao nhiêu so với
điều một chuỗi cổng **độc lập** sẽ đạt được.

## 2. Giả thuyết

**H1 (không độc lập).** Các cổng trong cùng một bể không độc lập. Dạng suy giảm ưa thích là
`power_in_log_k` chứ không phải `exponential_in_k`, với ΔAIC ≥ 2.

**H2 (bậc thang phụ thuộc).** `ρ_v` giảm đơn điệu qua ba bể: `same_model` ≥ `same_family` ≥
`cross_family`. Hiệu số giữa bể đầu và bể cuối là phần tương quan **do dùng chung trọng số**.

**H3 (trần).** Tồn tại khối điểm mù `π₀ > 0` với p < 0.05 theo hỗn hợp Chernoff, và cận dưới
profile CI lớn hơn 0.

**H4 (số cổng hữu hiệu).** Ở k = 50, `n_eff` nhỏ hơn 10 trong bể `same_model`.

**H5 (thăm dò).** Tồn tại `k†` mà sau đó độ tin cậy giảm, do báo động giả tích luỹ. Không dùng
để bác bỏ bất cứ điều gì.

## 3. Ngưỡng đã chốt

Trong `configs/cascade_thresholds.yaml`.

- `k_max` 50, `k_pilot` 5
- `min_gates_for_ceiling_fit` 10 — không tuyên bố trần nếu k quan sát chưa tới 10
- `n_items_per_failure_mode` 40
- mức ý nghĩa 0.05, CI 95% bằng profile likelihood
- ΔAIC ngưỡng 2.0 cho chọn mô hình suy giảm

Không sửa sau khi nhìn kết quả. Nếu buộc phải sửa, ghi lý do và ngày, và báo cáo cả kết quả theo
ngưỡng cũ.

## 4. Mẫu số và loại trừ

Mẫu số của `α` là số cổng **đã phân định**, không phải số cổng đã chạy. Cổng có verdict không đọc
được ghi `accepted = None` và bị loại. Tỷ lệ loại trừ báo cáo bắt buộc.

Nếu tỷ lệ không phân định vượt 2% ở một điều kiện, điều kiện đó bị đánh dấu không diễn giải được
cho tới khi nguyên nhân được điều tra.

## 5. Nguồn item và nhiễu nhãn

Nguồn chính là MAST-Data. Nhãn phần lớn do LLM gán, κ = 0.77 so với chuyên gia. Tập vàng người
gán chỉ có 19 trace, trải trên bốn vòng IAA dùng các bản nháp taxonomy **khác nhau**, nên không
được gộp qua vòng mà không kiểm `ids_with_multiple_titles`.

Đồng thuận tuyệt đối trên các dòng **có nghĩa** — có ít nhất một phiếu dương — là 0.750. Đó là
trần thật của chất lượng nhãn, và mọi ước lượng `α`, `ρ_v`, `π₀` không thể chính xác hơn ngưỡng
đó. Phải báo cáo phân tích độ nhạy nhiễu nhãn, đặc biệt cho mode 3.2 (đồng thuận có nghĩa 0.333).

Nguồn phụ là testbed `minimal-mas-failure-modes`: ground truth cơ học, không phụ thuộc annotator.
Dùng làm kiểm tra vững. Chỗ hai nguồn cho kết quả khác nhau là chỗ đáng báo cáo nhất.

## 6. Cắt transcript

Transcript dài hơn ngưỡng bị cắt **ở giữa**, giữ đầu và cuối. Với nhóm FC3 đây là lựa chọn có lý
do: dừng non, không kiểm chứng, kiểm chứng sai — bằng chứng của cả ba nằm ở cuối trace.

Ngưỡng cắt suy từ `num_ctx` chứ không đặt cứng. Tỷ lệ trace bị cắt phải báo cáo, và phải có phân
tích độ nhạy chạy cùng tập item ở hai ngưỡng cắt khác nhau.

## 7. Danh mục phải hoàn tất trước khi đóng băng

- [ ] Pilot chạy sạch: unparsed dưới 5%, indeterminate dưới 2%, α không suy biến
- [ ] Bộ kiểm null pass toàn bộ
- [ ] Rà soát tính mới hoàn tất theo `docs/related_work.md`
- [ ] Prompt verifier khoá lại cho từng target
- [ ] Ngày đóng băng ghi vào đây

Ngày đóng băng: chưa đặt.

## 7b. Nhật ký thay đổi phương pháp (ghi theo quy tắc mục 3)

**2026-08-07 — chuyển suy diễn từ đường cong sang mức-item.** Lý do: cả `concavity_test` và
`ceiling_test` cộng hợp lý nhị thức theo từng `k` của đường cong `reliability(k)` tích luỹ,
đếm trùng thông tin ~`k` lần (một item đo qua `k` cổng bị tính như `k` item). Thêm nữa
`concavity_test` khớp OLS trên log-survival và cho kết luận H1 **ngược** (báo cổng độc lập trên
dữ liệu có `ρ_v≈0.65`). Thay bằng MLE Beta-Binomial hỗn hợp-trần ở mức item
(`theory/betabinom_mixture.py`), thống nhất cả ba dự đoán; LRT biên dùng hỗn hợp Chernoff, CI
dùng profile. Không ngưỡng nào ở mục 3 bị nới; `min_gates=10`, α=0.05, ΔAIC=2 giữ nguyên.
Estimator được kiểm định bằng mô phỏng ở đúng `(n,K)` (xem `docs/methods_upgrade.md`,
`results/theory_fits/estimator_calibration.csv`): size đúng, coverage 95%, và mạnh hơn phép
kiểm cũ. Các phép kiểm đường cong cũ giữ lại làm comparator (đã chặn lỗi overflow concentration).

## 8. Kết quả âm

Nếu `π₀` không phân biệt được với 0, đó là kết quả được báo cáo đầy đủ: chuỗi verifier không có
trần trong dải k đã quan sát, và độ tin cậy tiếp tục tăng — chỉ là chậm. Đó vẫn là phát biểu có
giá trị về chi phí của việc thêm cổng.
