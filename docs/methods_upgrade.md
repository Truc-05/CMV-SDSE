# Nâng cấp phương pháp: từ khớp đường cong sang hợp lý mức-item

Ngày: 2026-08-07. Trạng thái: đã triển khai, kiểm định bằng mô phỏng, mọi test pass.

Tài liệu này ghi lại lỗi suy diễn trong bản đầu, bản sửa, và bằng chứng mô phỏng cho bản
sửa. Nó là nền cho phần Method + phần Estimator Validation của bài.

## 1. Lỗi: đếm trùng thông tin trên đường cong tích luỹ

Bản đầu suy diễn ba dự đoán từ **đường cong `reliability(k)`** — tỷ lệ lỗi bị bắt bởi cổng
`≤ k`. Cả `concavity_test` (H1) lẫn `ceiling_test` (H3) **cộng một hợp lý nhị thức theo từng
`k`**, với mẫu số `n` (số item) ở *mỗi* `k`.

Nhưng mọi item đều đi qua cả `k` cổng, nên `k` điểm của đường cong đến từ **cùng một tập
item** và gần như tự tương quan hoàn toàn (đường cong đơn điệu, tích luỹ). Cộng hợp lý nhị
thức theo `k` coi một item đo qua `k` cổng như `k` item độc lập, **thổi cỡ mẫu hữu hiệu lên
~`k` lần**. Trên pilot `k=20`, hợp lý "nghĩ" nó có `29 × 20 = 580` quan sát độc lập, trong
khi thực chất chỉ có `29` item. Thống kê tỷ số hợp lý và khoảng profile dựng trên nền đó là
**quá tự tin**.

Thêm một lỗi riêng ở H1: `concavity_test` khớp `log S(k)` bằng **OLS trên trục biến đổi**
(tuyến tính theo `k` vs theo `log k`), trọng số nhị thức tính sai cho phép biến đổi log. Trên
chính pilot, nó chọn `exponential_in_k` (ΔAIC −15) và **kết luận các cổng độc lập** — mâu
thuẫn với `ρ_v ≈ 0.64` mà `ceiling_test` cùng run suy ra từ dữ liệu đó. Một pipeline tự mâu
thuẫn.

## 2. Bản sửa: mô hình ở mức dữ liệu được sinh ra

Theo giả định trao đổi được (exchangeability, biểu diễn de Finetti của Aksu 2026b), `k` phán
quyết trên một item là i.i.d. có điều kiện cho một `α_i` tiềm ẩn của item. Vậy **thứ tự phán
quyết không mang thông tin**, và thống kê đủ của mỗi item là cặp `(mᵢ, Kᵢ)` — số phán quyết
**không mong muốn** trong `Kᵢ` cổng đã phân định (false-accept với item lỗi; false-alarm với
item sạch). Ba mô hình lồng nhau của lý thuyết trở thành, ở mức item:

    M1  Binomial(K, μ)                           cổng độc lập, đồng nhất: S(k)=μ^k
    M2  Beta-Binomial(K, a, b)                   dị chất, không trần; ρ_v = 1/(a+b+1)
    M3  π₀·1[m=K] + (1−π₀)·BetaBinom(K,a,b)      khối điểm mù / sàn false-alarm π₀

Mỗi item đóng góp **đúng một** số hạng hợp lý, nên cỡ mẫu hữu hiệu là **số item**, không phải
item × cổng. Tương quan nội-item của phán quyết **chính là** tương quan Beta-Binomial
`ρ_v = 1/(a+b+1)`, cũng là tham số hiệu ứng thiết kế, nên cả H1–H4 đọc ra từ **một** lần khớp:

- **H1** (không độc lập / lõm): LRT M2 vs M1 (overdispersion, `ρ_v > 0`)
- **H2** (bậc thang phụ thuộc): so `ρ_v` giữa các bể
- **H3** (trần): LRT M3 vs M2 (khối thừa tại `m=K`)
- **H4** (số cổng hữu hiệu): `n_eff(K) = K / (1 + (K−1)ρ_v)`

Cả hai LRT kiểm tham số **trên biên** (`ρ_v=0` là `a+b→∞`; `π₀=0` là mép `[0,1]`), nên phân
phối null là hỗn hợp 50:50 của điểm khối tại 0 và `χ²₁` (Chernoff 1954; Self & Liang 1987):
`p = 0.5·P(χ²₁ > LR)`. Khoảng tin cậy dùng profile likelihood, không dùng Wald.

Triển khai trong `theory/betabinom_mixture.py`. Tham số hoá qua `(logit μ, log(a+b))` (tách
trung bình khỏi độ tập trung) — điều kiện số tốt hơn `(log a, log b)` nhiều. Độ tập trung bị
chặn ở `a+b ≤ 10⁶` (`ρ_v ≥ ~10⁻⁶`): quá ngưỡng đó `betaln` mất hết độ chính xác và hợp lý suy
biến thành "xác suất 1 cho mọi item" — một lỗ hổng số mà optimizer sẽ khai thác thành **trần
giả**. Lỗi overflow này cũng tồn tại trong `ceiling_test` cũ và đã được chặn ở đó.

## 3. Bằng chứng mô phỏng (`theory/simulation.py`)

Mô phỏng cascade từ M1/M2/M3 với `(π₀, a, b)` đã biết, ở đúng thiết kế pilot `(n=34, K=20)` và
thiết kế full-sweep mục tiêu `(n=40, K=50)`; 400 lần lặp cho size/power, 120 cho coverage.
Kết quả trong `results/theory_fits/estimator_calibration.csv`, hình `fig2`.

| Chỉ số | Pilot 34×20 | Target 40×50 |
|---|---|---|
| H1 size (null Binomial) | 0.025 | 0.035 |
| H1 power (`ρ_v ≥ 0.2`) | 1.00 | 1.00 |
| **H3 power (trần `π₀=0.15`) — item-level** | **0.42** | **0.70** |
| H3 power (trần) — đường cong cũ | 0.06 | 0.44 |
| bias `ρ_v` (không trần) | ≈ 0.00 | ≈ −0.01 |
| coverage CI `ρ_v` (95% danh nghĩa) | 0.95 | 0.93 |
| coverage CI `π₀` | 0.96 | 0.95 |

Kết luận: estimator mới **đúng size, phủ CI đúng 95%, và mạnh hơn hẳn** phép kiểm đường cong
(bắt trần thật 0.42 vs 0.06 ở cỡ pilot). Quan trọng cho định hướng "giữ mục tiêu đầy đủ":
thiết kế `(40, 50)` **đủ power** cho cả H1 lẫn H3, trong khi pilot `(34, 20)` *đúng là*
thiếu power cho trần — nên `π₀` CI rộng `[0, 0.24]` ở pilot là trung thực, không phải kết quả
âm chắc chắn.

Lưu ý bias: khi có trần thật mà khớp M2 (bỏ qua trần), `ρ_v` bị chệch lên ~0.12 (M2 nuốt khối
trần vào độ tán). Vì vậy coverage CI của `ρ_v` chỉ có nghĩa ở kịch bản không trần; khi phát
hiện trần, đọc độ tán từ M3.

## 4. Kết quả pilot dưới hai phương pháp (đối đầu trực tiếp)

`FM-3.3 / same_model`, `n=29` item lỗi dùng được (5 item bị loại vì **toàn bộ 20 cổng
indeterminate**), `K=20`:

| | CŨ (đường cong) | MỚI (item-level) |
|---|---|---|
| H1 dạng suy giảm | `exponential` (ΔAIC −15) → **độc lập** | `power` (ΔAIC **+420**), `ρ_v`=**0.65** [0.52, 0.77], p=4e-94 |
| H3 trần `π₀` | 0.00, CI[0, 0.32], không phát hiện | 0.02, CI[0, **0.24**], không phát hiện |
| H4 `n_eff`@20 | — | **1.49** (DEFF 13.4) |

Phương pháp cũ **kết luận ngược** ở H1. Với H3 cả hai thống nhất "chưa thấy trần" — không claim
CI cũ hẹp giả (dữ liệu không ủng hộ), bản mới chỉ chặt và nhất quán hơn.

**Phát hiện đối xứng mới** (từ cùng bộ máy áp lên item sạch): false-alarm cũng dị chất mạnh —
`ρ_v^FA = 0.83`, và **sàn "luôn bị cờ" = 0.35** (35% item sạch bị *mọi* cổng cờ). Cả miss lẫn
false-alarm đều **nội tại theo item**, không phải nhiễu theo cổng. Đây là câu chuyện thống nhất:
thêm cổng tương quan gần như không gỡ được cả hai loại lỗi (`n_eff` ≈ 1.5 trên 20 cổng).

## 5. Nối MAST và tam phân (`fig3`, `net_utility_by_cost.csv`)

`net_utility_cost_frontier` quét chi phí false-alarm và, khi cho base rate FC3 (π=0.235) làm
trọng số hiện lưu (prevalence), báo độ sâu cascade tối ưu `k*`. Ở base rate MAST, cổng chỉ
"giúp" khi false-alarm gần như miễn phí (chi phí ≤ 0.1); từ 0.25 trở lên `k*=1`. Nghĩa là các
khung MAST thực tế vận hành **ở hoặc trên `k†`** trừ khi false-alarm rất rẻ — tam phân không
chỉ tồn tại mà điểm vận hành thực tế nằm ở phía "thêm cổng gây hại".

## 6. Việc còn lại cho bản full (trên cluster)

- Chạy `same_family` và `cross_family` (đã cấu hình turnkey với model có sẵn; đổi sang
  `cluster_pools` để `same_family` thành thang kích thước qwen thuần — cần pull qwen2.5:3b/14b)
  để kiểm **H2** (hiện chưa có dữ liệu — chỉ `same_model` được chạy).
- `k` tới 50 và `n=40` mỗi mode để chạm power mục tiêu ở §3.
- Phân tích độ nhạy nhiễu nhãn (đặc biệt mode 3.2) và độ nhạy cắt transcript, theo prereg §5–6.
