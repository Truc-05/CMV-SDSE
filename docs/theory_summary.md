# Mô hình lý thuyết

## Đại lượng

Một item là **lỗi thật** (ground truth). Nó đi qua *k* cổng verifier nối tiếp. Cổng nào cũng có
thể **chấp nhận nhầm**. Item chỉ thoát nếu **mọi** cổng đều chấp nhận.

- `α` — xác suất một cổng chấp nhận nhầm một item lỗi cụ thể
- `S(k)` — xác suất item sống sót qua cả k cổng
- `reliability(k) = 1 − S(k)`

## Ba mô hình lồng nhau

**1. Cổng độc lập, đồng nhất.**

    S(k) = α^k
    log S(k) = k · log α          tuyến tính theo k

Mô hình null. Độ tin cậy tiến về 1 theo **hàm mũ**.

**2. Không đồng nhất, không trần.** `α ~ Beta(a, b)`.

    S(k) = E[α^k] = B(a+k, b) / B(a, b)
    S(k) ~ Γ(b) · k^(−b)          khi k lớn
    log S(k) ≈ const − b · log k  tuyến tính theo log k

Độ tin cậy **vẫn tiến về 1**, nhưng theo **đa thức**. Không có trần.

Tương quan nội cụm giữa các phán quyết trên cùng một item:

    ρ_v = 1 / (a + b + 1)

Đây đúng là tham số của **hiệu ứng thiết kế** trong thống kê phân cụm:

    DEFF = 1 + (k − 1) · ρ_v        n_eff = k / DEFF

`n_eff` là số cổng **độc lập tương đương**.

**3. Có trần thật.** Khối lượng `π₀` là **điểm mù**: mọi cổng đều bỏ sót, α = 1.

    S(k) = π₀ + (1 − π₀) · E[α^k]
    reliability(k) → 1 − π₀   khi k → ∞

Mô hình **duy nhất** có trần, và trần chỉ định danh được khi mô hình hoá tường minh như vậy.

## Hai phép kiểm

**Phân biệt dạng suy giảm** (`theory/concavity_test.py`). Khớp `log S(k)` theo hai trục — tuyến
tính theo `k` (mô hình 1) và theo `log k` (mô hình 2) — chọn bằng AIC, trọng số theo phương sai
nhị thức. Loại các điểm `S(k) = 0` và **báo cáo số điểm bị loại**.

`b̂` từ phép kiểm này chỉ để **phân biệt mô hình**, không để ước lượng: nó lệch thấp vì xấp xỉ
`k^(−b)` là tiệm cận. Giá trị `b` không lệch lấy từ MLE của phép kiểm trần.

**Kiểm trần** (`theory/ceiling_test.py`). Khớp mô hình 3 bằng hợp lý nhị thức đầy đủ, tham số hoá
`(logit π₀, log a, log b)`. So với mô hình 2 bằng tỷ số hợp lý.

Vì `π₀ = 0` nằm trên **biên**, phân phối null của LR là hỗn hợp 50:50 của χ²₀ và χ²₁ (Chernoff):

    p = 0.5 · P(χ²₁ > LR)

Khoảng tin cậy dùng profile likelihood, không dùng Wald.

## Lỗi đã sửa

Bản đầu khớp `π·(1 − e^(−λk))` cho kiểm trần — giả định hội tụ mũ tới tiệm cận dưới 1, trong khi
mô hình 2 hội tụ **đa thức tới đúng 1**. Trên dữ liệu sinh từ mô hình 2, nó cho `π̂ < 1` với
R² ≈ 0.95–0.99 và `π̂` **trượt theo cửa sổ k**: 0.71 ở k_max = 10 lên 0.95 ở k_max = 5000. Toàn
bộ "khối điểm mù" đo được là artifact của cửa sổ quan sát.

Bản đầu cũng lấy sai phân bậc hai của log-odds trên lưới k không đều. `np.diff` hai lần trên
khoảng chia không đều trộn độ giãn của lưới vào độ cong, và việc kẹp reliability ở `1 − 1e−6` tạo
đoạn phẳng giả khi bão hoà. Kết quả: phép kiểm báo "lõm" trên chính đường odds law, vốn tuyến
tính theo định nghĩa.

Cả hai phép kiểm cũ đều cho dương tính trên mô hình null của chính chúng.

## Bộ kiểm null

`tests/test_null_models.py` sinh dữ liệu từ mô hình 1 và 2 rồi khẳng định phép kiểm **không**
phát hiện gì, sinh từ mô hình 3 rồi khẳng định nó phát hiện đúng `π₀`, và kiểm `π₀` bất biến theo
cửa sổ k.

## Ước lượng ρ_v

Không lấy từ việc khớp Beta lên `α̂ = false_accepts / n_decided` của từng item. `α̂` là ước lượng
nhị thức có nhiễu nên độ tán của nó **lớn hơn** độ tán thật của α, kéo `ρ_v` lên giả tạo. Trong
kiểm tra tổng hợp, cách này cho ρ_v = 0.546 khi giá trị thật là 0.167.

Lấy `a`, `b` từ MLE nhị thức đầy đủ của phép kiểm trần rồi tính `ρ_v = 1/(a+b+1)`. Giá trị khớp
lên `α̂` vẫn báo cáo dưới tên `rho_v_naive_from_alpha_hat` để đối chiếu, và phải nêu rõ là chệch lên.

## Cổng không phân định được

Verdict không đọc được **không** được tính là chấp nhận. Backend ném `VerdictParseError`, chain
thử lại, vẫn hỏng thì cổng ghi `accepted = None` và bị loại khỏi mẫu số của α. Tỷ lệ này báo cáo
riêng trong `results/tables/indeterminate_gates.csv`; vượt 2% là dấu hiệu verifier không tuân thủ
định dạng và α đang dựa trên mẫu đã bị mỏng đi.
