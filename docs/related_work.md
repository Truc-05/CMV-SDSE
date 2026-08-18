# Rà soát tính mới

Trạng thái: **chưa hoàn tất.** Ghi lại những gì đã đối chiếu và, quan trọng hơn, những gì còn phải kiểm.

## Nền tảng

**MAST — Why Do Multi-Agent LLM Systems Fail?** arXiv:2503.13657, Cemri et al., UC Berkeley,
NeurIPS 2025 Datasets and Benchmarks Track.

Đã xác minh trực tiếp từ phần chính bản v3:

- tỷ lệ thất bại 41% đến 86.7% trên 7 hệ MAS mã nguồn mở
- 150 trace phân tích Grounded Theory, 6 chuyên gia, hơn 20 giờ mỗi người
- κ = 0.88 giữa người, κ = 0.77 cho LLM annotator, κ = 0.79 khi tổng quát hoá
- 14 mode, can thiệp đạt cải thiện tối đa 15.6%

Chưa đọc Appendix A–N.

## Sai lệch giữa paper và dữ liệu phát hành

Đối chiếu trực tiếp `MAD_full_dataset.json`:

| | paper | dữ liệu |
|---|---|---|
| số trace | 1642 | **1242** |
| trace người gán | 21 | **19** |
| FM-2.5 | 1.90% | **0 trên 1242** |
| FM-2.2 | 6.80% | 9.9% |

**Phần trăm trong paper là tỷ trọng trên tổng số cờ, không phải prevalence.** 1242 trace mang
3064 lượt gắn cờ, tức 2.47 mode mỗi trace. Prevalence thật cao gấp 2–3 lần con số paper — FM-1.3
là 36.3% trace chứ không phải 15.7%, FM-2.6 là 39.9%.

Con số 2.47 cờ mỗi trace là bằng chứng ngoài, quy mô lớn, cho việc **các mode đồng xuất hiện**
chứ không tách rời.

## Tập vàng người gán không nhất quán

19 trace trải trên bốn vòng: Round 1, Round 2, Round 3, và một vòng tổng quát hoá. Mỗi vòng dùng
một bản nháp taxonomy khác. Kết quả:

- 7 mã mode không thuộc taxonomy 14 mode cuối: 1.6, 1.7, 2.7, 3.4, 4.1, 4.2, 4.3
- **mọi** mã từ 1.1 đến 3.3 mang 2–4 định nghĩa khác nhau; mã 2.4 mang bốn

Ai dùng tệp này làm gold mà không lọc theo `round` là đang gộp các nhãn không tương thích. Đây là
phát hiện về hạ tầng dữ liệu của cả lĩnh vực, không riêng dự án này.

## Việc rà soát còn phải làm

1. Danh sách công trình trích dẫn MAST trên Semantic Scholar, lọc thủ công tìm bất kỳ ai đã đo
   độ tin cậy của chuỗi verifier nối tiếp
2. Tra riêng "design effect", "effective sample size", "intraclass correlation" trong ngữ cảnh
   ensemble hoặc verifier LLM. Nếu đã có người dùng, đóng góp phương pháp mất
3. Tra dòng nghiên cứu cổ điển về định lý Condorcet với phiếu tương quan — Ladha, Boland, Berg —
   xác minh tên, năm và kết quả tiệm cận chính xác
4. Tra xem có ai mô hình hoá khối điểm mù như một hỗn hợp có thể định danh, hay tất cả đều khớp
   đường bão hoà rồi gọi tiệm cận là trần

Mục 4 quan trọng nhất: nếu lỗi khớp sai dạng hàm mà dự án này vừa sửa cũng tồn tại trong công
trình đã công bố, thì bản thân việc chỉ ra nó là một đóng góp.

## Kết quả rà soát tính mới (2026-08-07 — đã xác minh trên web)

Các mục ở phần "việc còn phải làm" phía dưới đã tra xong. Kết luận **thu hẹp đáng kể** tuyên
bố novelty. Phải đọc kỹ trước khi viết phần đóng góp.

**Bài lý thuyết đang kiểm định — Han (2026), arXiv:2607.13918** ("Partially Correlated Verifier
Cascades in LLM Harnesses"). *Cảnh báo trích dẫn:* README/docs dự án gọi là **"Aksu 2026b"**
nhưng tác giả arXiv là **Jiangang Han**, nộp 15/07/2026 — phải đối chiếu: hoặc là bài của chính
nhóm (đang tự kiểm định lý thuyết mình — cần công khai), hoặc nhãn trích dẫn sai. Abstract nêu:
concave log-odds, `1−r_k ≍ k^{−b}`, `ρ_v = 1/(a+b+1)`, khối điểm mù `1−π` chặn bằng chứng ở
`−ln(1−π)` nats, tam phân với `k†` dạng đóng. Quan trọng cho novelty: bài **tự nói** "the theory
is measurable: … **beta-binomial likelihood and NPMLE recover the reliability curve and the
ill-posed ceiling**", và "two verdicts identify `ρ_v`".

⟹ **Estimator Beta-Binomial mức-item / NPMLE / `ρ_v` KHÔNG phải đóng góp phương pháp của dự
án — chính bài gốc đã kê đơn nó.** Dự án là bên *hiện thực hoá và kiểm định*, không phải bên
phát minh. Bài gốc **chỉ có synthetic test, không có thực nghiệm MAS thật, không nhắc MAST.**

**Design effect / ICC cho đánh giá LLM đã tồn tại.** DiagnosticIQ (arXiv:2605.08614) tính ICC
theo cụm-rule cho benchmark LLM (ICC 0.87–0.91, DEFF ~40). Beta-Binomial cho overdispersion của
LLM-judge cũng đã có (ngưỡng ρ 0.01/0.10). ⟹ khung "n_eff cho ensemble LLM" **không mới về khái
niệm**; cái còn mới là *áp cho cascade nối tiếp gate-level trên MAS thật*.

**Condorcet với phiếu tương quan** (đã xác minh tên/năm): Ladha (1992, 1995), Berg (1993a,b),
Boland (1989) — phiếu tương quan qua urn hypergeometric/Pólya; hiệu quả majority-vote giảm khi
tương quan tăng (SEP "Jury Theorems"). Đây là *song song* (majority vote), khác *nối tiếp*
(AND-cascade) của ta — điểm phân biệt hợp lệ, nên nêu để định vị.

**LLM gần đây, cùng chủ đề:** "Consensus is Not Verification" (arXiv:2603.06612) — khi lỗi LLM
tương quan, không luật *đồng thuận song song* nào scale được truthfulness nếu thiếu verifier
ngoài. Rất gần thông điệp của ta nhưng là *song song*; phải trích và phân biệt rõ với *nối tiếp*.

**Novelty còn đứng vững (thu hẹp, trung thực):**
1. **Thực nghiệm thật đầu tiên** cho lý thuyết 2607.13918 trên MAS thật (bài gốc chỉ synthetic).
2. **Nối MAST**: gắn khối điểm mù đo được với base rate FC3 (23.5%) quan sát — chưa ai làm.
3. Phát hiện **đối xứng miss/false-alarm** đều nội tại theo item (ρ_v và ρ_v^FA đều cao).
4. **Calibration hữu hạn mẫu** của estimator (bias/coverage/power ở n=40,k=50) — thao tác hoá
   "đo được" của bài gốc thành khẳng định về thiết kế đủ power; đóng góp *khiêm tốn*.

**Rủi ro còn lại:** H2 (bậc thang phụ thuộc) hiện chưa có dữ liệu → đóng góp thực nghiệm hiện
mỏng (pilot một điều kiện). Phải chạy full sweep mới đủ sức cho (1)–(3).

## Ghi chú lịch sử

Hai hướng đã bị loại sau khi rà soát cho thấy đã có người làm: liên hệ theory of mind với lỗi
phối hợp, và mô hình hoá cascade lỗi. Cả hai lúc đầu đều được cho là còn trống.

Bài học: giả định tính mới mà không rà soát là sai lầm đắt nhất ở giai đoạn này.
