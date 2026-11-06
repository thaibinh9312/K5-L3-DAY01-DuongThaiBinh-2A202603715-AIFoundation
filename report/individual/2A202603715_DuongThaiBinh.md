# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ✅ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✅ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ✅ |

## 2. Kết quả chính

### Bảng: % đoán ĐÚNG theo nhóm case (càng đậm càng tốt)

| Nhóm case | 1. BoW + LogReg | 2. BoW + MLP | 3a. Emb + KNN | 3b. Emb + LogReg | 3c. Emb + MLP head | 4a. LLM nhỏ zero-shot | 4d. LLM nhỏ + prompt phòng thủ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tưởng spam nhưng không phải | **100%** | 90% | 70% | 80% | 80% | **100%** | **100%** |
| Thao túng / lừa đảo | 12% | 38% | **100%** | 25% | 38% | 0% | 0% |
| Spam giống tự nhiên | 0% | 0% | 40% | 40% | 40% | 0% | 0% |
| Né từ khoá | 17% | 0% | **83%** | **83%** | 67% | 0% | 0% |
| Prompt injection | 67% | 83% | 83% | **100%** | **100%** | 17% | 17% |
| Nói về spam | **100%** | **100%** | 50% | 50% | 50% | **100%** | **100%** |
| Tin giao dịch hợp lệ | 83% | **100%** | 83% | 83% | 83% | **100%** | **100%** |
| **TỔNG (50 case)** | **50%** | **54%** | **72%** | **64%** | **64%** | **42%** | **42%** |

## 3. Dự đoán và thực tế

- Thí nghiệm 1.B: dự đoán loss dao động mạnh và hội tụ, nhưng thực tế lại giảm rất nhanh và hội tụ ở 0.004. Nguyên nhân là learning rate lớn khiến trọng số được cập nhật rất mạnh, làm loss tăng đột ngột.

## 4. Phân tích 50 case khó

- Mô hình tụt F1 nhiều nhất là BoW + MLP, từ 0.919 trên tập test thường xuống 0.378 trên 50 case khó, giảm 0.541. 2 case spam giọng tự nhiên và né từ khóa có tỉ lệ đoán đúng là 0%. Nguyên nhân là mô hình vẫn phụ thuộc vào đặc trưng Bag-of-Words nên gặp khó khăn với spam viết tự nhiên hoặc né từ khóa. Điểm test thường quá lạc quan vì dữ liệu có phân phối tương tự tập train, trong khi stress test chứa nhiều trường hợp bất thường và cố tình đánh lừa mô hình.

## 5. Thử thách cá nhân

🎯 Đề #1: ML · Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1. Báo precision/recall/F1 từng ngưỡng. Một ngân hàng sợ chặn nhầm tin OTP thì nên chọn ngưỡng nào? Vì sao?
- Code:
# Lấy xác suất spam từ mô hình Phần 1
X_te = vectorize(texts_te, vocab)
probs = predict_proba(X_te, w, b)

# So sánh 3 ngưỡng
results = []

for threshold in [0.2, 0.5, 0.8]:
    y_pred = (probs >= threshold).astype(int)
    m = metrics(y_te, y_pred)

    results.append({
        "Threshold": threshold,
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1": m["f1"]
    })

df = pd.DataFrame(results)
print(df.round(3).to_string(index=False))
- Kết quả + giải thích:
    - Khi thử ba ngưỡng quyết định, ngưỡng 0.2 cho kết quả tốt nhất với Precision = 0.874, Recall = 0.907 và F1 = 0.890. So với ngưỡng 0.5 (F1 = 0.856), việc giảm ngưỡng giúp phát hiện thêm nhiều tin spam nhưng cũng tăng nguy cơ chặn nhầm ham. Ngưỡng 0.8 có Precision = 1.000 nhưng Recall chỉ còn 0.617, khiến F1 giảm xuống 0.763. Vì vậy, trong ba ngưỡng thử nghiệm, tôi chọn 0.2 vì có F1 cao nhất và cân bằng Precision - Recall tốt hơn.
    - Ngân hàng nên ưu tiên ngưỡng có Precision cao để tránh chặn nhầm tin OTP hợp lệ (False Positive). Trong kết quả thí nghiệm, cả ngưỡng 0.5 và 0.8 đều đạt Precision = 100%, nhưng ngưỡng 0.5 có Recall cao hơn (0.748 so với 0.617). Vì vậy, nên chọn ngưỡng 0.5 vì vừa không chặn nhầm ham trên tập test, vừa phát hiện được nhiều spam hơn. Khi triển khai thực tế, cần kiểm tra riêng trên dữ liệu OTP và đánh giá tỷ lệ False Positive trước khi quyết định.

## 6. Phương án cho sản phẩm thật

- Với 10 triệu SMS/ngày (khoảng 116 tin/giây) và SLA p95 dưới 50 ms, cả 5 mô hình BoW và Embedding đều đạt yêu cầu độ trễ trong thí nghiệm. Tôi chọn Encoder (MiniLM) + MLP Head vì có F1 cao nhất trên test thường (0.942), p95 = 7.71 ms và throughput theo lô đạt 1.883 tin/giây. BoW + MLP cũng là lựa chọn tốt nếu ưu tiên tốc độ và tiết kiệm tài nguyên, với F1 = 0.930. Nếu sử dụng LLM cho toàn bộ SMS, chi phí ước tính theo bảng giá bài học và giả định 100 input tokens, 1 output token mỗi tin là khoảng 3.150 USD/tháng cho model nhỏ hoặc 31.500 USD/tháng cho model lớn. LLM còn có rủi ro về độ trễ, hạn mức API và chi phí vận hành

## 7. AI log và điều đã học

- Tôi sử dụng AI để hỗ trợ code phần 4, đồng thời kiểm tra các lỗi logic trong code. Tôi cũng học được cách sử dụng các mô hình ML, DL, Embedding và LLM để lọc tin nhắn rác, cách đánh giá hiệu suất của các mô hình và cách tối ưu hóa ngưỡng quyết định để cân bằng giữa Precision và Recall.
